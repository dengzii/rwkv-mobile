#if !__has_feature(objc_arc)
#error This file must be compiled with automatic reference counting enabled (-fobjc-arc)
#endif

#import "rwkv-coreml.h"
#import "rwkv-coreml-impl.h"
#import "rwkv-coreml-stateful-impl.h"

#import <CoreML/CoreML.h>

#include <stdlib.h>
#include <cstdio>
#include <vector>
#include "half.hpp"

#if __cplusplus
extern "C" {
#endif

struct rwkv_coreml_context {
    const void * data;
    int n_layers;
    int num_heads;
    int head_dim;
    int embd_dim;
    int vocab_size;

    bool is_stateful;

    MLMultiArray * state_wkv = NULL;
    MLMultiArray * state_tokenshift = NULL;

    rwkv_coreml_stateful_implState * state = NULL;

    std::vector<float> logits;
};

NSArray<NSNumber *> * get_shape_by_name(NSDictionary *model_inputs, NSString *name) {
    MLFeatureDescription *desc = model_inputs[name];
    if (desc.type == MLFeatureTypeMultiArray) {
        return desc.multiArrayConstraint.shape;
    }
    return nil;
}

struct rwkv_coreml_context * rwkv_coreml_init(const char * path_model) {
    NSString * path_model_str = [[NSString alloc] initWithUTF8String:path_model];

    NSURL * url_model = [NSURL fileURLWithPath: path_model_str];

    // select which device to run the Core ML model on
    MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
    // config.computeUnits = MLComputeUnitsCPUAndGPU;
    // config.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
    config.computeUnits = MLComputeUnitsAll;

    NSError *error = nil;

    MLModel *mlmodel = [MLModel modelWithContentsOfURL:url_model configuration:config error:&error];

    if (error || !mlmodel) {
        NSLog(@"Error loading model: %@", error);
        return NULL;
    }

    rwkv_coreml_context * ctx = new rwkv_coreml_context;

    NSDictionary *model_inputs = mlmodel.modelDescription.inputDescriptionsByName;
    NSDictionary *model_outputs = mlmodel.modelDescription.outputDescriptionsByName;
    int num_inputs = mlmodel.modelDescription.inputDescriptionsByName.count;

    ctx->is_stateful = num_inputs == 1;

    if (!ctx->is_stateful) {
        NSArray<NSNumber *> *state_wkv_in_shape = get_shape_by_name(model_inputs, @"state_wkv_in");
        if (state_wkv_in_shape == nil) {
            return NULL;
        }

        ctx->n_layers = [state_wkv_in_shape[0] intValue];
        ctx->num_heads = [state_wkv_in_shape[1] intValue];
        ctx->head_dim = [state_wkv_in_shape[2] intValue];
        ctx->embd_dim = ctx->head_dim * ctx->num_heads;

        rwkv_coreml_impl * model = [[rwkv_coreml_impl alloc] initWithMLModel:mlmodel];
        ctx->data = CFBridgingRetain(model);
    } else {
        rwkv_coreml_stateful_impl * model = [[rwkv_coreml_stateful_impl alloc] initWithMLModel:mlmodel];
        ctx->data = CFBridgingRetain(model);

        ctx->state = [model newState];
        __block MLMultiArray *state_wkv;
        [ctx->state getMultiArrayForState:rwkv_coreml_stateful_implStateNameState_wkv handler:^(MLMultiArray *buffer) {
            state_wkv = buffer;
        }];
        NSArray<NSNumber *> *state_wkv_shape = state_wkv.shape;
        if (state_wkv_shape == nil) {
            return NULL;
        }
        ctx->n_layers = [state_wkv_shape[0] intValue];
        ctx->num_heads = [state_wkv_shape[1] intValue];
        ctx->head_dim = [state_wkv_shape[2] intValue];
        ctx->embd_dim = ctx->head_dim * ctx->num_heads;
    }
    if (ctx->data == NULL) {
        return NULL;
    }

    NSArray<NSNumber *> *logits_out_shape = get_shape_by_name(model_outputs, @"logits");
    if (logits_out_shape == nil) {
        return NULL;
    }
    ctx->vocab_size = [logits_out_shape[2] intValue];

    printf("num_heads: %d, head_dim: %d, vocab_size: %d, n_layers: %d\n", ctx->num_heads, ctx->head_dim, ctx->vocab_size, ctx->n_layers);

    ctx->logits = std::vector<float>(ctx->vocab_size, 0.0f);

    return ctx;
}

void rwkv_coreml_free(struct rwkv_coreml_context * ctx) {
    if (ctx) {
        if (ctx->data) {
            CFRelease(ctx->data);
        }
        delete ctx;
    }
}

void rwkv_coreml_decode(struct rwkv_coreml_context * ctx, int token) {
    float token_float = (float)token;
    MLMultiArray * inMultiArray = [
        [MLMultiArray alloc] initWithDataPointer: &token
                                           shape: @[@1, @(1)]
                                        dataType: MLMultiArrayDataTypeInt32
                                         strides: @[@(1), @(1)]
                                     deallocator: nil
                                           error: nil
    ];

    if (!ctx->is_stateful) {
        if (!ctx->state_tokenshift) {
            ctx->state_tokenshift = [
                [MLMultiArray alloc] initWithShape: @[@1, @(2 * ctx->n_layers), @(ctx->embd_dim)]
                                            dataType: MLMultiArrayDataTypeFloat16
                                            error: nil
            ];
            memset(ctx->state_tokenshift.dataPointer, 0, ctx->state_tokenshift.count * sizeof(uint16_t));
        }

        if (!ctx->state_wkv) {
            ctx->state_wkv = [
                [MLMultiArray alloc] initWithShape: @[@(ctx->n_layers), @(ctx->num_heads), @(ctx->head_dim), @(ctx->head_dim)]
                                            dataType: MLMultiArrayDataTypeFloat16
                                            error: nil
            ];
            memset(ctx->state_wkv.dataPointer, 0, ctx->state_wkv.count * sizeof(uint16_t));
        }

        @autoreleasepool {
            rwkv_coreml_implOutput * outCoreML = [(__bridge id) ctx->data predictionFromIn0: inMultiArray state_tokenshift_in: ctx->state_tokenshift state_wkv_in: ctx->state_wkv error: nil];
            ctx->state_tokenshift = outCoreML.state_tokenshift_out;
            ctx->state_wkv = outCoreML.state_wkv_out;
            for (int i = 0; i < ctx->vocab_size; i++) {
                ctx->logits[i] = ((__fp16 *)outCoreML.logits.dataPointer)[i];
            }
        }
    } else {
        if (!ctx->state) {
            ctx->state = [(__bridge rwkv_coreml_stateful_impl *) ctx->data newState];
        }

        @autoreleasepool {
            rwkv_coreml_stateful_implOutput * outCoreML = [(__bridge id) ctx->data predictionFromIn0: inMultiArray usingState: ctx->state error: nil];
            for (int i = 0; i < ctx->vocab_size; i++) {
                ctx->logits[i] = ((__fp16 *)outCoreML.logits.dataPointer)[i];
            }
        }
    }
}

float * rwkv_coreml_get_logits(struct rwkv_coreml_context * ctx) {
    return ctx->logits.data();
}

int rwkv_coreml_get_vocab_size(struct rwkv_coreml_context * ctx) {
    return ctx->vocab_size;
}

int rwkv_coreml_get_n_layers(struct rwkv_coreml_context * ctx) {
    return ctx->n_layers;
}

int rwkv_coreml_get_num_heads(struct rwkv_coreml_context * ctx) {
    return ctx->num_heads;
}

int rwkv_coreml_get_head_dim(struct rwkv_coreml_context * ctx) {
    return ctx->head_dim;
}

int rwkv_coreml_get_hidden_dim(struct rwkv_coreml_context * ctx) {
    return ctx->embd_dim;
}

#if __cplusplus
}
#endif