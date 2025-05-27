#if !__has_feature(objc_arc)
#error This file must be compiled with automatic reference counting enabled (-fobjc-arc)
#endif

#import "rwkv-coreml.h"
#import "rwkv-coreml-impl.h"

#import <CoreML/CoreML.h>

#include <stdlib.h>

#if __cplusplus
extern "C" {
#endif

struct rwkv_coreml_context {
    const void * data;
};

struct rwkv_coreml_context * rwkv_coreml_init(const char * path_model) {
    NSString * path_model_str = [[NSString alloc] initWithUTF8String:path_model];

    NSURL * url_model = [NSURL fileURLWithPath: path_model_str];

    // select which device to run the Core ML model on
    MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
    // config.computeUnits = MLComputeUnitsCPUAndGPU;
    config.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
    // config.computeUnits = MLComputeUnitsAll;

    const void * data = CFBridgingRetain([[rwkv_coreml_impl alloc] initWithContentsOfURL:url_model configuration:config error:nil]);

    if (data == NULL) {
        return NULL;
    }

    rwkv_coreml_context * ctx = new rwkv_coreml_context;

    ctx->data = data;

    return ctx;
}

void rwkv_coreml_free(struct rwkv_coreml_context * ctx) {
    CFRelease(ctx->data);
    delete ctx;
}

void rwkv_coreml_decode(
        const rwkv_coreml_context * ctx,
                             int64_t   n_ctx,
                             int64_t   n_tokens,
                             int64_t * tokens,
                               float * out) {
    MLMultiArray * inMultiArray = [
        [MLMultiArray alloc] initWithDataPointer: tokens
                                           shape: @[@1, @(n_mel), @(n_ctx)]
                                        dataType: MLMultiArrayDataTypeFloat32
                                         strides: @[@(n_ctx*n_mel), @(n_ctx), @1]
                                     deallocator: nil
                                           error: nil
    ];

    @autoreleasepool {
        // rwkv_coreml_implOutput * outCoreML = [(__bridge id) ctx->data predictionFromLogmel_data:inMultiArray error:nil];

        // memcpy(out, outCoreML.output.dataPointer, outCoreML.output.count * sizeof(float));
    }
}

#if __cplusplus
}
#endif