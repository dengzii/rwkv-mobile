//
// rwkv-coreml-impl.m
//
// This file was automatically generated and should not be edited.
//

#if !__has_feature(objc_arc)
#error This file must be compiled with automatic reference counting enabled (-fobjc-arc)
#endif

#import "rwkv-coreml-impl.h"

@implementation rwkv_coreml_implInput

- (instancetype)initWithIn0:(MLMultiArray *)in0 state_tokenshift_in:(MLMultiArray *)state_tokenshift_in state_wkv_in:(MLMultiArray *)state_wkv_in {
    self = [super init];
    if (self) {
        _in0 = in0;
        _state_tokenshift_in = state_tokenshift_in;
        _state_wkv_in = state_wkv_in;
    }
    return self;
}

- (NSSet<NSString *> *)featureNames {
    return [NSSet setWithArray:@[@"in0", @"state_tokenshift_in", @"state_wkv_in"]];
}

- (nullable MLFeatureValue *)featureValueForName:(NSString *)featureName {
    if ([featureName isEqualToString:@"in0"]) {
        return [MLFeatureValue featureValueWithMultiArray:self.in0];
    }
    if ([featureName isEqualToString:@"state_tokenshift_in"]) {
        return [MLFeatureValue featureValueWithMultiArray:self.state_tokenshift_in];
    }
    if ([featureName isEqualToString:@"state_wkv_in"]) {
        return [MLFeatureValue featureValueWithMultiArray:self.state_wkv_in];
    }
    return nil;
}

@end

@implementation rwkv_coreml_implOutput

- (instancetype)initWithLogits:(MLMultiArray *)logits state_tokenshift_out:(MLMultiArray *)state_tokenshift_out state_wkv_out:(MLMultiArray *)state_wkv_out {
    self = [super init];
    if (self) {
        _logits = logits;
        _state_tokenshift_out = state_tokenshift_out;
        _state_wkv_out = state_wkv_out;
    }
    return self;
}

- (NSSet<NSString *> *)featureNames {
    return [NSSet setWithArray:@[@"logits", @"state_tokenshift_out", @"state_wkv_out"]];
}

- (nullable MLFeatureValue *)featureValueForName:(NSString *)featureName {
    if ([featureName isEqualToString:@"logits"]) {
        return [MLFeatureValue featureValueWithMultiArray:self.logits];
    }
    if ([featureName isEqualToString:@"state_tokenshift_out"]) {
        return [MLFeatureValue featureValueWithMultiArray:self.state_tokenshift_out];
    }
    if ([featureName isEqualToString:@"state_wkv_out"]) {
        return [MLFeatureValue featureValueWithMultiArray:self.state_wkv_out];
    }
    return nil;
}

@end

@implementation rwkv_coreml_impl


/**
    URL of the underlying .mlmodelc directory.
*/
+ (nullable NSURL *)URLOfModelInThisBundle {
    NSString *assetPath = [[NSBundle bundleForClass:[self class]] pathForResource:@"rwkv_coreml_impl" ofType:@"mlmodelc"];
    if (nil == assetPath) { os_log_error(OS_LOG_DEFAULT, "Could not load rwkv_coreml_impl.mlmodelc in the bundle resource"); return nil; }
    return [NSURL fileURLWithPath:assetPath];
}


/**
    Initialize rwkv_coreml_impl instance from an existing MLModel object.

    Usually the application does not use this initializer unless it makes a subclass of rwkv_coreml_impl.
    Such application may want to use `-[MLModel initWithContentsOfURL:configuration:error:]` and `+URLOfModelInThisBundle` to create a MLModel object to pass-in.
*/
- (instancetype)initWithMLModel:(MLModel *)model {
    if (model == nil) {
        return nil;
    }
    self = [super init];
    if (self != nil) {
        _model = model;
    }
    return self;
}


/**
    Initialize rwkv_coreml_impl instance with the model in this bundle.
*/
- (nullable instancetype)init {
    return [self initWithContentsOfURL:(NSURL * _Nonnull)self.class.URLOfModelInThisBundle error:nil];
}


/**
    Initialize rwkv_coreml_impl instance with the model in this bundle.

    @param configuration The model configuration object
    @param error If an error occurs, upon return contains an NSError object that describes the problem. If you are not interested in possible errors, pass in NULL.
*/
- (nullable instancetype)initWithConfiguration:(MLModelConfiguration *)configuration error:(NSError * _Nullable __autoreleasing * _Nullable)error {
    return [self initWithContentsOfURL:(NSURL * _Nonnull)self.class.URLOfModelInThisBundle configuration:configuration error:error];
}


/**
    Initialize rwkv_coreml_impl instance from the model URL.

    @param modelURL URL to the .mlmodelc directory for rwkv_coreml_impl.
    @param error If an error occurs, upon return contains an NSError object that describes the problem. If you are not interested in possible errors, pass in NULL.
*/
- (nullable instancetype)initWithContentsOfURL:(NSURL *)modelURL error:(NSError * _Nullable __autoreleasing * _Nullable)error {
    MLModel *model = [MLModel modelWithContentsOfURL:modelURL error:error];
    if (model == nil) { return nil; }
    return [self initWithMLModel:model];
}


/**
    Initialize rwkv_coreml_impl instance from the model URL.

    @param modelURL URL to the .mlmodelc directory for rwkv_coreml_impl.
    @param configuration The model configuration object
    @param error If an error occurs, upon return contains an NSError object that describes the problem. If you are not interested in possible errors, pass in NULL.
*/
- (nullable instancetype)initWithContentsOfURL:(NSURL *)modelURL configuration:(MLModelConfiguration *)configuration error:(NSError * _Nullable __autoreleasing * _Nullable)error {
    MLModel *model = [MLModel modelWithContentsOfURL:modelURL configuration:configuration error:error];
    if (model == nil) { return nil; }
    return [self initWithMLModel:model];
}


/**
    Construct rwkv_coreml_impl instance asynchronously with configuration.
    Model loading may take time when the model content is not immediately available (e.g. encrypted model). Use this factory method especially when the caller is on the main thread.

    @param configuration The model configuration
    @param handler When the model load completes successfully or unsuccessfully, the completion handler is invoked with a valid rwkv_coreml_impl instance or NSError object.
*/
+ (void)loadWithConfiguration:(MLModelConfiguration *)configuration completionHandler:(void (^)(rwkv_coreml_impl * _Nullable model, NSError * _Nullable error))handler {
    [self loadContentsOfURL:(NSURL * _Nonnull)[self URLOfModelInThisBundle]
              configuration:configuration
          completionHandler:handler];
}


/**
    Construct rwkv_coreml_impl instance asynchronously with URL of .mlmodelc directory and optional configuration.

    Model loading may take time when the model content is not immediately available (e.g. encrypted model). Use this factory method especially when the caller is on the main thread.

    @param modelURL The model URL.
    @param configuration The model configuration
    @param handler When the model load completes successfully or unsuccessfully, the completion handler is invoked with a valid rwkv_coreml_impl instance or NSError object.
*/
+ (void)loadContentsOfURL:(NSURL *)modelURL configuration:(MLModelConfiguration *)configuration completionHandler:(void (^)(rwkv_coreml_impl * _Nullable model, NSError * _Nullable error))handler {
    [MLModel loadContentsOfURL:modelURL
                 configuration:configuration
             completionHandler:^(MLModel *model, NSError *error) {
        if (model != nil) {
            rwkv_coreml_impl *typedModel = [[rwkv_coreml_impl alloc] initWithMLModel:model];
            handler(typedModel, nil);
        } else {
            handler(nil, error);
        }
    }];
}

- (nullable rwkv_coreml_implOutput *)predictionFromFeatures:(rwkv_coreml_implInput *)input error:(NSError * _Nullable __autoreleasing * _Nullable)error {
    return [self predictionFromFeatures:input options:[[MLPredictionOptions alloc] init] error:error];
}

- (nullable rwkv_coreml_implOutput *)predictionFromFeatures:(rwkv_coreml_implInput *)input options:(MLPredictionOptions *)options error:(NSError * _Nullable __autoreleasing * _Nullable)error {
    id<MLFeatureProvider> outFeatures = [self.model predictionFromFeatures:input options:options error:error];
    if (!outFeatures) { return nil; }
    return [[rwkv_coreml_implOutput alloc] initWithLogits:(MLMultiArray *)[outFeatures featureValueForName:@"logits"].multiArrayValue state_tokenshift_out:(MLMultiArray *)[outFeatures featureValueForName:@"state_tokenshift_out"].multiArrayValue state_wkv_out:(MLMultiArray *)[outFeatures featureValueForName:@"state_wkv_out"].multiArrayValue];
}

- (void)predictionFromFeatures:(rwkv_coreml_implInput *)input completionHandler:(void (^)(rwkv_coreml_implOutput * _Nullable output, NSError * _Nullable error))completionHandler {
    [self.model predictionFromFeatures:input completionHandler:^(id<MLFeatureProvider> prediction, NSError *predictionError) {
        if (prediction != nil) {
            rwkv_coreml_implOutput *output = [[rwkv_coreml_implOutput alloc] initWithLogits:(MLMultiArray *)[prediction featureValueForName:@"logits"].multiArrayValue state_tokenshift_out:(MLMultiArray *)[prediction featureValueForName:@"state_tokenshift_out"].multiArrayValue state_wkv_out:(MLMultiArray *)[prediction featureValueForName:@"state_wkv_out"].multiArrayValue];
            completionHandler(output, predictionError);
        } else {
            completionHandler(nil, predictionError);
        }
    }];
}

- (void)predictionFromFeatures:(rwkv_coreml_implInput *)input options:(MLPredictionOptions *)options completionHandler:(void (^)(rwkv_coreml_implOutput * _Nullable output, NSError * _Nullable error))completionHandler {
    [self.model predictionFromFeatures:input options:options completionHandler:^(id<MLFeatureProvider> prediction, NSError *predictionError) {
        if (prediction != nil) {
            rwkv_coreml_implOutput *output = [[rwkv_coreml_implOutput alloc] initWithLogits:(MLMultiArray *)[prediction featureValueForName:@"logits"].multiArrayValue state_tokenshift_out:(MLMultiArray *)[prediction featureValueForName:@"state_tokenshift_out"].multiArrayValue state_wkv_out:(MLMultiArray *)[prediction featureValueForName:@"state_wkv_out"].multiArrayValue];
            completionHandler(output, predictionError);
        } else {
            completionHandler(nil, predictionError);
        }
    }];
}

- (nullable rwkv_coreml_implOutput *)predictionFromIn0:(MLMultiArray *)in0 state_tokenshift_in:(MLMultiArray *)state_tokenshift_in state_wkv_in:(MLMultiArray *)state_wkv_in error:(NSError * _Nullable __autoreleasing * _Nullable)error {
    rwkv_coreml_implInput *input_ = [[rwkv_coreml_implInput alloc] initWithIn0:in0 state_tokenshift_in:state_tokenshift_in state_wkv_in:state_wkv_in];
    return [self predictionFromFeatures:input_ error:error];
}

- (nullable NSArray<rwkv_coreml_implOutput *> *)predictionsFromInputs:(NSArray<rwkv_coreml_implInput*> *)inputArray options:(MLPredictionOptions *)options error:(NSError * _Nullable __autoreleasing * _Nullable)error {
    id<MLBatchProvider> inBatch = [[MLArrayBatchProvider alloc] initWithFeatureProviderArray:inputArray];
    id<MLBatchProvider> outBatch = [self.model predictionsFromBatch:inBatch options:options error:error];
    if (!outBatch) { return nil; }
    NSMutableArray<rwkv_coreml_implOutput*> *results = [NSMutableArray arrayWithCapacity:(NSUInteger)outBatch.count];
    for (NSInteger i = 0; i < outBatch.count; i++) {
        id<MLFeatureProvider> resultProvider = [outBatch featuresAtIndex:i];
        rwkv_coreml_implOutput * result = [[rwkv_coreml_implOutput alloc] initWithLogits:(MLMultiArray *)[resultProvider featureValueForName:@"logits"].multiArrayValue state_tokenshift_out:(MLMultiArray *)[resultProvider featureValueForName:@"state_tokenshift_out"].multiArrayValue state_wkv_out:(MLMultiArray *)[resultProvider featureValueForName:@"state_wkv_out"].multiArrayValue];
        [results addObject:result];
    }
    return results;
}

@end
