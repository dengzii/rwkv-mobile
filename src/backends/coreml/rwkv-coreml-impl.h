//
// rwkv-coreml-impl.h
//
// This file was automatically generated and should not be edited.
//

#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#include <stdint.h>
#include <os/log.h>

NS_ASSUME_NONNULL_BEGIN

/// Model Prediction Input Type
API_AVAILABLE(macos(15.0), ios(18.0), watchos(11.0), tvos(18.0)) __attribute__((visibility("hidden")))
@interface rwkv_coreml_implInput : NSObject<MLFeatureProvider>

/// in0 as 1 by 1 matrix of 32-bit integers
@property (readwrite, nonatomic, strong) MLMultiArray * in0;

/// state_tokenshift_in as 1 × (2 * num_layers) × embd_dim 3-dimensional array of floats
@property (readwrite, nonatomic, strong) MLMultiArray * state_tokenshift_in;

/// state_wkv_in as num_layers × num_heads × head_size × head_size 4-dimensional array of floats
@property (readwrite, nonatomic, strong) MLMultiArray * state_wkv_in;
- (instancetype)init NS_UNAVAILABLE;
- (instancetype)initWithIn0:(MLMultiArray *)in0 state_tokenshift_in:(MLMultiArray *)state_tokenshift_in state_wkv_in:(MLMultiArray *)state_wkv_in NS_DESIGNATED_INITIALIZER;

@end

/// Model Prediction Output Type
API_AVAILABLE(macos(15.0), ios(18.0), watchos(11.0), tvos(18.0)) __attribute__((visibility("hidden")))
@interface rwkv_coreml_implOutput : NSObject<MLFeatureProvider>

/// logits as 1 × 1 × 65536 3-dimensional array of floats
@property (readwrite, nonatomic, strong) MLMultiArray * logits;

/// state_tokenshift_out as 1 × (2 * num_layers) × embd_dim 3-dimensional array of floats
@property (readwrite, nonatomic, strong) MLMultiArray * state_tokenshift_out;

/// state_wkv_out as num_layers × num_heads × head_size × head_size 4-dimensional array of floats
@property (readwrite, nonatomic, strong) MLMultiArray * state_wkv_out;
- (instancetype)init NS_UNAVAILABLE;
- (instancetype)initWithLogits:(MLMultiArray *)logits state_tokenshift_out:(MLMultiArray *)state_tokenshift_out state_wkv_out:(MLMultiArray *)state_wkv_out NS_DESIGNATED_INITIALIZER;

@end

/// Class for model loading and prediction
API_AVAILABLE(macos(15.0), ios(18.0), watchos(11.0), tvos(18.0)) __attribute__((visibility("hidden")))
@interface rwkv_coreml_impl : NSObject
@property (readonly, nonatomic, nullable) MLModel * model;

/**
    URL of the underlying .mlmodelc directory.
*/
+ (nullable NSURL *)URLOfModelInThisBundle;

/**
    Initialize rwkv_coreml_impl instance from an existing MLModel object.

    Usually the application does not use this initializer unless it makes a subclass of rwkv_coreml_impl.
    Such application may want to use `-[MLModel initWithContentsOfURL:configuration:error:]` and `+URLOfModelInThisBundle` to create a MLModel object to pass-in.
*/
- (instancetype)initWithMLModel:(MLModel *)model NS_DESIGNATED_INITIALIZER;

/**
    Initialize rwkv_coreml_impl instance with the model in this bundle.
*/
- (nullable instancetype)init;

/**
    Initialize rwkv_coreml_impl instance with the model in this bundle.

    @param configuration The model configuration object
    @param error If an error occurs, upon return contains an NSError object that describes the problem. If you are not interested in possible errors, pass in NULL.
*/
- (nullable instancetype)initWithConfiguration:(MLModelConfiguration *)configuration error:(NSError * _Nullable __autoreleasing * _Nullable)error;

/**
    Initialize rwkv_coreml_impl instance from the model URL.

    @param modelURL URL to the .mlmodelc directory for rwkv_coreml_impl.
    @param error If an error occurs, upon return contains an NSError object that describes the problem. If you are not interested in possible errors, pass in NULL.
*/
- (nullable instancetype)initWithContentsOfURL:(NSURL *)modelURL error:(NSError * _Nullable __autoreleasing * _Nullable)error;

/**
    Initialize rwkv_coreml_impl instance from the model URL.

    @param modelURL URL to the .mlmodelc directory for rwkv_coreml_impl.
    @param configuration The model configuration object
    @param error If an error occurs, upon return contains an NSError object that describes the problem. If you are not interested in possible errors, pass in NULL.
*/
- (nullable instancetype)initWithContentsOfURL:(NSURL *)modelURL configuration:(MLModelConfiguration *)configuration error:(NSError * _Nullable __autoreleasing * _Nullable)error;

/**
    Construct rwkv_coreml_impl instance asynchronously with configuration.
    Model loading may take time when the model content is not immediately available (e.g. encrypted model). Use this factory method especially when the caller is on the main thread.

    @param configuration The model configuration
    @param handler When the model load completes successfully or unsuccessfully, the completion handler is invoked with a valid rwkv_coreml_impl instance or NSError object.
*/
+ (void)loadWithConfiguration:(MLModelConfiguration *)configuration completionHandler:(void (^)(rwkv_coreml_impl * _Nullable model, NSError * _Nullable error))handler;

/**
    Construct rwkv_coreml_impl instance asynchronously with URL of .mlmodelc directory and optional configuration.

    Model loading may take time when the model content is not immediately available (e.g. encrypted model). Use this factory method especially when the caller is on the main thread.

    @param modelURL The model URL.
    @param configuration The model configuration
    @param handler When the model load completes successfully or unsuccessfully, the completion handler is invoked with a valid rwkv_coreml_impl instance or NSError object.
*/
+ (void)loadContentsOfURL:(NSURL *)modelURL configuration:(MLModelConfiguration *)configuration completionHandler:(void (^)(rwkv_coreml_impl * _Nullable model, NSError * _Nullable error))handler;

/**
    Make a prediction using the standard interface
    @param input an instance of rwkv_coreml_implInput to predict from
    @param error If an error occurs, upon return contains an NSError object that describes the problem. If you are not interested in possible errors, pass in NULL.
    @return the prediction as rwkv_coreml_implOutput
*/
- (nullable rwkv_coreml_implOutput *)predictionFromFeatures:(rwkv_coreml_implInput *)input error:(NSError * _Nullable __autoreleasing * _Nullable)error;

/**
    Make a prediction using the standard interface
    @param input an instance of rwkv_coreml_implInput to predict from
    @param options prediction options
    @param error If an error occurs, upon return contains an NSError object that describes the problem. If you are not interested in possible errors, pass in NULL.
    @return the prediction as rwkv_coreml_implOutput
*/
- (nullable rwkv_coreml_implOutput *)predictionFromFeatures:(rwkv_coreml_implInput *)input options:(MLPredictionOptions *)options error:(NSError * _Nullable __autoreleasing * _Nullable)error;

/**
    Make an asynchronous prediction using the standard interface
    @param input an instance of rwkv_coreml_implInput to predict from
    @param completionHandler a block that will be called upon completion of the prediction. error will be nil if no error occurred.
*/
- (void)predictionFromFeatures:(rwkv_coreml_implInput *)input completionHandler:(void (^)(rwkv_coreml_implOutput * _Nullable output, NSError * _Nullable error))completionHandler;

/**
    Make an asynchronous prediction using the standard interface
    @param input an instance of rwkv_coreml_implInput to predict from
    @param options prediction options
    @param completionHandler a block that will be called upon completion of the prediction. error will be nil if no error occurred.
*/
- (void)predictionFromFeatures:(rwkv_coreml_implInput *)input options:(MLPredictionOptions *)options completionHandler:(void (^)(rwkv_coreml_implOutput * _Nullable output, NSError * _Nullable error))completionHandler;

/**
    Make a prediction using the convenience interface
    @param in0 1 by 1 matrix of 32-bit integers
    @param state_tokenshift_in 1 × (2 * num_layers) × embd_dim 3-dimensional array of floats
    @param state_wkv_in num_layers × num_heads × head_size × head_size 4-dimensional array of floats
    @param error If an error occurs, upon return contains an NSError object that describes the problem. If you are not interested in possible errors, pass in NULL.
    @return the prediction as rwkv_coreml_implOutput
*/
- (nullable rwkv_coreml_implOutput *)predictionFromIn0:(MLMultiArray *)in0 state_tokenshift_in:(MLMultiArray *)state_tokenshift_in state_wkv_in:(MLMultiArray *)state_wkv_in error:(NSError * _Nullable __autoreleasing * _Nullable)error;

/**
    Batch prediction
    @param inputArray array of rwkv_coreml_implInput instances to obtain predictions from
    @param options prediction options
    @param error If an error occurs, upon return contains an NSError object that describes the problem. If you are not interested in possible errors, pass in NULL.
    @return the predictions as NSArray<rwkv_coreml_implOutput *>
*/
- (nullable NSArray<rwkv_coreml_implOutput *> *)predictionsFromInputs:(NSArray<rwkv_coreml_implInput*> *)inputArray options:(MLPredictionOptions *)options error:(NSError * _Nullable __autoreleasing * _Nullable)error;
@end

NS_ASSUME_NONNULL_END
