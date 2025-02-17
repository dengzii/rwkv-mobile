#ifndef LOGGER_H
#define LOGGER_H

enum {
    RWKV_LOG_LEVEL_DEBUG = 0,
    RWKV_LOG_LEVEL_INFO,
    RWKV_LOG_LEVEL_WARN,
    RWKV_LOG_LEVEL_ERROR,
};

#define DEFAULT_LOG_LEVEL RWKV_LOG_LEVEL_INFO

#if defined(__ANDROID__)
#include <android/log.h>
#define LOG_TAG "RWKV-MOBILE"
#define LOGI(...) \
    { if (DEFAULT_LOG_LEVEL <= RWKV_LOG_LEVEL_INFO) { __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__); } }
#define LOGD(...) \
    { if (DEFAULT_LOG_LEVEL <= RWKV_LOG_LEVEL_DEBUG) { __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__); } }
#define LOGW(...) \
    { if (DEFAULT_LOG_LEVEL <= RWKV_LOG_LEVEL_WARN) { __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__); } }
#define LOGE(...) \
    { if (DEFAULT_LOG_LEVEL <= RWKV_LOG_LEVEL_ERROR) { __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__); } }
#else
#include <cstdio>
#define LOGI(...) \
    { if (DEFAULT_LOG_LEVEL <= RWKV_LOG_LEVEL_INFO) { printf("[INFO]: " __VA_ARGS__); } }
#define LOGD(...) \
    { if (DEFAULT_LOG_LEVEL <= RWKV_LOG_LEVEL_DEBUG) { printf("[DEBUG]: " __VA_ARGS__); } }
#define LOGW(...) \
    { if (DEFAULT_LOG_LEVEL <= RWKV_LOG_LEVEL_WARN) { printf("[WARN]: " __VA_ARGS__); } }
#define LOGE(...) \
    { if (DEFAULT_LOG_LEVEL <= RWKV_LOG_LEVEL_ERROR) { printf("[ERROR]: " __VA_ARGS__); } }
#endif

#endif