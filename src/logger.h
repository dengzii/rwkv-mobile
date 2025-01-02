#ifndef LOGGER_H
#define LOGGER_H

#if defined(__ANDROID__)
#include <android/log.h>
#define LOG_TAG "RWKV-MOBILE"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#else
#include <cstdio>
#define LOGI(...) printf("[INFO]: " __VA_ARGS__)
#define LOGD(...) printf("[DEBUG]: " __VA_ARGS__)
#define LOGW(...) printf("[WARN]: " __VA_ARGS__)
#define LOGE(...) printf("[ERROR]: " __VA_ARGS__)
#endif

#endif