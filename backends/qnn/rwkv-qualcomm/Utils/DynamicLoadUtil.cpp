//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All rights reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <iostream>

#include "DynamicLoadUtil.hpp"
#include "logger.h"
#include "PAL/DynamicLoading.hpp"

using namespace qnn;
using namespace qnn::tools;

typedef Qnn_ErrorHandle_t (*QnnInterfaceGetProvidersFn_t)(const QnnInterface_t*** providerList,
                                                          uint32_t* numProviders);

typedef Qnn_ErrorHandle_t (*QnnSystemInterfaceGetProvidersFn_t)(
    const QnnSystemInterface_t*** providerList, uint32_t* numProviders);

template <class T>
static inline T resolveSymbol(void* libHandle, const char* sym) {
  T ptr = (T)pal::dynamicloading::dlSym(libHandle, sym);
  if (ptr == nullptr) {
    LOGE("Unable to access symbol [%s]. pal::dynamicloading::dlError(): %s",
              sym,
              pal::dynamicloading::dlError());
  }
  return ptr;
}

dynamicloadutil::StatusCode dynamicloadutil::getQnnFunctionPointers(
    std::string backendPath,
    std::string modelPath,
    rwkv_app::QnnFunctionPointers* qnnFunctionPointers,
    void** backendHandleRtn,
    bool loadModelLib,
    void** modelHandleRtn) {
// #if defined(__ANDROID__)
//   void* libBackendHandle = pal::dynamicloading::dlOpen(
//       backendPath.c_str(), pal::dynamicloading::DL_NOW | pal::dynamicloading::DL_LOCAL);
// #else
  void* libBackendHandle = pal::dynamicloading::dlOpen(
      backendPath.c_str(), pal::dynamicloading::DL_NOW | pal::dynamicloading::DL_GLOBAL);
// #endif
  if (nullptr == libBackendHandle) {
    LOGE("Unable to load backend. pal::dynamicloading::dlError(): %s",
              pal::dynamicloading::dlError());
    return StatusCode::FAIL_LOAD_BACKEND;
  }
  if (nullptr != backendHandleRtn) {
    *backendHandleRtn = libBackendHandle;
  }
  // Get QNN Interface
  QnnInterfaceGetProvidersFn_t getInterfaceProviders{nullptr};
  getInterfaceProviders =
      resolveSymbol<QnnInterfaceGetProvidersFn_t>(libBackendHandle, "QnnInterface_getProviders");
  if (nullptr == getInterfaceProviders) {
    return StatusCode::FAIL_SYM_FUNCTION;
  }
  QnnInterface_t** interfaceProviders{nullptr};
  uint32_t numProviders{0};
  if (QNN_SUCCESS !=
      getInterfaceProviders((const QnnInterface_t***)&interfaceProviders, &numProviders)) {
    LOGE("Failed to get interface providers.");
    return StatusCode::FAIL_GET_INTERFACE_PROVIDERS;
  }
  if (nullptr == interfaceProviders) {
    LOGE("Failed to get interface providers: null interface providers received.");
    return StatusCode::FAIL_GET_INTERFACE_PROVIDERS;
  }
  if (0 == numProviders) {
    LOGE("Failed to get interface providers: 0 interface providers.");
    return StatusCode::FAIL_GET_INTERFACE_PROVIDERS;
  }
  bool foundValidInterface{false};
  for (size_t pIdx = 0; pIdx < numProviders; pIdx++) {
    if (QNN_API_VERSION_MAJOR == interfaceProviders[pIdx]->apiVersion.coreApiVersion.major &&
        QNN_API_VERSION_MINOR <= interfaceProviders[pIdx]->apiVersion.coreApiVersion.minor) {
      foundValidInterface               = true;
      qnnFunctionPointers->qnnInterface = interfaceProviders[pIdx]->QNN_INTERFACE_VER_NAME;
      break;
    }
  }
  if (!foundValidInterface) {
    LOGE("Unable to find a valid interface.");
    libBackendHandle = nullptr;
    return StatusCode::FAIL_GET_INTERFACE_PROVIDERS;
  }

  if (true == loadModelLib) {
    LOGI("Loading model shared library ([model].so)");
    void* libModelHandle = pal::dynamicloading::dlOpen(
        modelPath.c_str(), pal::dynamicloading::DL_NOW | pal::dynamicloading::DL_LOCAL);
    if (nullptr == libModelHandle) {
      LOGE("Unable to load model. pal::dynamicloading::dlError(): %s",
                pal::dynamicloading::dlError());
      return StatusCode::FAIL_LOAD_MODEL;
    }
    if (nullptr != modelHandleRtn) {
      *modelHandleRtn = libModelHandle;
    }

    std::string modelPrepareFunc = "QnnModel_composeGraphs";
    qnnFunctionPointers->composeGraphsFnHandle =
        resolveSymbol<rwkv_app::ComposeGraphsFnHandleType_t>(libModelHandle,
                                                               modelPrepareFunc.c_str());
    if (nullptr == qnnFunctionPointers->composeGraphsFnHandle) {
      return StatusCode::FAIL_SYM_FUNCTION;
    }

    std::string modelFreeFunc = "QnnModel_freeGraphsInfo";
    qnnFunctionPointers->freeGraphInfoFnHandle =
        resolveSymbol<rwkv_app::FreeGraphInfoFnHandleType_t>(libModelHandle,
                                                               modelFreeFunc.c_str());
    if (nullptr == qnnFunctionPointers->freeGraphInfoFnHandle) {
      return StatusCode::FAIL_SYM_FUNCTION;
    }
  } else {
    LOGI("Model wasn't loaded from a shared library.");
  }
  return StatusCode::SUCCESS;
}

dynamicloadutil::StatusCode dynamicloadutil::getQnnSystemFunctionPointers(
    std::string systemLibraryPath, rwkv_app::QnnFunctionPointers* qnnFunctionPointers) {
  if (!qnnFunctionPointers) {
    LOGE("nullptr provided for qnnFunctionPointers");
    return StatusCode::FAILURE;
  }
  void* systemLibraryHandle = pal::dynamicloading::dlOpen(
      systemLibraryPath.c_str(), pal::dynamicloading::DL_NOW | pal::dynamicloading::DL_LOCAL);
  if (nullptr == systemLibraryHandle) {
    LOGE("Unable to load system library. pal::dynamicloading::dlError(): %s",
              pal::dynamicloading::dlError());
    return StatusCode::FAIL_LOAD_SYSTEM_LIB;
  }
  QnnSystemInterfaceGetProvidersFn_t getSystemInterfaceProviders{nullptr};
  getSystemInterfaceProviders = resolveSymbol<QnnSystemInterfaceGetProvidersFn_t>(
      systemLibraryHandle, "QnnSystemInterface_getProviders");
  if (nullptr == getSystemInterfaceProviders) {
    return StatusCode::FAIL_SYM_FUNCTION;
  }
  QnnSystemInterface_t** systemInterfaceProviders{nullptr};
  uint32_t numProviders{0};
  if (QNN_SUCCESS != getSystemInterfaceProviders(
                         (const QnnSystemInterface_t***)&systemInterfaceProviders, &numProviders)) {
    LOGE("Failed to get system interface providers.");
    return StatusCode::FAIL_GET_INTERFACE_PROVIDERS;
  }
  if (nullptr == systemInterfaceProviders) {
    LOGE("Failed to get system interface providers: null interface providers received.");
    return StatusCode::FAIL_GET_INTERFACE_PROVIDERS;
  }
  if (0 == numProviders) {
    LOGE("Failed to get interface providers: 0 interface providers.");
    return StatusCode::FAIL_GET_INTERFACE_PROVIDERS;
  }
  bool foundValidSystemInterface{false};
  for (size_t pIdx = 0; pIdx < numProviders; pIdx++) {
    if (QNN_SYSTEM_API_VERSION_MAJOR == systemInterfaceProviders[pIdx]->systemApiVersion.major &&
        QNN_SYSTEM_API_VERSION_MINOR <= systemInterfaceProviders[pIdx]->systemApiVersion.minor) {
      foundValidSystemInterface = true;
      qnnFunctionPointers->qnnSystemInterface =
          systemInterfaceProviders[pIdx]->QNN_SYSTEM_INTERFACE_VER_NAME;
      break;
    }
  }
  if (!foundValidSystemInterface) {
    LOGE("Unable to find a valid system interface.");
    return StatusCode::FAIL_GET_INTERFACE_PROVIDERS;
  }
  return StatusCode::SUCCESS;
}