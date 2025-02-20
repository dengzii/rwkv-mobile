#include <fstream>
#include <sstream>
#include <filesystem>
#include <chrono>

#include "backend.h"
#include "qnn_backend.h"
#include "commondef.h"
#include "logger.h"
#include "half.hpp"

#ifdef ENABLE_QNN
#include "PAL/DynamicLoading.hpp"
#include "DynamicLoadUtil.hpp"
#include "DataUtil.hpp"
#include "Utils.hpp"
#include "QnnTypeMacros.hpp"
#include <HTP/QnnHtpPerfInfrastructure.h>
#include <HTP/QnnHtpDevice.h>
#include <HTP/QnnHtpGraph.h>
#include <HTP/QnnHtpContext.h>
#include <QnnContext.h>
#endif

#define USE_MMAP 0
#define DEFAULT_QNN_LOGLEVEL QNN_LOG_LEVEL_INFO

namespace rwkvmobile {

#ifdef ENABLE_QNN
using namespace qnn::tools;

static void logCallback(const char* fmt,
    QnnLog_Level_t level,
    uint64_t timestamp,
    va_list argp) {

    if (nullptr == fmt) {
        return;
    }

    return; // buggy
    // std::cout << fmt << std::endl;

    // switch (level) {
    //     case QNN_LOG_LEVEL_ERROR:
    //         LOGE(fmt, argp);
    //         break;
    //     case QNN_LOG_LEVEL_WARN:
    //         LOGW(fmt, argp);
    //         break;
    //     case QNN_LOG_LEVEL_INFO:
    //         LOGI(fmt, argp);
    //         break;
    //     case QNN_LOG_LEVEL_DEBUG:
    //     case QNN_LOG_LEVEL_VERBOSE:
    //         LOGD(fmt, argp);
    //         break;
    //     case QNN_LOG_LEVEL_MAX:
    //         LOGE(fmt, argp);
    //         break;
    // }
}

static void getTensorDims(std::vector<size_t>& dims,
    uint32_t* inDimensions,
    uint32_t rank) {
    if (nullptr == inDimensions) {
        LOGE("input dimensions is nullptr");
        return;
    }
    for (size_t r = 0; r < rank; r++) {
        dims.push_back(inDimensions[r]);
    }
}


int qnn_backend::init(void * extra) {
    if (extra != nullptr) {
        qnnBackendPath = std::string((char *)extra);
        LOGI("Setting QNN Backend Path: %s\n", qnnBackendPath.c_str());
    } else {
        qnnBackendPath = "libQnnHtp.so";
        LOGI("Using default QNN Backend Path: %s\n", qnnBackendPath.c_str());
    }

    return RWKV_SUCCESS;
}

int qnn_backend::load_model(std::string model_path) {
    if (!std::filesystem::exists(model_path)) {
        return RWKV_ERROR_MODEL | RWKV_ERROR_IO;
    }

    if (qnnBackendPath.empty()) {
        return RWKV_ERROR_BACKEND | RWKV_ERROR_INVALID_PARAMETERS;
    }

    bool is_context_cache = 
#ifdef WIN32
        model_path.find(".dll") == std::string::npos;
#else
        model_path.find(".so") == std::string::npos;
#endif

    // load QNN functions
    auto qnnStatusCode = dynamicloadutil::getQnnFunctionPointers(
        qnnBackendPath, model_path, &qnnFunctionPointers, &qnnBackendLibraryHandle, !is_context_cache, &qnnModelHandle);

    if (dynamicloadutil::StatusCode::SUCCESS != qnnStatusCode) {
        if (dynamicloadutil::StatusCode::FAIL_LOAD_BACKEND == qnnStatusCode) {
            LOGE("Error initializing QNN Function Pointers: could not load backend: %s", qnnBackendPath.c_str());
            return RWKV_ERROR_BACKEND | RWKV_ERROR_IO;
        } else if (dynamicloadutil::StatusCode::FAIL_LOAD_MODEL == qnnStatusCode) {
            LOGE("Error initializing QNN Function Pointers: could not load model:%s ", model_path.c_str());
            return RWKV_ERROR_MODEL | RWKV_ERROR_IO;
        } else {
            LOGE("Error initializing QNN Function Pointers");
            return RWKV_ERROR_BACKEND | RWKV_ERROR_INIT;
        }
    }

    if (is_context_cache) {
        std::string qnnSystemLibPath = 
#ifdef WIN32
            qnnBackendPath.substr(0, qnnBackendPath.find("QnnHtp.dll")) + "QnnSystem.dll";
#else
            qnnBackendPath.substr(0, qnnBackendPath.find("libQnnHtp.so")) + "libQnnSystem.so";
#endif

        auto qnnSystemLibStatus = dynamicloadutil::getQnnSystemFunctionPointers(qnnSystemLibPath, &qnnFunctionPointers);
        if (dynamicloadutil::StatusCode::SUCCESS != qnnSystemLibStatus) {
            LOGE("Error initializing QNN System Function Pointers");
            return RWKV_ERROR_BACKEND | RWKV_ERROR_INIT;
        }
    }

    bool usingHtp = qnnBackendPath.find("Htp") != std::string::npos;
    if (usingHtp) {
        LOGI("Using QNN HTP Backend");
    }
    else {
        LOGE("Do not use QNN CPU/GPU backends!");
        return RWKV_ERROR_BACKEND | RWKV_ERROR_INIT;
    }

    // initialize QNN logging
    auto logLevel = DEFAULT_QNN_LOGLEVEL;
    if (QNN_SUCCESS !=
        qnnFunctionPointers.qnnInterface.logCreate(logCallback, logLevel, &qnnLogHandle)) {
      LOGW("Unable to initialize logging in the backend.");
    }

    // initialize QNN backend
    auto qnnBackendStatus = qnnFunctionPointers.qnnInterface.backendCreate(
        qnnLogHandle, nullptr, &qnnBackendHandle);
    if (QNN_BACKEND_NO_ERROR != qnnBackendStatus) {
      LOGE("Could not initialize backend due to error = %lu", qnnBackendStatus);
      return RWKV_ERROR_BACKEND | RWKV_ERROR_INIT;
    }
    LOGI("Initialize Backend Returned Status = %lu", qnnBackendStatus);

    if (nullptr != qnnFunctionPointers.qnnInterface.propertyHasCapability) {
        auto qnnDevicePropertyStatus = qnnFunctionPointers.qnnInterface.propertyHasCapability(QNN_PROPERTY_GROUP_DEVICE);
        if (QNN_PROPERTY_NOT_SUPPORTED == qnnDevicePropertyStatus) {
            LOGW("Device property is not supported");
        }
        if (QNN_PROPERTY_ERROR_UNKNOWN_KEY == qnnDevicePropertyStatus) {
            LOGE("Device property is not known to backend");
            return RWKV_ERROR_BACKEND | RWKV_ERROR_INIT;
        }

        auto qnnCreateDeviceStatus = qnnFunctionPointers.qnnInterface.deviceCreate(qnnLogHandle, nullptr, &qnnDeviceHandle);
        if (QNN_SUCCESS != qnnCreateDeviceStatus && QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE != qnnCreateDeviceStatus) {
            LOGE("Failed to create device");
            return RWKV_ERROR_BACKEND | RWKV_ERROR_INIT;
        }
    }

    if (usingHtp) {
        if (RWKV_SUCCESS != qnn_create_power_config_id()) {
            LOGE("Could not create HTP power config id");
        } else {
            if (RWKV_SUCCESS != qnn_set_power_config()) {
                LOGE("Could not set HTP power config");
            }
        }

        qnn_set_rpc_latency_and_polling();

#ifdef WIN32
        // TODO
#else
        const char* ldLibraryPath = getenv("LD_LIBRARY_PATH");
        if (ldLibraryPath) {
            std::string pathStr(ldLibraryPath);
            std::stringstream ss(pathStr);
            std::string dir;
            while (std::getline(ss, dir, ':')) {
                std::string fullPath = dir + "/libQnnRwkvWkvOpPackage.so";
                std::ifstream file(fullPath);
                if (file.good()) {
                    LOGI("Found libQnnRwkvWkvOpPackage.so in LD_LIBRARY_PATH: %s", fullPath.c_str());
                    if (RWKV_SUCCESS != qnn_register_op_package(fullPath, "RwkvWkvOpPackageInterfaceProvider")) {
                        LOGE("Op package registration failed");
                    }
                    break;
                }
            }
        }
#endif
    }

    if (is_context_cache) {
        if (nullptr == qnnFunctionPointers.qnnSystemInterface.systemContextCreate ||
            nullptr == qnnFunctionPointers.qnnSystemInterface.systemContextGetBinaryInfo ||
            nullptr == qnnFunctionPointers.qnnSystemInterface.systemContextFree) {
            LOGE("QNN System function pointers are not populated.");
            return RWKV_ERROR_BACKEND | RWKV_ERROR_INIT;
        }

        std::vector<std::shared_ptr<uint8_t>> buffer;
        std::vector<uint64_t> bufferSizes;

        int n_chunks = 1;
        auto pos = model_path.find("_chunk");
        if (pos != std::string::npos) {
            n_chunks = std::stoi(model_path.substr(model_path.find("of") + 2));
            LOGI("Number of chunks: %d", n_chunks);
        }

        buffer.resize(n_chunks);
        bufferSizes.resize(n_chunks);
        qnnContextHandles.resize(n_chunks);

        // read model binaries
        datautil::StatusCode binaryReadingStatus{datautil::StatusCode::SUCCESS};
        for (int i = 0; i < n_chunks; i++) {
            if (n_chunks > 1) {
                model_path = model_path.substr(0, pos) + "_chunk" + std::to_string(i+1) + "of" + std::to_string(n_chunks) + ".bin";
                std::cout << "Reading chunk: " << model_path << std::endl;
            }
            std::tie(binaryReadingStatus, bufferSizes[i]) = datautil::getFileSize(model_path);
            if (0 == bufferSizes[i]) {
                LOGE("Received path to an empty file. Nothing to deserialize.");
                return RWKV_ERROR_MODEL | RWKV_ERROR_IO;
            }
            std::cout << "Buffer size: " << bufferSizes[i] << std::endl;

#if USE_MMAP
            int fd = open(model_path.c_str(), O_RDONLY);
            if (fd < 0) {
                LOGE("Failed to open file %s", model_path.c_str());
                return RWKV_ERROR_MODEL | RWKV_ERROR_IO;
            }

            buffer[i] = std::shared_ptr<uint8_t>(
                (uint8_t*)mmap(NULL, bufferSizes[i], PROT_READ, MAP_SHARED, fd, 0), [bufferSizes, i](uint8_t* p) {
                    if (p) {
                        munmap(p, bufferSizes[i]);
                    }
                    }
                );

            if (buffer[i].get() == MAP_FAILED) {
                LOGE("Failed to mmap file %s", model_path.c_str());
                close(fd);
                return RWKV_ERROR_MODEL | RWKV_ERROR_IO;
            }
#else
            buffer[i] = std::shared_ptr<uint8_t>(new uint8_t[bufferSizes[i]], std::default_delete<uint8_t[]>());
            if (!buffer[i]) {
                LOGE("Failed to allocate memory.");
                return RWKV_ERROR_MODEL | RWKV_ERROR_ALLOC;
            }

            binaryReadingStatus = datautil::readBinaryFromFile(
                model_path, reinterpret_cast<uint8_t*>(buffer[i].get()), bufferSizes[i]);
            if (binaryReadingStatus != datautil::StatusCode::SUCCESS) {
                LOGE("Failed to read binary data.");
                return RWKV_ERROR_MODEL | RWKV_ERROR_IO;
            }
#endif
        }

        // inspect binary info
        int returnStatus = RWKV_SUCCESS;
        std::vector<qnn_wrapper_api::GraphInfo_t **> graphInfos(n_chunks);
        std::vector<uint32_t> graphCounts(n_chunks);

        for (int i = 0; i < n_chunks; i++)
        {
            QnnSystemContext_Handle_t sysCtxHandle{nullptr};
            if (QNN_SUCCESS != qnnFunctionPointers.qnnSystemInterface.systemContextCreate(&sysCtxHandle)) {
                LOGE("Could not create system handle.");
                returnStatus = RWKV_ERROR_MODEL | RWKV_ERROR_IO;
            }

            const QnnSystemContext_BinaryInfo_t* binaryInfo{nullptr};
            Qnn_ContextBinarySize_t binaryInfoSize{0};
            if (RWKV_SUCCESS == returnStatus &&
                QNN_SUCCESS != qnnFunctionPointers.qnnSystemInterface.systemContextGetBinaryInfo(
                                    sysCtxHandle,
                                    static_cast<void*>(buffer[i].get()),
                                    bufferSizes[i],
                                    &binaryInfo,
                                    &binaryInfoSize)) {
                LOGE("Failed to get context binary info");
                returnStatus = RWKV_ERROR_MODEL | RWKV_ERROR_IO;
            }

            // fill GraphInfo_t based on binary info
            if (RWKV_SUCCESS == returnStatus &&
                !qnn::tools::rwkv_app::copyMetadataToGraphsInfo(binaryInfo, graphInfos[i], graphCounts[i])) {
                LOGE("Failed to copy metadata.");
                returnStatus = RWKV_ERROR_MODEL;
            }
            qnnFunctionPointers.qnnSystemInterface.systemContextFree(sysCtxHandle);
            sysCtxHandle = nullptr;

            if (RWKV_SUCCESS == returnStatus &&
                nullptr == qnnFunctionPointers.qnnInterface.contextCreateFromBinary) {
                LOGE("contextCreateFromBinaryFnHandle is nullptr.");
                returnStatus = RWKV_ERROR_MODEL;
            }

            // QnnHtpContext_CustomConfig_t customConfig;
            // customConfig.option = QNN_HTP_CONTEXT_CONFIG_OPTION_IO_MEESTIMATION;
            // customConfig.ioMemEstimation = true;
            // QnnContext_Config_t* cfgs[] = {(QnnContext_Config_t*)&customConfig, NULL};

            if (RWKV_SUCCESS == returnStatus &&
                qnnFunctionPointers.qnnInterface.contextCreateFromBinary(
                    qnnBackendHandle,
                    qnnDeviceHandle,
                    nullptr, // (const QnnContext_Config_t**)cfgs,
                    static_cast<void*>(buffer[i].get()),
                    bufferSizes[i],
                    &qnnContextHandles[i],
                    nullptr)) {
                LOGE("Could not create context from binary.");
                returnStatus = RWKV_ERROR_MODEL;
            }

            isContextCreated = true;
            if (RWKV_SUCCESS == returnStatus) {
                for (size_t graphIdx = 0; graphIdx < graphCounts[i]; graphIdx++) {
                    if (nullptr == qnnFunctionPointers.qnnInterface.graphRetrieve) {
                        LOGE("graphRetrieveFnHandle is nullptr.");
                        returnStatus = RWKV_ERROR_MODEL;
                        break;
                    }
                    if (QNN_SUCCESS !=
                        qnnFunctionPointers.qnnInterface.graphRetrieve(
                            qnnContextHandles[i], (*graphInfos[i])[graphIdx].graphName, &((*graphInfos[i])[graphIdx].graph))) {
                        LOGE("Unable to retrieve graph handle for graph Idx: %zu", graphIdx);
                        returnStatus = RWKV_ERROR_MODEL;
                    }
                }
            }
            if (RWKV_SUCCESS != returnStatus) {
                LOGD("Cleaning up graph Info structures.");
                qnn_wrapper_api::freeGraphsInfo(&graphInfos[i], graphCounts[i]);
            }
        }

        qnnGraphsCount = 0;
        for (int i = 0; i < n_chunks; i++) {
            qnnGraphsCount += graphCounts[i];
        }

        qnnGraphsInfo = (qnn_wrapper_api::GraphInfo_t **)calloc(qnnGraphsCount, sizeof(qnn_wrapper_api::GraphInfo_t *));
        qnn_wrapper_api::GraphInfo_t *graphInfoArr =
            (qnn_wrapper_api::GraphInfo_t *)calloc(qnnGraphsCount, sizeof(qnn_wrapper_api::GraphInfo_t));
        if (nullptr == qnnGraphsInfo || nullptr == graphInfoArr) {
            LOGE("Failure to allocate memory for *graphInfo");
            returnStatus = RWKV_ERROR_MODEL;
        }

        if (RWKV_SUCCESS == returnStatus) {
            int gidx = 0;
            for (int i = 0; i < n_chunks; i++) {
              for (int j = 0; j < graphCounts[i]; j++) {
                qnnGraphsInfo[gidx] = graphInfoArr + gidx;
                qnnGraphsInfo[gidx]->graph = (*graphInfos[i])[j].graph;
                qnnGraphsInfo[gidx]->graphName = strdup((*graphInfos[i])[j].graphName);
                qnnGraphsInfo[gidx]->inputTensors = (*graphInfos[i])[j].inputTensors;
                qnnGraphsInfo[gidx]->numInputTensors = (*graphInfos[i])[j].numInputTensors;
                qnnGraphsInfo[gidx]->outputTensors = (*graphInfos[i])[j].outputTensors;
                qnnGraphsInfo[gidx]->numOutputTensors = (*graphInfos[i])[j].numOutputTensors;
                gidx++;
              }
            }
          }
    } else {
        // create context
        qnnContextHandles.resize(1);
        if (QNN_CONTEXT_NO_ERROR != qnnFunctionPointers.qnnInterface.contextCreate(
                    qnnBackendHandle,
                    qnnDeviceHandle,
                    nullptr, // const QnnContext_Config_t**
                    &qnnContextHandles[0])) {
            LOGE("Could not create context");
            return RWKV_ERROR_BACKEND;
        }
        isContextCreated = true;

        // conpose graphs
        if (graphConfigsInfo == nullptr) {
            graphConfigsInfoCount = 2;

            graphConfigsInfo = new qnn_wrapper_api::GraphConfigInfo_t*[graphConfigsInfoCount];
            graphConfigsInfo[0] = new qnn_wrapper_api::GraphConfigInfo_t();
            graphConfigsInfo[0]->graphName = (char*)"model";
            graphConfigsInfo[0]->graphConfigs = (const QnnGraph_Config_t**)new QnnGraph_Config_t*[2];
            graphConfigsInfo[1] = new qnn_wrapper_api::GraphConfigInfo_t();
            graphConfigsInfo[1]->graphName = (char*)"model_fp16";
            graphConfigsInfo[1]->graphConfigs = (const QnnGraph_Config_t**)new QnnGraph_Config_t*[2];
        
            static QnnHtpGraph_CustomConfig_t customConfig;
            customConfig.option = QNN_HTP_GRAPH_CONFIG_OPTION_PRECISION;
            customConfig.precision = QNN_PRECISION_FLOAT16;
            static QnnGraph_Config_t graphConfig;
            graphConfig.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
            graphConfig.customConfig = &customConfig;
            for (int i = 0; i < graphConfigsInfoCount; i++) {
                graphConfigsInfo[i]->graphConfigs[0] = &graphConfig;
                graphConfigsInfo[i]->graphConfigs[1] = nullptr;
            }
        }

        if (qnn_wrapper_api::ModelError_t::MODEL_NO_ERROR !=
            qnnFunctionPointers.composeGraphsFnHandle(
                qnnBackendHandle,
                qnnFunctionPointers.qnnInterface,
                qnnContextHandles[0],
                (const qnn_wrapper_api::GraphConfigInfo_t**)graphConfigsInfo,
                graphConfigsInfoCount,
                &qnnGraphsInfo,
                &qnnGraphsCount,
                false,
                logCallback,
                DEFAULT_QNN_LOGLEVEL)) {
          LOGE("Failed in composeGraphs()");
          return RWKV_ERROR_MODEL;
        }

        // finalize graphs
        for (size_t graphIdx = 0; graphIdx < qnnGraphsCount; graphIdx++) {
            if (QNN_GRAPH_NO_ERROR !=
                qnnFunctionPointers.qnnInterface.graphFinalize(
                    (*qnnGraphsInfo)[graphIdx].graph, nullptr, nullptr)) {
                return RWKV_ERROR_MODEL;
            }
        }

        // save context cache
// #if WIN32
//         qnn_save_context_cache(model_path.substr(0, model_path.find('.dll')) + "_cache.bin");
// #else
//         qnn_save_context_cache(model_path.substr(0, model_path.find('.so')) + "_cache.bin");
// #endif

    }

    if (RWKV_SUCCESS != qnn_initialize_tensors()) {
        LOGE("Could not initialize tensors");
        return RWKV_ERROR_MODEL;
    }

    // version = info.version;
    n_layers = ((*qnnGraphsInfo)[0].numInputTensors - 1) / 3;
    for (size_t i = 0; i < (*qnnGraphsInfo)[0].numInputTensors; i++) {
        std::string inputName = std::string(QNN_TENSOR_GET_NAME(inputTensors[0][i]));
        if (inputName.find("state1_in") != std::string::npos) {
            std::vector<size_t> dims;
            getTensorDims(dims, QNN_TENSOR_GET_DIMENSIONS(inputTensors[0][i]), QNN_TENSOR_GET_RANK(inputTensors[0][i]));
            num_heads = dims[0];
            hidden_size = num_heads * dims[1];
        }
    }

    std::vector<size_t> dims;
    getTensorDims(dims, QNN_TENSOR_GET_DIMENSIONS(outputTensors[qnnGraphsCount - 1][outputIdx[qnnGraphsCount - 1]]),
        QNN_TENSOR_GET_RANK(outputTensors[qnnGraphsCount - 1][outputIdx[qnnGraphsCount - 1]]));
    vocab_size = dims[2];
    return RWKV_SUCCESS;
}

int qnn_backend::qnn_initialize_tensors() {
    int ret = RWKV_SUCCESS;
    if (nullptr == inputTensors[0] || nullptr == outputTensors[0]) {
        for (int graph_id = 0; graph_id < qnnGraphsCount; graph_id++) {
            auto graphInfo     = (*qnnGraphsInfo)[graph_id];
            LOGD("Graph %d : %s", graph_id, graphInfo.graphName);

            if (iotensor::StatusCode::SUCCESS !=
                qnnIOTensorUtils.setupInputAndOutputTensors(&inputTensors[graph_id], &outputTensors[graph_id], graphInfo)) {
                LOGE("Error in setting up Input and output Tensors");
                return RWKV_ERROR_IO;
            }

            inputIdx.push_back(-1);
            outputIdx.push_back(-1);
            vfirstInIdx.push_back(-1);
            vfirstOutIdx.push_back(-1);

            for (size_t i = 0; i < graphInfo.numInputTensors; i++) {
                LOGD("Input Tensor %zu : %s Type: %d", i, QNN_TENSOR_GET_NAME(inputTensors[graph_id][i]), QNN_TENSOR_GET_DATA_TYPE(inputTensors[graph_id][i]));
        
                std::string inputName = std::string(QNN_TENSOR_GET_NAME(inputTensors[graph_id][i]));
                if (inputName == "in") {
                  inputIdx[graph_id] = i;
                } else if (inputName == "v_first_in") {
                  vfirstInIdx[graph_id] = i;
                }
        
                std::vector<size_t> dims;
                getTensorDims(dims, QNN_TENSOR_GET_DIMENSIONS(inputTensors[graph_id][i]), QNN_TENSOR_GET_RANK(inputTensors[graph_id][i]));

                if (QNN_TENSOR_GET_DATA_TYPE(inputTensors[graph_id][i]) == QNN_DATATYPE_FLOAT_16)
                    memset(QNN_TENSOR_GET_CLIENT_BUF(inputTensors[graph_id][i]).data, 0, datautil::calculateElementCount(dims) * sizeof(uint16_t));
                else if (QNN_TENSOR_GET_DATA_TYPE(inputTensors[graph_id][i]) == QNN_DATATYPE_FLOAT_32)
                    memset(QNN_TENSOR_GET_CLIENT_BUF(inputTensors[graph_id][i]).data, 0, datautil::calculateElementCount(dims) * sizeof(float));
                else {
                    float *ptr = new float[datautil::calculateElementCount(dims)];
                    memset(ptr, 0, datautil::calculateElementCount(dims) * sizeof(float));
                    qnnIOTensorUtils.copyFromFloatToNative(ptr, &inputTensors[graph_id][i]);
                    delete[] ptr;
                }
            }

            for (size_t i = 0; i < graphInfo.numOutputTensors; i++) {
                LOGD("Output Tensor %zu : %s Type: %d", i, QNN_TENSOR_GET_NAME(outputTensors[graph_id][i]), QNN_TENSOR_GET_DATA_TYPE(outputTensors[graph_id][i]));
        
                std::string outputName = std::string(QNN_TENSOR_GET_NAME(outputTensors[graph_id][i]));
                if (outputName == "out") {
                    outputIdx[graph_id] = i;
                } else if (outputName == "v_first_out") {
                    vfirstOutIdx[graph_id] = i;
                }
        
                std::vector<size_t> dims;
                getTensorDims(dims, QNN_TENSOR_GET_DIMENSIONS(outputTensors[graph_id][i]), QNN_TENSOR_GET_RANK(outputTensors[graph_id][i]));

                if (QNN_TENSOR_GET_DATA_TYPE(outputTensors[graph_id][i]) == QNN_DATATYPE_FLOAT_16)
                    memset(QNN_TENSOR_GET_CLIENT_BUF(outputTensors[graph_id][i]).data, 0, datautil::calculateElementCount(dims) * sizeof(uint16_t));
                else if (QNN_TENSOR_GET_DATA_TYPE(outputTensors[graph_id][i]) == QNN_DATATYPE_FLOAT_32)
                    memset(QNN_TENSOR_GET_CLIENT_BUF(outputTensors[graph_id][i]).data, 0, datautil::calculateElementCount(dims) * sizeof(float));
                else {
                    float *ptr = new float[datautil::calculateElementCount(dims)];
                    memset(ptr, 0, datautil::calculateElementCount(dims) * sizeof(float));
                    qnnIOTensorUtils.copyFromFloatToNative(ptr, &outputTensors[graph_id][i]);
                    delete[] ptr;
                }
            }

            // state copy map
            std::vector<int> tmp(graphInfo.numInputTensors);
            for (size_t i = 0; i < graphInfo.numInputTensors; i++) {
                std::string inputName = std::string(QNN_TENSOR_GET_NAME(inputTensors[graph_id][i]));
                if (inputName.find("state") != std::string::npos) {
                    for (size_t j = 0; j < graphInfo.numInputTensors; j++) {
                        std::string outputName = std::string(QNN_TENSOR_GET_NAME(outputTensors[graph_id][j]));
                        if (outputName.find("state") != std::string::npos) {
                            if (inputName.substr(0, inputName.find("_in")) == outputName.substr(0, outputName.find("_out"))) {
                                tmp[i] = j;
                                break;
                            }
                        }
                    }
                } else {
                    tmp[i] = -1;
                }
            }
            stateCopyMap.push_back(tmp);
        }
    }

    return ret;
}

int qnn_backend::eval(int id, std::vector<float> &logits) {
    if (nullptr == inputTensors[0] || nullptr == outputTensors[0])
        return RWKV_ERROR_EVAL;

    int *token_input = (int*)QNN_TENSOR_GET_CLIENT_BUF(inputTensors[0][0]).data;
    *token_input = id;

    // copy states
    // TODO: zero copy
    for (size_t graph_id = 0; graph_id < qnnGraphsCount; graph_id++) {
        for (size_t idx = 0; idx < stateCopyMap[graph_id].size(); idx++) {
            if (stateCopyMap[graph_id][idx] != -1) {
                auto tmp = getQnnTensorClientBuf(inputTensors[graph_id][idx]);
                setQnnTensorClientBuf(inputTensors[graph_id][idx], getQnnTensorClientBuf(outputTensors[graph_id][stateCopyMap[graph_id][idx]]));
                setQnnTensorClientBuf(outputTensors[graph_id][stateCopyMap[graph_id][idx]], tmp);
            }
        }
    }

    for (int graph_id = 0; graph_id < qnnGraphsCount; graph_id++) {
        auto graphInfo     = (*qnnGraphsInfo)[graph_id];
        if (graph_id) { // chunked models
          auto tmp = getQnnTensorClientBuf(&inputTensors[graph_id][inputIdx[graph_id]]);
          setQnnTensorClientBuf(&inputTensors[graph_id][inputIdx[graph_id]], getQnnTensorClientBuf(&outputTensors[graph_id - 1][outputIdx[graph_id - 1]]));
          setQnnTensorClientBuf(&outputTensors[graph_id - 1][outputIdx[graph_id - 1]], tmp);
    
            if (vfirstInIdx[graph_id] != -1) {
                auto tmp = getQnnTensorClientBuf(&inputTensors[graph_id][vfirstInIdx[graph_id]]);
                setQnnTensorClientBuf(&inputTensors[graph_id][vfirstInIdx[graph_id]], getQnnTensorClientBuf(&outputTensors[graph_id - 1][vfirstOutIdx[graph_id - 1]]));
                setQnnTensorClientBuf(&outputTensors[graph_id - 1][vfirstOutIdx[graph_id - 1]], tmp);
            }
        }

        std::chrono::high_resolution_clock::time_point infer_start = std::chrono::high_resolution_clock::now();
        auto executeStatus = qnnFunctionPointers.qnnInterface.graphExecute(graphInfo.graph,
                                                            inputTensors[graph_id],
                                                            graphInfo.numInputTensors,
                                                            outputTensors[graph_id],
                                                            graphInfo.numOutputTensors,
                                                            nullptr,
                                                            nullptr);
        std::chrono::high_resolution_clock::time_point infer_end = std::chrono::high_resolution_clock::now();
    }

    // copy logits
    if (logits.empty()) logits = std::vector<float>(vocab_size);

    int graph_id = qnnGraphsCount - 1;
    int tensor_id = outputIdx[graph_id];
    if (QNN_TENSOR_GET_DATA_TYPE(outputTensors[graph_id][tensor_id]) == QNN_DATATYPE_FLOAT_32) {
        float *ptr = (float*)QNN_TENSOR_GET_CLIENT_BUF(outputTensors[graph_id][tensor_id]).data;
        logits = std::vector<float>(ptr, ptr + vocab_size);
    } else if (QNN_TENSOR_GET_DATA_TYPE(outputTensors[graph_id][tensor_id]) == QNN_DATATYPE_FLOAT_16) {
        half_float::half *ptr = (half_float::half*)QNN_TENSOR_GET_CLIENT_BUF(outputTensors[graph_id][tensor_id]).data;
        for (int i = 0; i < vocab_size; i++) {
            logits[i] = ptr[i];
        }
    } else {
        float *buffer;
        qnnIOTensorUtils.convertToFloat(&buffer, &outputTensors[graph_id][tensor_id]);
        memcpy(logits.data(), buffer, vocab_size * sizeof(float));
        free(buffer);
    }


    return RWKV_SUCCESS;
}

int qnn_backend::eval(std::vector<int> ids, std::vector<float> &logits) {
    for (auto id : ids) {
        if (RWKV_SUCCESS != eval(id, logits)) {
            return RWKV_ERROR_MODEL;
        }
    }
    return RWKV_SUCCESS;
}

bool qnn_backend::is_available() {
    // TODO: Detect this
    return true;
}

int qnn_backend::clear_state() {
    for (int graph_id = 0; graph_id < qnnGraphsCount; graph_id++) {
        auto graphInfo     = (*qnnGraphsInfo)[graph_id];

        for (size_t i = 0; i < graphInfo.numOutputTensors; i++) {
            std::string outputName = std::string(QNN_TENSOR_GET_NAME(outputTensors[graph_id][i]));
            
            if (outputName.find("state") != std::string::npos) {
                std::vector<size_t> dims;
                getTensorDims(dims, QNN_TENSOR_GET_DIMENSIONS(outputTensors[graph_id][i]), QNN_TENSOR_GET_RANK(outputTensors[graph_id][i]));

                if (QNN_TENSOR_GET_DATA_TYPE(outputTensors[graph_id][i]) == QNN_DATATYPE_FLOAT_16)
                    memset(QNN_TENSOR_GET_CLIENT_BUF(outputTensors[graph_id][i]).data, 0, datautil::calculateElementCount(dims) * sizeof(uint16_t));
                else if (QNN_TENSOR_GET_DATA_TYPE(outputTensors[graph_id][i]) == QNN_DATATYPE_FLOAT_32)
                    memset(QNN_TENSOR_GET_CLIENT_BUF(outputTensors[graph_id][i]).data, 0, datautil::calculateElementCount(dims) * sizeof(float));
            }
        }
    }
    return RWKV_SUCCESS;
}

int qnn_backend::get_state(std::any &state) {
    auto new_state = std::vector<std::vector<uint8_t>>();
    for (int graph_id = 0; graph_id < qnnGraphsCount; graph_id++) {
        auto graphInfo     = (*qnnGraphsInfo)[graph_id];

        for (size_t i = 0; i < graphInfo.numOutputTensors; i++) {
            std::string outputName = std::string(QNN_TENSOR_GET_NAME(outputTensors[graph_id][i]));
            
            if (outputName.find("state") != std::string::npos) {
                new_state.push_back(std::vector<uint8_t>((uint8_t*)QNN_TENSOR_GET_CLIENT_BUF(outputTensors[graph_id][i]).data, (uint8_t*)QNN_TENSOR_GET_CLIENT_BUF(outputTensors[graph_id][i]).data + QNN_TENSOR_GET_CLIENT_BUF(outputTensors[graph_id][i]).dataSize));
            }
        }
    }
    state = new_state;
    return RWKV_SUCCESS;
}

int qnn_backend::set_state(std::any state) {
    if (!state.has_value()) return RWKV_SUCCESS;
    auto new_state = std::any_cast<std::vector<std::vector<uint8_t>>>(state);
    int idx = 0;
    for (int graph_id = 0; graph_id < qnnGraphsCount; graph_id++) {
        auto graphInfo     = (*qnnGraphsInfo)[graph_id];

        for (size_t i = 0; i < graphInfo.numOutputTensors; i++) {
            std::string outputName = std::string(QNN_TENSOR_GET_NAME(outputTensors[graph_id][i]));
            
            if (outputName.find("state") != std::string::npos) {
                memcpy(QNN_TENSOR_GET_CLIENT_BUF(outputTensors[graph_id][i]).data, new_state[idx].data(), new_state[idx].size());
                idx++;
            }
        }
    }
    return RWKV_SUCCESS;
}

int qnn_backend::free_state(std::any state) {
    if (!state.has_value()) return RWKV_SUCCESS;
    auto new_state = std::any_cast<std::vector<std::vector<uint8_t>>>(state);
    for (auto &s : new_state) {
        s.clear();
    }
    new_state.clear();
    return RWKV_SUCCESS;
}

int qnn_backend::release_model() {
    // free graphs
    for (int i = 0; i < qnnGraphsCount; i++) {
        auto graphInfo     = (*qnnGraphsInfo)[i];
        qnnIOTensorUtils.tearDownInputAndOutputTensors(
            inputTensors[i], outputTensors[i], graphInfo.numInputTensors, graphInfo.numOutputTensors);
        inputTensors[i]  = nullptr;
        outputTensors[i] = nullptr;
    }

    qnn_wrapper_api::freeGraphsInfo(&qnnGraphsInfo, qnnGraphsCount);
    qnnGraphsInfo = nullptr;

    if (QNN_CONTEXT_NO_ERROR !=
        qnnFunctionPointers.qnnInterface.contextFree(qnnContextHandles[0], nullptr)) {
        LOGE("Could not free context");
    }

    qnn_destory_power_config_id();

    if (nullptr != qnnFunctionPointers.qnnInterface.propertyHasCapability) {
        auto qnnDevicePropertyStatus = qnnFunctionPointers.qnnInterface.propertyHasCapability(QNN_PROPERTY_GROUP_DEVICE);
        if (QNN_PROPERTY_NOT_SUPPORTED == qnnDevicePropertyStatus) {
            LOGW("Device property is not supported");
        }

        auto qnnStatus = qnnFunctionPointers.qnnInterface.deviceFree(qnnDeviceHandle);
        if (QNN_SUCCESS != qnnStatus && QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE != qnnStatus) {
            LOGE("Failed to free device");
        }
    }

    if (qnnBackendLibraryHandle)
        pal::dynamicloading::dlClose(qnnBackendLibraryHandle);

    if (qnnModelHandle)
        pal::dynamicloading::dlClose(qnnModelHandle);

    for (int i = 0; i < graphConfigsInfoCount; i++) {
        delete graphConfigsInfo[i];
    }
    delete graphConfigsInfo;

    if ((nullptr != qnnBackendHandle && nullptr != qnnFunctionPointers.qnnInterface.backendFree) &&
        QNN_BACKEND_NO_ERROR != qnnFunctionPointers.qnnInterface.backendFree(qnnBackendHandle)) {
        LOGE("Could not terminate QNN backend");
    }
    qnnBackendHandle = nullptr;

    if (nullptr != qnnFunctionPointers.qnnInterface.logFree && nullptr != qnnLogHandle) {
        if (QNN_SUCCESS != qnnFunctionPointers.qnnInterface.logFree(qnnLogHandle)) {
            LOGW("Unable to terminate logging in the backend.");
        }
    }

    return RWKV_SUCCESS;
}

int qnn_backend::release() {
    return RWKV_SUCCESS;
}

int qnn_backend::qnn_register_op_package(std::string package_path, std::string interface_provider) {
    if (nullptr == qnnFunctionPointers.qnnInterface.backendRegisterOpPackage) {
        LOGE("backendRegisterOpPackageFnHandle is nullptr.");
        return RWKV_ERROR_UNSUPPORTED;
    }
    if (QNN_BACKEND_NO_ERROR != qnnFunctionPointers.qnnInterface.backendRegisterOpPackage(
                qnnBackendHandle,
                package_path.c_str(),
                interface_provider.c_str(),
                nullptr)) {
        LOGE("Could not register Op Package: %s and interface provider: %s",
            package_path.c_str(),
            interface_provider.c_str());
        return RWKV_ERROR_BACKEND | RWKV_ERROR_INIT;
    }
    LOGI("Registered Op Package: %s and interface provider: %s",
        package_path.c_str(),
        interface_provider.c_str()
    );
    return RWKV_SUCCESS;
}

int qnn_backend::qnn_create_power_config_id() {
    QnnDevice_Infrastructure_t deviceInfra = nullptr;
    Qnn_ErrorHandle_t devErr = qnnFunctionPointers.qnnInterface.deviceGetInfrastructure(&deviceInfra);
    if (devErr != QNN_SUCCESS) {
        LOGE("deviceGetInfrastructure error");
        return RWKV_ERROR_BACKEND | RWKV_ERROR_INIT;
    }
    QnnHtpDevice_Infrastructure_t *htpInfra = static_cast<QnnHtpDevice_Infrastructure_t *>(deviceInfra);
    QnnHtpDevice_PerfInfrastructure_t perfInfra = htpInfra->perfInfra;
    Qnn_ErrorHandle_t perfInfraErr = perfInfra.createPowerConfigId(deviceId, coreId, &powerConfigId);
    if (perfInfraErr != QNN_SUCCESS) {
      LOGE("createPowerConfigId failed");
      return RWKV_ERROR_BACKEND | RWKV_ERROR_INIT;
    }
    return RWKV_SUCCESS;
}

int qnn_backend::qnn_destory_power_config_id() {
    QnnDevice_Infrastructure_t deviceInfra = nullptr;
    Qnn_ErrorHandle_t devErr = qnnFunctionPointers.qnnInterface.deviceGetInfrastructure(&deviceInfra);
    if (devErr != QNN_SUCCESS) {
        LOGE("deviceGetInfrastructure error");
        return RWKV_ERROR_BACKEND | RWKV_ERROR_RELEASE;
    }
    QnnHtpDevice_Infrastructure_t *htpInfra = static_cast<QnnHtpDevice_Infrastructure_t *>(deviceInfra);
    QnnHtpDevice_PerfInfrastructure_t perfInfra = htpInfra->perfInfra;

    Qnn_ErrorHandle_t perfInfraErr = perfInfra.destroyPowerConfigId(powerConfigId);
    if (perfInfraErr != QNN_SUCCESS) {
        LOGE("destroyPowerConfigId failed");
        return RWKV_ERROR_BACKEND | RWKV_ERROR_RELEASE;
    }
    return RWKV_SUCCESS;
}

int qnn_backend::qnn_set_power_config() {
    QnnDevice_Infrastructure_t deviceInfra = nullptr;
    Qnn_ErrorHandle_t devErr = qnnFunctionPointers.qnnInterface.deviceGetInfrastructure(&deviceInfra);
    if (devErr != QNN_SUCCESS) {
        LOGE("device error");
        return RWKV_ERROR_BACKEND | RWKV_ERROR_INIT;
    }
    QnnHtpDevice_Infrastructure_t *htpInfra = static_cast<QnnHtpDevice_Infrastructure_t *>(deviceInfra);
    QnnHtpDevice_PerfInfrastructure_t perfInfra = htpInfra->perfInfra;

    QnnHtpPerfInfrastructure_PowerConfig_t powerConfig;
    memset(&powerConfig, 0, sizeof(powerConfig));
    powerConfig.option                     = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
    powerConfig.dcvsV3Config.dcvsEnable    = 0; //True to enable Dcvs, False to disbale
    powerConfig.dcvsV3Config.setDcvsEnable = 1;
    powerConfig.dcvsV3Config.contextId     = powerConfigId;  //use the power config id created

    // refer QnnHtpPerfInfrastructure.h
    powerConfig.dcvsV3Config.powerMode       = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
    powerConfig.dcvsV3Config.setSleepLatency = 1; //True to consider Latency parameter otherwise False
    powerConfig.dcvsV3Config.setBusParams    = 1; //True to consider Bus parameter otherwise False
    powerConfig.dcvsV3Config.setCoreParams   = 1; //True to consider Core parameter otherwise False
    powerConfig.dcvsV3Config.sleepDisable    = 1; //True to disable sleep, False to re-enable sleep
    powerConfig.dcvsV3Config.setSleepDisable = 1; //True to consider sleep disable/enable parameter otherwise False

    //Set Sleep latency parameter
    powerConfig.dcvsV3Config.sleepLatency    =  40; // set dsp sleep latency ranges 10-65535 micro sec, refer hexagon sdk

    //set Bus Clock Parameters (refer QnnHtpPerfInfrastructure.h)
    powerConfig.dcvsV3Config.busVoltageCornerMin     = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    powerConfig.dcvsV3Config.busVoltageCornerTarget  = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    powerConfig.dcvsV3Config.busVoltageCornerMax     = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;

    //set Core Clock Parameters (refer QnnHtpPerfInfrastructure.h)
    powerConfig.dcvsV3Config.coreVoltageCornerMin    = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    powerConfig.dcvsV3Config.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    powerConfig.dcvsV3Config.coreVoltageCornerMax    = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;

    // Set power config with different performance parameters
    const QnnHtpPerfInfrastructure_PowerConfig_t *powerConfigs[] = {&powerConfig, NULL};

    Qnn_ErrorHandle_t perfInfraErr = perfInfra.setPowerConfig(powerConfigId, powerConfigs);
    if (perfInfraErr != QNN_SUCCESS) {
        LOGE("setPowerConfig failed");
        return RWKV_ERROR_BACKEND | RWKV_ERROR_INIT;
    }
    return RWKV_SUCCESS;
}

int qnn_backend::qnn_set_rpc_latency_and_polling() {
    QnnDevice_Infrastructure_t deviceInfra = nullptr;
    Qnn_ErrorHandle_t devErr = qnnFunctionPointers.qnnInterface.deviceGetInfrastructure(&deviceInfra);
    if (devErr != QNN_SUCCESS) {
      LOGE("deviceGetInfrastructure error");
      return RWKV_ERROR_BACKEND | RWKV_ERROR_INIT;
    }
    QnnHtpDevice_Infrastructure_t *htpInfra = static_cast<QnnHtpDevice_Infrastructure_t *>(deviceInfra);
    QnnHtpDevice_PerfInfrastructure_t perfInfra = htpInfra->perfInfra;

    // set RPC Control Latency
    QnnHtpPerfInfrastructure_PowerConfig_t rpcControlLatency;            // refer QnnHtpPerfInfrastructure.h
    memset(&rpcControlLatency, 0, sizeof(rpcControlLatency));
    rpcControlLatency.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_CONTROL_LATENCY;
    rpcControlLatency.rpcControlLatencyConfig = 100;         // use rpc control latency recommended 100 us, refer hexagon sdk
    const QnnHtpPerfInfrastructure_PowerConfig_t *powerConfigs1[] = {&rpcControlLatency, NULL};

    Qnn_ErrorHandle_t perfInfraErr = perfInfra.setPowerConfig(powerConfigId, powerConfigs1);  // set RPC latency config on power config id created
    if (perfInfraErr != QNN_SUCCESS) {
        LOGE("setPowerConfig failed");
        return RWKV_ERROR_BACKEND | RWKV_ERROR_INIT;
    }

    // set RPC Polling
    QnnHtpPerfInfrastructure_PowerConfig_t rpcPollingTime;   // refer QnnHtpPerfInfrastructure.h
    memset(&rpcPollingTime, 0, sizeof(rpcPollingTime));
    rpcPollingTime.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_POLLING_TIME;
    rpcPollingTime.rpcPollingTimeConfig = 5000;     // use rpc polling time recommended 0-10000 us
    const QnnHtpPerfInfrastructure_PowerConfig_t* powerConfigs2[] = {&rpcPollingTime, NULL};

    perfInfraErr = perfInfra.setPowerConfig(powerConfigId, powerConfigs2); // set RPC polling config on power config id created
    if (perfInfraErr != QNN_SUCCESS) {
        LOGE("setPowerConfig failed");
        return RWKV_ERROR_BACKEND | RWKV_ERROR_INIT;
    }
    return RWKV_SUCCESS;
}

#else

int qnn_backend::init(void * extra) {
    return RWKV_ERROR_BACKEND | RWKV_ERROR_UNSUPPORTED;
}

int qnn_backend::load_model(std::string model_path) {
    return RWKV_ERROR_BACKEND | RWKV_ERROR_UNSUPPORTED;
}

int qnn_backend::eval(int id, std::vector<float> &logits) {
    return RWKV_ERROR_BACKEND | RWKV_ERROR_UNSUPPORTED;
}

int qnn_backend::eval(std::vector<int> ids, std::vector<float> &logits) {
    return RWKV_ERROR_BACKEND | RWKV_ERROR_UNSUPPORTED;
}

int qnn_backend::clear_state() {
    return RWKV_ERROR_BACKEND | RWKV_ERROR_UNSUPPORTED;
}

int qnn_backend::get_state(std::any &state) {
    return RWKV_ERROR_BACKEND | RWKV_ERROR_UNSUPPORTED;
}

int qnn_backend::set_state(std::any state) {
    return RWKV_ERROR_BACKEND | RWKV_ERROR_UNSUPPORTED;
}

int qnn_backend::free_state(std::any state) {
    return RWKV_ERROR_BACKEND | RWKV_ERROR_UNSUPPORTED;
}

bool qnn_backend::is_available() {
    return false;
}

int qnn_backend::release_model() {
    return RWKV_ERROR_BACKEND | RWKV_ERROR_UNSUPPORTED;
}

int qnn_backend::release() {
    return RWKV_ERROR_BACKEND | RWKV_ERROR_UNSUPPORTED;
}

#endif

} // namespace rwkvmobile