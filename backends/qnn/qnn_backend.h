#ifndef QNN_BACKEND_H
#define QNN_BACKEND_H

#include "backend.h"
#include "rwkv-qualcomm/Interfaces.hpp"
#include "rwkv-qualcomm/Utils/IOTensor.hpp"

namespace rwkvmobile {

class qnn_backend : public execution_provider {
public:
    ~qnn_backend() {
        release_model();
        release();
    }
    int init(void * extra) override;
    int load_model(std::string model_path) override;
    int eval(int id, std::vector<float> &logits) override;
    int eval(std::vector<int> ids, std::vector<float> &logits) override;
    bool is_available() override;
    int clear_state() override;
    int get_state(std::any &state) override;
    int set_state(std::any state) override;
    int free_state(std::any state) override;
    int release_model() override;
    int release() override;

private:
    std::string qnnBackendPath;
    void *qnnBackendLibraryHandle = nullptr;
    void *qnnModelHandle = nullptr;

    uint32_t powerConfigId;
    uint32_t deviceId = 0;
    uint32_t coreId = 0;

    bool isContextCreated = false;

    qnn::tools::rwkv_app::QnnFunctionPointers qnnFunctionPointers;

    Qnn_LogHandle_t qnnLogHandle = nullptr;
    Qnn_BackendHandle_t qnnBackendHandle = nullptr;
    Qnn_DeviceHandle_t qnnDeviceHandle = nullptr;
    std::vector<Qnn_ContextHandle_t> qnnContextHandles;

    uint32_t qnnGraphsCount = 0;
    qnn_wrapper_api::GraphInfo_t **qnnGraphsInfo = nullptr;
    uint32_t graphConfigsInfoCount = 0;
    qnn_wrapper_api::GraphConfigInfo_t **graphConfigsInfo = nullptr;


    std::vector<std::vector<int>> stateCopyMap;
    std::vector<int> inputIdx;
    std::vector<int> outputIdx;
    std::vector<int> vfirstInIdx;
    std::vector<int> vfirstOutIdx;

    Qnn_Tensor_t *inputTensors[8] = {nullptr};
    Qnn_Tensor_t *outputTensors[8] = {nullptr};

    qnn::tools::iotensor::IOTensor qnnIOTensorUtils;

    int qnn_create_power_config_id();
    int qnn_destory_power_config_id();
    int qnn_set_power_config();
    int qnn_register_op_package(std::string package_path, std::string interface_provider);
    int qnn_set_rpc_latency_and_polling();
    int qnn_initialize_tensors();
};

}

#endif
