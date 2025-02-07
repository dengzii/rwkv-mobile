#ifndef NCNN_RWKV_BACKEND_H
#define NCNN_RWKV_BACKEND_H

#include "backend.h"
#include "net.h"
#include "mat.h"

namespace rwkvmobile {

class ncnn_rwkv_backend : public execution_provider {
public:
    int init(void * extra) override;
    int load_model(std::string model_path) override;
    int eval(int id, std::vector<float> &logits) override;
    int eval(std::vector<int> ids, std::vector<float> &logits) override;
    bool is_available() override;
    int get_state(std::any &state) override;
    int set_state(std::any state) override;
    int free_state(std::any state) override;
    int clear_state() override;
    int release_model() override;
    int release() override;

private:
    ncnn::Net net;
    std::vector<ncnn::Mat> states;
};

}

#endif
