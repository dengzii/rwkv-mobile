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
    int eval(int id, float *& logits) override;
    int eval(std::vector<int> ids, float *& logits, bool skip_logits_copy = false) override;
    void free_logits_if_allocated(float *& logits) override {
        return;
    };
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
    ncnn::Mat logits_mat;
};

}

#endif
