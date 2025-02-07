#ifndef LLAMA_CPP_BACKEND_H
#define LLAMA_CPP_BACKEND_H

#include "backend.h"
#include "llama.h"

namespace rwkvmobile {

class llama_cpp_backend : public execution_provider {
public:
    int init(void * extra) override;
    int load_model(std::string model_path) override;
    int eval(int id, std::vector<float> &logits) override;
    int eval(std::vector<int> ids, std::vector<float> &logits) override;
    bool is_available() override;
    int clear_state() override;
    int get_state(std::any &state) override;
    int set_state(std::any state) override;
    int free_state(std::any state) override;

private:
    llama_model * model;
    llama_context * ctx;
};

}

#endif
