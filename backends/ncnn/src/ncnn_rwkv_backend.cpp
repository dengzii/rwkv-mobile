#include <fstream>
#include <filesystem>

#include "backend.h"
#include "ncnn_rwkv_backend.h"
#include "commondef.h"

#ifdef ENABLE_NCNN
#include "net.h"
#include "mat.h"
#endif

namespace rwkvmobile {

#ifdef ENABLE_NCNN
int ncnn_rwkv_backend::init(void * extra) {
    return RWKV_SUCCESS;
}

int ncnn_rwkv_backend::load_model(std::string model_path) {
    if (!std::filesystem::exists(model_path)) {
        return RWKV_ERROR_MODEL | RWKV_ERROR_IO;
    }
    auto remove_extension = [](std::string path) {
        size_t lastindex = path.find_last_of(".");
        return path.substr(0, lastindex);
    };
    std::string param_path = remove_extension(model_path) + ".param";
    std::string bin_path = remove_extension(model_path) + ".bin";

    int ret = 0;
    ret = net.load_param(param_path.c_str());
    if (ret == -1) {
        return RWKV_ERROR_MODEL | RWKV_ERROR_IO;
    }
    ret = net.load_model(bin_path.c_str());
    if (ret == -1) {
        return RWKV_ERROR_MODEL | RWKV_ERROR_IO;
    }

    ncnn::Extractor ex = net.create_extractor();
    ncnn::Mat model_info;
    ex.extract("model_info", model_info);
    version = model_info[0];
    n_layers = model_info[1];
    num_heads = model_info[2];
    head_size = model_info[3];
    vocab_size = model_info[4];

    for (int i = 0; i < n_layers; i++) {
        states.push_back(ncnn::Mat(num_heads * head_size));
        states.push_back(ncnn::Mat(head_size, head_size, num_heads));
        states.push_back(ncnn::Mat(num_heads * head_size));
    }

    for (auto &state : states) {
        state.fill(0.0f);
    }

    return RWKV_SUCCESS;
}

int ncnn_rwkv_backend::eval(int id, std::vector<float> &logits) {
    int token = id;
    ncnn::Mat input = ncnn::Mat(1, &token);
    ncnn::Extractor ex = net.create_extractor();

    for (int i = 0; i < n_layers; i++) {
        ex.input(("state_" + std::to_string(3 * i) + "_in").c_str(), states[i * 3]);
        ex.input(("state_" + std::to_string(3 * i + 1) + "_in").c_str(), states[i * 3 + 1]);
        ex.input(("state_" + std::to_string(3 * i + 2) + "_in").c_str(), states[i * 3 + 2]);
    }
    ex.input("token", input);

    for (int i = 0; i < n_layers; i++) {
        ex.extract(("state_" + std::to_string(3 * i) + "_out").c_str(), states[i * 3]);
        ex.extract(("state_" + std::to_string(3 * i + 1) + "_out").c_str(), states[i * 3 + 1]);
        ex.extract(("state_" + std::to_string(3 * i + 2) + "_out").c_str(), states[i * 3 + 2]);
    }
    ncnn::Mat logits_mat;
    ex.extract("logits", logits_mat);
    memcpy(logits.data(), logits_mat.channel(0), logits.size() * sizeof(float));

    return RWKV_SUCCESS;
}

int ncnn_rwkv_backend::eval(std::vector<int> ids, std::vector<float> &logits) {
    // TODO: sequential prefill
    for (int i = 0; i < ids.size(); i++) {
        int id = ids[i];
        ncnn::Mat input = ncnn::Mat(1, &id);
        ncnn::Extractor ex = net.create_extractor();

        for (int i = 0; i < n_layers; i++) {
            ex.input(("state_" + std::to_string(3 * i) + "_in").c_str(), states[i * 3]);
            ex.input(("state_" + std::to_string(3 * i + 1) + "_in").c_str(), states[i * 3 + 1]);
            ex.input(("state_" + std::to_string(3 * i + 2) + "_in").c_str(), states[i * 3 + 2]);
        }
        ex.input("token", input);

        if (i == ids.size() - 1) {
            ncnn::Mat logits_mat;
            ex.extract("logits", logits_mat);
            memcpy(logits.data(), logits_mat.channel(0), logits.size() * sizeof(float));
        }
        for (int i = 0; i < n_layers; i++) {
            ex.extract(("state_" + std::to_string(3 * i) + "_out").c_str(), states[i * 3]);
            ex.extract(("state_" + std::to_string(3 * i + 1) + "_out").c_str(), states[i * 3 + 1]);
            ex.extract(("state_" + std::to_string(3 * i + 2) + "_out").c_str(), states[i * 3 + 2]);
        }
    }

    return RWKV_SUCCESS;
}

bool ncnn_rwkv_backend::is_available() {
    // always available
    return true;
}

#else

int ncnn_rwkv_backend::init(void * extra) {
    return RWKV_ERROR_BACKEND | RWKV_ERROR_UNSUPPORTED;
}

int ncnn_rwkv_backend::load_model(std::string model_path) {
    return RWKV_ERROR_BACKEND | RWKV_ERROR_UNSUPPORTED;
}

int ncnn_rwkv_backend::eval(int id, std::vector<float> &logits) {
    return RWKV_ERROR_BACKEND | RWKV_ERROR_UNSUPPORTED;
}

int ncnn_rwkv_backend::eval(std::vector<int> ids, std::vector<float> &logits) {
    return RWKV_ERROR_BACKEND | RWKV_ERROR_UNSUPPORTED;
}

bool ncnn_rwkv_backend::is_available() {
    return false;
}

#endif

} // namespace rwkvmobile