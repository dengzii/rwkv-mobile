//
// Created by dengzi on 2025/7/14.
//

#ifndef LLAMA_CPP_EMBEDDING_H
#define LLAMA_CPP_EMBEDDING_H

#include "llama.h"
#include <string>
#include <vector>

class rwkv_embedding {
public:
    explicit rwkv_embedding();

    ~rwkv_embedding();

    int load_model(const std::string &model_path);

    std::vector<float> embed(const std::string &text, int embd_norm = 2) const;

    static float similarity(const std::vector<float> &emb1, const std::vector<float> &emb2);

private:
    llama_model *model;
    llama_context *ctx;

    static void embd_normalize(const float *inp, float *out, int n, int embd_norm = -1);

    static void batch_add_seq(llama_batch &batch, const std::vector<int32_t> &tokens, llama_seq_id seq_id);

    void batch_decode(const llama_batch &batch, float *output, int embd_norm) const;

    std::vector<int32_t> tokenize(const std::string &text) const;

    void release();
};

#endif // LLAMA_CPP_EMBEDDING_H
