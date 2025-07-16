//
// Created by dengzi on 2025/7/14.
//

#include "rwkv_embedding.h"
#include <stdexcept>
#include <cmath>

#include "llama-vocab.h"


rwkv_embedding::rwkv_embedding(): model(nullptr), ctx(nullptr) {
}


int rwkv_embedding::load_model(const std::string &model_path) {
    if (model) {
        llama_model_free(model);
    }
    if (ctx) {
        llama_free(ctx);
    }

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model) {
        release();
        return 1;
    }
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.embeddings = true;
    ctx_params.pooling_type = LLAMA_POOLING_TYPE_MEAN;
    ctx_params.no_perf = true;
    ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        release();
        return 1;
    }
    return 0;
}

rwkv_embedding::~rwkv_embedding() {
    if (ctx) {
        llama_free(ctx);
    }
    if (model) {
        llama_model_free(model);
    }
}

void rwkv_embedding::release() {
    if (ctx) {
        llama_free(ctx);
    }
    if (model) {
        llama_model_free(model);
    }
    ctx = nullptr;
    model = nullptr;
}


std::vector<int32_t> rwkv_embedding::tokenize(const std::string &text) const {
    const auto *vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens = vocab->tokenize(text, true, true);
    return std::vector<int32_t>(tokens.begin(), tokens.end());
}

void rwkv_embedding::embd_normalize(const float *inp, float *out, const int n, const int embd_norm) {
    double sum = 0.0;
    switch (embd_norm) {
        case -1: // no normalisation
            sum = 1.0;
            break;
        case 0: // max absolute
            for (int i = 0; i < n; i++) {
                if (sum < std::abs(inp[i])) {
                    sum = std::abs(inp[i]);
                }
            }
            sum /= 32760.0; // make an int16 range
            break;
        case 2: // euclidean
            for (int i = 0; i < n; i++) {
                sum += inp[i] * inp[i];
            }
            sum = std::sqrt(sum);
            break;
        default: // p-norm (euclidean is p-norm p=2)
            for (int i = 0; i < n; i++) {
                sum += std::pow(std::abs(inp[i]), embd_norm);
            }
            sum = std::pow(sum, 1.0 / embd_norm);
            break;
    }
    const float norm = sum > 0.0 ? static_cast<float>(1.0 / sum) : 0.0f;
    for (int i = 0; i < n; i++) {
        out[i] = inp[i] * norm;
    }
}

void rwkv_embedding::batch_add_seq(llama_batch &batch, const std::vector<int32_t> &tokens, const llama_seq_id seq_id) {
    const size_t n_tokens = tokens.size();
    for (size_t i = 0; i < n_tokens; i++) {
        const std::vector<llama_seq_id> &seq_ids = {seq_id};
        if (!batch.seq_id[batch.n_tokens]) {
            throw std::runtime_error("llama_batch size exceeded");
        }
        batch.token[batch.n_tokens] = tokens[i];
        batch.pos[batch.n_tokens] = static_cast<int>(i);
        batch.n_seq_id[batch.n_tokens] = static_cast<int>(seq_ids.size());
        for (size_t j = 0; j < seq_ids.size(); ++j) {
            batch.seq_id[batch.n_tokens][j] = seq_ids[j];
        }
        batch.logits[batch.n_tokens] = true;
        batch.n_tokens++;
    }
}

void rwkv_embedding::batch_decode(const llama_batch &batch, float *output, const int embd_norm) const {
    const enum llama_pooling_type pooling_type = llama_pooling_type(ctx);
    // clear previous kv_cache values (irrelevant for embeddings)
    llama_memory_clear(llama_get_memory(ctx), true);

    const auto n_embd = llama_model_n_embd(model);

    // run model
    if (llama_decode(ctx, batch) < 0) {
        throw std::runtime_error("failed to process");
    }
    for (int i = 0; i < batch.n_tokens; i++) {
        if (!batch.logits[i]) {
            continue;
        }
        const float *embd = nullptr;
        int embd_pos = 0;

        if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
            // try to get token embeddings
            embd = llama_get_embeddings_ith(ctx, i);
            embd_pos = i;
            if (!embd) {
                throw std::runtime_error("failed to get token embeddings");
            }
        } else {
            // try to get sequence embeddings - supported only when pooling_type is not NONE
            embd = llama_get_embeddings_seq(ctx, batch.seq_id[i][0]);
            embd_pos = batch.seq_id[i][0];
            if (!embd) {
                throw std::runtime_error("failed to get sequence embeddings");
            }
        }

        float *out = output + embd_pos * n_embd;
        embd_normalize(embd, out, n_embd, embd_norm);
    }
}

std::vector<float> rwkv_embedding::embed(const std::string &text, const int embd_norm) const {
    const std::vector<int32_t> tokens = tokenize(text);
    if (tokens.empty()) {
        return {};
    }

    const auto n_batch = static_cast<int>(llama_n_batch(ctx));
    const auto n_seq_max = static_cast<int>(llama_n_seq_max(ctx));
    llama_batch batch = llama_batch_init(n_batch, 0, n_seq_max);

    batch_add_seq(batch, tokens, 0);

    const enum llama_pooling_type pooling_type = llama_pooling_type(ctx);

    const int n_embd_count = (pooling_type == LLAMA_POOLING_TYPE_NONE) ? tokens.size() : 1;
    const auto n_embd = llama_model_n_embd(model);
    std::vector<float> embeddings(n_embd_count * n_embd, 0);

    batch_decode(batch, embeddings.data(), embd_norm);
    llama_batch_free(batch);

    return embeddings;
}

float rwkv_embedding::similarity(const std::vector<float> &emb1, const std::vector<float> &emb2) {
    if (emb1.size() != emb2.size() || emb1.empty()) {
        return 0.0f;
    }

    double sum = 0.0, sum1 = 0.0, sum2 = 0.0;
    for (size_t i = 0; i < emb1.size(); i++) {
        sum += emb1[i] * emb2[i];
        sum1 += emb1[i] * emb1[i];
        sum2 += emb2[i] * emb2[i];
    }

    if (sum1 == 0.0 || sum2 == 0.0) {
        return (sum1 == 0.0 && sum2 == 0.0) ? 1.0f : 0.0f;
    }

    return static_cast<float>(sum / (std::sqrt(sum1) * std::sqrt(sum2)));
}
