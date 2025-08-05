//
// Created by dengzi on 2025/7/14.
//

#include "rwkv_embedding.h"
#include <cmath>

#include "llama-context.h"
#include "llama-model.h"
#include "llama-vocab.h"
#include "logger.h"

struct result {
    std::string text;
    std::vector<llama_token> tokens;
    std::vector<float> embedding;
};

static auto batch_add(llama_batch &batch,
                      const llama_token id,
                      const llama_pos pos,
                      const std::vector<llama_seq_id> &seq_ids,
                      const bool logits) -> void {
    if (seq_ids.size() > 8) {
        // LLAMA_MAX_SEQS is typically 8
        rwkvmobile::LOGE("too many sequence IDs: %zu", seq_ids.size());
        return;
    }

    batch.token[batch.n_tokens] = id;
    batch.pos[batch.n_tokens] = pos;
    batch.n_seq_id[batch.n_tokens] = seq_ids.size();
    for (size_t i = 0; i < seq_ids.size(); ++i) {
        batch.seq_id[batch.n_tokens][i] = seq_ids[i];
    }
    batch.logits[batch.n_tokens] = logits;

    batch.n_tokens++;
}


rwkv_embedding::rwkv_embedding(): model(nullptr), ctx(nullptr),
                                  model_rerank(nullptr), ctx_rerank(nullptr) {
}


int rwkv_embedding::load_model(const std::string &model_path) {
    release();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model) {
        release();
        return 1;
    }
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.embeddings = true;
    ctx_params.no_perf = true;
    // ctx_params.n_ctx = 32768;

    ctx = llama_init_from_model(model, ctx_params);
    rwkvmobile::LOGI("embedding model loaded: %s, pooling_type: %d, dim: %d", model->name.c_str(), ctx->pooling_type(),
                     llama_model_n_embd(model));
    if (!ctx) {
        release();
        return 2;
    }
    return 0;
}


int rwkv_embedding::load_rerank_model(const std::string &model_path) {
    llama_model_params lparams = llama_model_default_params();
    lparams.n_gpu_layers = 0;
    model_rerank = llama_model_load_from_file(model_path.c_str(), lparams);
    if (!model_rerank) {
        rwkvmobile::LOGE("Failed to load rerank model");
        return 1;
    }

    llama_context_params params = llama_context_default_params();
    params.pooling_type = LLAMA_POOLING_TYPE_RANK;
    // params.n_batch = 1024;
    // params.n_ctx = 2048;

    ctx_rerank = llama_init_from_model(model_rerank, params);
    if (!ctx_rerank) {
        rwkvmobile::LOGE("Failed to initialize rerank context");
        llama_model_free(model_rerank);
        model_rerank = nullptr;
        return 2;
    }
    rwkvmobile::LOGI("rerank model loaded: %s", model_rerank->name.c_str());
    return 0;
}

void rwkv_embedding::release() {
    if (ctx) {
        llama_free(ctx);
    }
    if (model) {
        llama_model_free(model);
        rwkvmobile::LOGI("release embedding model");
    }
    if (ctx_rerank) {
        llama_free(ctx_rerank);
    }
    if (model_rerank) {
        llama_model_free(model_rerank);
        rwkvmobile::LOGI("release rerank model");
    }
    ctx = nullptr;
    model = nullptr;
    ctx_rerank = nullptr;
    model_rerank = nullptr;
}


std::vector<int32_t> rwkv_embedding::tokenize(const llama_model *model,
                                              const std::string &text,
                                              const bool add_special,
                                              const bool parse_special,
                                              const bool add_eos,
                                              const bool add_sep) {
    const auto *vocab = llama_model_get_vocab(model);
    if (!vocab) {
        rwkvmobile::LOGE("vocab is null");
        return {};
    }

    std::vector<llama_token> tokens = vocab->tokenize(text, add_special, parse_special);

    if (add_eos) {
        if (llama_vocab_eos(vocab) >= 0 && (tokens.empty() || tokens.back() != llama_vocab_eos(vocab))) {
            tokens.push_back(llama_vocab_eos(vocab));
        }
    }

    if (add_sep) {
        if (llama_vocab_sep(vocab) >= 0 && (tokens.empty() || tokens.back() != llama_vocab_sep(vocab))) {
            tokens.push_back(llama_vocab_sep(vocab));
        }
    }

    return std::vector<int32_t>(tokens.begin(), tokens.end());
}

void rwkv_embedding::embd_normalize(const float *inp, float *out, const int n, const int normalization) {
    if (!inp || !out || n <= 0) {
        rwkvmobile::LOGE("invalid parameters in embd_normalize");
        return;
    }

    double sum = 0.0;
    switch (normalization) {
        case -1: // no normalisation
            sum = 1.0;
            break;
        case 0: // max absolute
            for (int i = 0; i < n; i++) {
                if (sum < std::abs(inp[i])) {
                    sum = std::abs(inp[i]);
                }
            }
            if (sum > 0.0) {
                sum /= 32760.0; // make an int16 range
            }
            break;
        case 2: // euclidean
            for (int i = 0; i < n; i++) {
                sum += inp[i] * inp[i];
            }
            sum = std::sqrt(sum);
            break;
        default: // p-norm (euclidean is p-norm p=2)
            for (int i = 0; i < n; i++) {
                sum += std::pow(std::abs(inp[i]), normalization);
            }
            sum = std::pow(sum, 1.0 / normalization);
            break;
    }
    const float norm = sum > 0.0 ? static_cast<float>(1.0 / sum) : 0.0f;
    for (int i = 0; i < n; i++) {
        out[i] = inp[i] * norm;
    }
}

auto rwkv_embedding::batch_add_seq(llama_batch &batch, const std::vector<int32_t> &tokens,
                                   const llama_seq_id seq_id) -> int {
    const size_t n_tokens = tokens.size();
    if (n_tokens == 0) {
        return 0;
    }

    for (size_t i = 0; i < n_tokens; i++) {
        batch_add(batch, tokens[i], i, {seq_id}, true);
    }
    return 0;
}

int rwkv_embedding::batch_decode(llama_context *ctx, const llama_batch &batch, float *output, const int n_embd,
                                 const int normalization) {
    if (!ctx) {
        rwkvmobile::LOGE("context is null");
        return -1;
    }
    if (!output) {
        rwkvmobile::LOGE("output is null");
        return -1;
    }

    if (llama_decode(ctx, batch) < 0) {
        rwkvmobile::LOGE("failed to decode");
        return -1;
    }
    for (int i = 0; i < batch.n_tokens; i++) {
        if (!batch.logits[i]) {
            continue;
        }
        const float *embd = llama_get_embeddings_seq(ctx, batch.seq_id[i][0]);
        if (embd == nullptr) {
            embd = llama_get_embeddings_ith(ctx, i);
            if (embd == nullptr) {
                rwkvmobile::LOGE("failed to get embeddings");
                return -1;
            }
        }
        float *out = output + batch.seq_id[i][0] * n_embd;
        embd_normalize(embd, out, n_embd, normalization);
    }
    llama_memory_clear(llama_get_memory(ctx), true);
    return 0;
}

std::vector<std::vector<float> > rwkv_embedding::get_embeddings(const std::vector<std::string> &inputs) const {
    if (!model) {
        rwkvmobile::LOGE("model not loaded");
        return {};
    }
    if (!ctx) {
        rwkvmobile::LOGE("context not initialized");
        return {};
    }

    std::vector<result> results;

    // max batch size
    const auto n_batch = static_cast<int>(llama_n_batch(ctx));

    for (auto &input: inputs) {
        auto tokens = tokenize(model, input, true, false, true, false);
        if (n_batch < static_cast<int>(tokens.size())) {
            rwkvmobile::LOGE("input size exceeds max batch size");
            return {};
        }
        results.push_back(result{input, tokens, {}});
    }

    const auto n_seq_max = static_cast<int>(llama_n_seq_max(ctx));
    const auto n_embd = llama_model_n_embd(model);
    const auto input_size = results.size();

    llama_batch batch = llama_batch_init(n_batch, 0, n_seq_max);

    std::vector<float> embeddings(input_size * n_embd, 0);
    float *emb = embeddings.data();

    int p = 0; // number of prompts processed already
    int s = 0; // number of prompts in current batch
    for (int k = 0; k < input_size; k++) {
        // clamp to n_batch tokens
        auto &inp = results[k].tokens;
        const uint64_t n_toks = inp.size();

        // encode if at capacity
        if (batch.n_tokens + n_toks > n_batch) {
            float *out = emb + p * n_embd;
            if (const auto r = batch_decode(ctx, batch, out, n_embd, 2); r < 0) {
                rwkvmobile::LOGE("failed to decode batch");
                return {};
            }
            batch.n_tokens = 0;
            p += s;
            s = 0;
        }
        // add to batch
        batch_add_seq(batch, inp, s);
        s += 1;
    }

    float *out = emb + p * n_embd;
    const auto ret = batch_decode(ctx, batch, out, n_embd, 2);
    if (ret < 0) {
        rwkvmobile::LOGE("failed to decode batch");
        return {};
    }

    std::vector<std::vector<float> > result_embeddings;
    for (int i = 0; i < input_size; i++) {
        results[i].embedding = std::vector<float>(emb + i * n_embd, emb + (i + 1) * n_embd);
        results[i].tokens.clear();
        result_embeddings.push_back(results[i].embedding);
    }
    llama_batch_free(batch);

    return result_embeddings;
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

std::vector<llama_token> truncate_tokens(const std::vector<llama_token> &tokens, int limit_size,
                                         llama_token eos_token) {
    std::vector<llama_token> new_tokens = tokens;

    if ((int) tokens.size() > limit_size) {
        rwkvmobile::LOGE("token size exceeds limit");
        new_tokens.resize(limit_size);
    }
    // add eos if not present
    if (new_tokens.empty() || new_tokens.back() != eos_token) {
        new_tokens.push_back(eos_token);
    }
    return new_tokens;
}


float rwkv_embedding::rank_document(const std::string &query, const std::string &document) const {
    if (!model_rerank || !ctx_rerank) {
        rwkvmobile::LOGE("rerank model not loaded");
        return 0.0f;
    }

    std::vector<llama_token> tokens;

    const auto n_batch = llama_n_batch(ctx_rerank);
    const auto *vocab = llama_model_get_vocab(model_rerank);
    const auto tokenBos = llama_vocab_bos(vocab);
    const auto tokenEos = llama_vocab_eos(vocab);
    const auto tokenSep = llama_vocab_sep(vocab);

    tokens.reserve(n_batch);

    tokens.push_back(tokenBos);

    auto part1 = tokenize(model_rerank, query, false, true, false, false);
    part1 = truncate_tokens(part1, static_cast<int>(n_batch / 2) - 2, tokenEos);
    tokens.insert(tokens.end(), part1.begin(), part1.end());

    tokens.push_back(tokenEos);
    tokens.push_back(tokenSep);

    auto part2 = tokenize(model_rerank, document, false, true, false, false);
    part2 = truncate_tokens(part2, static_cast<int>(n_batch / 2) - 2, tokenEos);
    tokens.insert(tokens.end(), part2.begin(), part2.end());

    tokens.push_back(tokenEos);

    //////////////

    const int n_ctx = ctx_rerank->n_ctx();
    const int n_embd = llama_model_n_embd(model_rerank);

    if (static_cast<int>(tokens.size()) > n_ctx) {
        rwkvmobile::LOGE("input size exceeds max context");
        return 0.0f;
    }

    llama_batch batch = llama_batch_init(n_batch, 0, 1);

    batch_add_seq(batch, tokens, 0);

    if (llama_encode(this->ctx_rerank, batch)) {
        rwkvmobile::LOGE("llama_encode failed");
        llama_batch_free(batch);
        return 0.0f;
    }


    std::vector<float> embeddings(n_embd, 0.0f);

    for (int i = 0; i < batch.n_tokens; ++i) {
        if (!batch.logits[i]) {
            continue;
        }
        float *embd = llama_get_embeddings_seq(this->ctx_rerank, batch.seq_id[i][0]);
        if (embd == nullptr) {
            embd = llama_get_embeddings_ith(this->ctx_rerank, i);
        }
        if (embd == nullptr) {
            rwkvmobile::LOGE("Failed to get embeddings");
            continue;
        }
        embd_normalize(embd, embeddings.data(), n_embd, -1);
    }

    llama_batch_free(batch);

    return embeddings[0];
}


std::vector<float> rwkv_embedding::rerank(const std::string &query, const std::vector<std::string> &documents) const {
    std::vector<float> scores;
    scores.reserve(documents.size());
    for (const std::string &doc: documents) {
        scores.push_back(rank_document(query, doc));
    }
    return scores;
}
