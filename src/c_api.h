#ifndef C_API_H
#define C_API_H

typedef void * rwkvmobile_runtime_t;

struct sampler_params {
    float temperature;
    int top_k;
    float top_p;
};

struct penalty_params {
    float presence_penalty;
    float frequency_penalty;
    float penalty_decay;
};

#ifdef __cplusplus
extern "C" {
#endif

int rwkvmobile_runtime_get_available_backend_names(char * backend_names_buffer, int buffer_size);

rwkvmobile_runtime_t rwkvmobile_runtime_init_with_name(const char * backend_name);

int rwkvmobile_runtime_release(rwkvmobile_runtime_t runtime);

int rwkvmobile_runtime_load_model(rwkvmobile_runtime_t runtime, const char * model_path);

int rwkvmobile_runtime_load_tokenizer(rwkvmobile_runtime_t runtime, const char * vocab_file);

int rwkvmobile_runtime_eval_logits(rwkvmobile_runtime_t runtime, const int *ids, int ids_len, float * logits, int logits_len);

int rwkvmobile_runtime_eval_chat(rwkvmobile_runtime_t runtime, const char * input, char * response, const int max_length, void (*callback)(const char *));

int rwkvmobile_runtime_eval_chat_with_history(rwkvmobile_runtime_t handle, const char ** inputs, const int num_inputs, char * response, const int max_length, void (*callback)(const char *));

int rwkvmobile_runtime_set_prompt(rwkvmobile_runtime_t runtime, const char * prompt);

int rwkvmobile_runtime_get_prompt(rwkvmobile_runtime_t runtime, char * prompt, const int buf_len);

int rwkvmobile_runtime_gen_completion(rwkvmobile_runtime_t runtime, const char * prompt, char * completion, const int max_length, const int stop_code, void (*callback)(const char *));

int rwkvmobile_runtime_clear_state(rwkvmobile_runtime_t runtime);

struct sampler_params rwkvmobile_runtime_get_sampler_params(rwkvmobile_runtime_t runtime);

void rwkvmobile_runtime_set_sampler_params(rwkvmobile_runtime_t runtime, struct sampler_params params);

struct penalty_params rwkvmobile_runtime_get_penalty_params(rwkvmobile_runtime_t runtime);

void rwkvmobile_runtime_set_penalty_params(rwkvmobile_runtime_t runtime, struct penalty_params params);

#ifdef __cplusplus
}
#endif

#endif // C_API_H