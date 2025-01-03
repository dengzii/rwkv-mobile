#ifndef C_API_H
#define C_API_H

typedef void * rwkvmobile_runtime_t;

#ifdef __cplusplus
extern "C" {
#endif

// ============================
// init runtime with backend name
// returns: runtime handle
rwkvmobile_runtime_t rwkvmobile_runtime_init_with_name(const char * backend_name);

int rwkvmobile_runtime_release(rwkvmobile_runtime_t runtime);

// ============================
// load model file
// args: runtime handle, model file path
// returns: Error codes
int rwkvmobile_runtime_load_model(rwkvmobile_runtime_t runtime, const char * model_path);

// ============================
// load tokenizer from vocab_file
// args: runtime handle, vocab_file path
// returns: Error codes
int rwkvmobile_runtime_load_tokenizer(rwkvmobile_runtime_t runtime, const char * vocab_file);

int rwkvmobile_runtime_eval_logits(rwkvmobile_runtime_t runtime, const int *ids, int ids_len, float * logits, int logits_len);

int rwkvmobile_runtime_eval_chat(rwkvmobile_runtime_t runtime, const char * input, char * response, const int max_length, void (*callback)(const char *));

int rwkvmobile_runtime_eval_chat_with_history(rwkvmobile_runtime_t handle, const char ** inputs, const int num_inputs, char * response, const int max_length, void (*callback)(const char *));

int rwkvmobile_runtime_gen_completion(rwkvmobile_runtime_t runtime, const char * prompt, char * completion, const int length);


// ============================
// clear state
// args: runtime handle
// returns: Error codes
int rwkvmobile_runtime_clear_state(rwkvmobile_runtime_t runtime);

int rwkvmobile_runtime_get_available_backend_names(rwkvmobile_runtime_t handle, char * backend_names_buffer, int buffer_size);

#ifdef __cplusplus
}
#endif

#endif // C_API_H