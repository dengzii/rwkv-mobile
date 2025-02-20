#include <iostream>
#include <chrono>

#include "commondef.h"
#include "runtime.h"
#include "c_api.h"

#define ENSURE_SUCCESS_OR_LOG_EXIT(x, msg) if (x != rwkvmobile::RWKV_SUCCESS) { std::cout << msg << std::endl; return 1; }

void callback(const char *msg, const int) {
    std::cout << msg;
}

int main(int argc, char **argv) {
    // set stdout to be unbuffered
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <vocab_file> <model_file> <backend>" << std::endl;
        return 1;
    }

    rwkvmobile_runtime_t runtime = rwkvmobile_runtime_init_with_name(argv[3]);
    rwkvmobile_runtime_load_tokenizer(runtime, argv[1]);
    rwkvmobile_runtime_load_model(runtime, argv[2]);
    rwkvmobile_runtime_set_penalty_params(runtime, {0, 0, 0});
    rwkvmobile_runtime_set_sampler_params(runtime, {1.0, 1, 1.0});

    rwkvmobile_runtime_release(runtime);

    runtime = rwkvmobile_runtime_init_with_name(argv[3]);
    rwkvmobile_runtime_load_tokenizer(runtime, argv[1]);
    rwkvmobile_runtime_load_model(runtime, argv[2]);
    rwkvmobile_runtime_set_penalty_params(runtime, {0, 0, 0});
    rwkvmobile_runtime_set_sampler_params(runtime, {1.0, 1, 1.0});
    rwkvmobile_runtime_release(runtime);

    runtime = rwkvmobile_runtime_init_with_name(argv[3]);
    rwkvmobile_runtime_load_tokenizer(runtime, argv[1]);
    rwkvmobile_runtime_load_model(runtime, argv[2]);
    rwkvmobile_runtime_set_penalty_params(runtime, {0, 0, 0});
    rwkvmobile_runtime_set_sampler_params(runtime, {1.0, 1, 1.0});
    rwkvmobile_runtime_release(runtime);

    runtime = rwkvmobile_runtime_init_with_name(argv[3]);
    rwkvmobile_runtime_load_tokenizer(runtime, argv[1]);
    rwkvmobile_runtime_load_model(runtime, argv[2]);
    rwkvmobile_runtime_set_penalty_params(runtime, {0, 0, 0});
    rwkvmobile_runtime_set_sampler_params(runtime, {1.0, 1, 1.0});
    rwkvmobile_runtime_release(runtime);
    return 0;
}
