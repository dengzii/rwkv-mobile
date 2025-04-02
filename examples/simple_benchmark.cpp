#include <iostream>
#include <chrono>
#include <random>
#include <vector>

#include "commondef.h"
#include "runtime.h"

#define ENSURE_SUCCESS_OR_LOG_EXIT(x, msg) if (x != rwkvmobile::RWKV_SUCCESS) { std::cout << msg << std::endl; return 1; }

int main(int argc, char **argv) {
    // set stdout to be unbuffered
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <model_file> <backend>" << std::endl;
        return 1;
    }

    rwkvmobile::runtime runtime;
    ENSURE_SUCCESS_OR_LOG_EXIT(runtime.init(argv[2]), "Failed to initialize runtime");
    ENSURE_SUCCESS_OR_LOG_EXIT(runtime.load_model(argv[1]), "Failed to load model");

    int vocab_size = runtime.get_vocab_size();

    std::vector<int> prompt_ids(512);
    for (int i = 0; i < 512; i++) {
        prompt_ids[i] = rand() % vocab_size;
    }
    float *logits = nullptr;
    runtime.eval_logits(prompt_ids, logits);
    runtime.free_logits_if_allocated(logits);

    std::cout << "Prefill speed: " << runtime.get_avg_prefill_speed() << " tokens/s" << std::endl;

    for (int i = 0; i < 128; i++) {
        runtime.eval_logits(rand() % vocab_size, logits);
        runtime.free_logits_if_allocated(logits);
    }
    std::cout << "Decode speed: " << runtime.get_avg_decode_speed() << " tokens/s" << std::endl;

    runtime.release();

    return 0;
}
