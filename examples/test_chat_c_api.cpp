#include <iostream>
#include <chrono>
#include <unistd.h>
#include "commondef.h"
#include "c_api.h"

#define ENSURE_SUCCESS_OR_LOG_EXIT(x, msg) if (x != rwkvmobile::RWKV_SUCCESS) { std::cout << msg << std::endl; return 1; }

std::string response;
void callback(const char *msg, const int) {
    // std::cout << "Callback: " << msg << std::endl;
    response = std::string(msg);
};

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
    rwkvmobile_runtime_set_sampler_params(runtime, {1.0, 1, 1.0});

    std::cout << "Testing original input list" << std::endl;
    char *input_list[] = {
        "Hello!",
        "Hello! I'm your AI assistant. I'm here to help you with various tasks, such as answering questions, brainstorming ideas, drafting emails, writing code, providing advice, and much more.",
        "What's the weather like today?",
    };

    rwkvmobile_runtime_eval_chat_with_history(runtime, (const char **)input_list, 3, 50, callback, 0);
    while (rwkvmobile_runtime_is_generating(runtime)) {
        std::cout << "Waiting for generation to finish..." << std::endl;
        sleep(1);
    }
    std::cout << "Response: " << response << std::endl;

    input_list[2] = "Write me a poem";
    rwkvmobile_runtime_eval_chat_with_history(runtime, (const char **)input_list, 3, 50, callback, 0);
    while (rwkvmobile_runtime_is_generating(runtime)) {
        std::cout << "Waiting for generation to finish..." << std::endl;
        sleep(1);
    }
    std::cout << "Response: " << response << std::endl;

    input_list[2] = "What's the weather like today?";
    rwkvmobile_runtime_eval_chat_with_history(runtime, (const char **)input_list, 3, 50, callback, 0);
    while (rwkvmobile_runtime_is_generating(runtime)) {
        std::cout << "Waiting for generation to finish..." << std::endl;
        sleep(1);
    }
    std::cout << "Response: " << response << std::endl;

    std::cout << std::endl;

    return 0;
}
