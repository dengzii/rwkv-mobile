#include <iostream>
#include <chrono>

#include "commondef.h"
#include "runtime.h"

#define ENSURE_SUCCESS_OR_LOG_EXIT(x, msg) if (x != rwkvmobile::RWKV_SUCCESS) { std::cout << msg << std::endl; return 1; }

int main(int argc, char **argv) {
    // set stdout to be unbuffered
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <vocab_file> <model_file> <backend>" << std::endl;
        return 1;
    }

    rwkvmobile::runtime runtime;
    ENSURE_SUCCESS_OR_LOG_EXIT(runtime.init(argv[3]), "Failed to initialize runtime");
    ENSURE_SUCCESS_OR_LOG_EXIT(runtime.load_tokenizer(argv[1]), "Failed to load tokenizer");
    ENSURE_SUCCESS_OR_LOG_EXIT(runtime.load_model(argv[2]), "Failed to load model");

    runtime.set_prompt("User: Hello!\n\nAssistant: Hi!\n\n");

    std::vector<std::string> input_list = {
        "Hello!"
    };
    runtime.chat(input_list, 50, nullptr);
    std::cout << "Response: " << runtime.get_response_buffer_content() << std::endl;

    runtime.clear_state();

    input_list = {
        "Hi!"
    };
    runtime.chat(input_list, 50, nullptr);
    std::cout << "Response: " << runtime.get_response_buffer_content() << std::endl;

    std::cout << std::endl;

    return 0;
}
