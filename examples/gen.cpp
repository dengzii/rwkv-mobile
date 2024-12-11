#include <iostream>

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

    std::cout << "Generating demo text..." << std::endl;
    std::string result;

    std::string prompt = "The Eiffel tower is in the city of";
    std::cout << prompt;
    ENSURE_SUCCESS_OR_LOG_EXIT(runtime.gen_completion(prompt, result, 1), "Failed to generate chat message");
    std::cout << result;
    for (int i = 0; i < 200; i++ ) {
        std::string input(result);
        ENSURE_SUCCESS_OR_LOG_EXIT(runtime.gen_completion(input, result, 1), "Failed to generate chat message");
        std::cout << result;
    }

    std::cout << std::endl;

    return 0;
}
