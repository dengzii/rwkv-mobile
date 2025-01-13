#include <iostream>
#include <chrono>

#include "commondef.h"
#include "runtime.h"

#define ENSURE_SUCCESS_OR_LOG_EXIT(x, msg) if (x != rwkvmobile::RWKV_SUCCESS) { std::cout << msg << std::endl; return 1; }

std::string current = "";
void callback(const char *msg) {
    std::cout << msg + current.size();
    current = msg;
    // std::cout << msg << "\n\n";
}

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
    runtime.set_sampler_params(1.0, 1, 1.0);
    runtime.set_penalty_params(0, 0, 0);

    std::string prompt = "<input>\n"
                        "● ● ● ● ● ● ● ● \n"
                        "● · ○ ○ ● ● ● ○ \n"
                        "● ○ ○ ○ ○ ● ● ○ \n"
                        "● ○ ○ ○ ○ ● ● ● \n"
                        "● ○ ○ ○ ○ ● ● ● \n"
                        "● ○ ○ ○ ○ ● ● ● \n"
                        "● · · · · ● ● ● \n"
                        "● · · ○ ○ ○ ○ ○ \n"
                        "NEXT ● \n"
                        "MAX_WIDTH-2\n"
                        "MAX_DEPTH-2\n"
                        "</input>\n\n";

    auto ids = runtime.tokenizer_encode(prompt);
    for (auto id : ids) {
        std::cout << id << ", ";
    }
    std::cout << std::endl;
    std::string result;

    runtime.gen_completion(prompt, result, 64000, 0, callback);
    return 0;
}
