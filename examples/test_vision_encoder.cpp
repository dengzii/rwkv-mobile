#include <iostream>
#include <chrono>

#include "commondef.h"
#include "runtime.h"
#include "clip.h"
#include "llava.h"

#define ENSURE_SUCCESS_OR_LOG_EXIT(x, msg) if (x != rwkvmobile::RWKV_SUCCESS) { std::cout << msg << std::endl; return 1; }

int main(int argc, char **argv) {
    // set stdout to be unbuffered
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0] << " <model_file> <encoder_file> <tokenizer_file> <image_file> <backend>" << std::endl;
        return 1;
    }

    rwkvmobile::runtime runtime;
    ENSURE_SUCCESS_OR_LOG_EXIT(runtime.init(argv[5]), "Failed to initialize runtime");
    ENSURE_SUCCESS_OR_LOG_EXIT(runtime.load_tokenizer(argv[3]), "Failed to load tokenizer");
    ENSURE_SUCCESS_OR_LOG_EXIT(runtime.load_model(argv[1]), "Failed to load model");
    ENSURE_SUCCESS_OR_LOG_EXIT(runtime.load_vision_encoder(argv[2]), "Failed to load vision encoder");
    runtime.set_sampler_params(1.0, 1, 1.0);

    runtime.set_image_prompt(argv[4]);

    std::string response;
    runtime.chat("Describe this image", response, 100);
    std::cout << response << std::endl;

    return 0;
}
