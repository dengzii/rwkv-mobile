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
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <encoder_file> <image_file>" << std::endl;
        return 1;
    }

    auto ctx_clip = clip_model_load(argv[1], 1);
    if (!ctx_clip) {
        std::cerr << "Failed to load encoder" << std::endl;
        return 1;
    }

    auto embd = llava_image_embed_make_with_filename(ctx_clip, 8, argv[2]);
    if (!embd) {
        std::cerr << "Failed to load image" << std::endl;
        return 1;
    }

    double sum[576] = {0.0};
    for (int i = 0; i < 576; i++) {
        for (int j = 0; j < 1024; j++) {
            sum[i] += embd->embed[i * 1024 + j];
        }
    }

    llava_image_embed_free(embd);

    // rwkvmobile::runtime runtime;
    // ENSURE_SUCCESS_OR_LOG_EXIT(runtime.init(argv[3]), "Failed to initialize runtime");
    // ENSURE_SUCCESS_OR_LOG_EXIT(runtime.load_tokenizer(argv[1]), "Failed to load tokenizer");
    // ENSURE_SUCCESS_OR_LOG_EXIT(runtime.load_model(argv[2]), "Failed to load model");
    // runtime.set_sampler_params(1.0, 1, 1.0);

    return 0;
}
