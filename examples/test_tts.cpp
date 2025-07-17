#include <iostream>
#include <fstream>
#include <chrono>

#include "commondef.h"
#include "runtime.h"
#include "c_api.h"
#include "logger.h"

#define ENSURE_SUCCESS_OR_LOG_EXIT(x, msg) if (x != rwkvmobile::RWKV_SUCCESS) { std::cout << msg << std::endl; return 1; }

int main(int argc, char **argv) {
    // set stdout to be unbuffered
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc != 5 && argc != 6) {
        std::cerr << "Usage: " << argv[0] << " <model_file> <backend> <tokenizer_file> <wav_file>" << std::endl;
        return 1;
    }
    rwkvmobile::runtime runtime;
    runtime.init(argv[2]);
    runtime.load_model(argv[1]);
    runtime.load_tokenizer(argv[3]);

    runtime.sparktts_load_models(
        "/home/molly/rwkv-mobile/wav2vec2-large-xlsr-53.mnn",
        "/home/molly/rwkv-mobile/BiCodecTokenize.mnn",
        "/home/molly/rwkv-mobile/BiCodecDetokenize.mnn"
    );

    // rwkvmobile::wav_file wav_file;
    // wav_file.load("/home/molly/rwkv-mobile/kafka.wav");
    // wav_file.resample(16000);

    // std::vector<int> global_tokens;
    // std::vector<int> semantic_tokens;
    // sparktts.tokenize_audio(wav_file.samples, global_tokens, semantic_tokens);

    auto start = std::chrono::high_resolution_clock::now();
    runtime.run_spark_tts("为了点燃青少年对科技的热情，培养他们的创新思维与动手能力，杏花岭区巨轮街道社区教育学校携手中车社区教育分校，与太原市科学技术协会联手，于暑期精心策划了一场别开生面的青少年数智技术服务港探索之旅，吸引了众多社区青少年的积极参与。", "", argv[4], "/home/molly/rwkv-mobile/output.wav");
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Total time taken: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    return 0;
}
