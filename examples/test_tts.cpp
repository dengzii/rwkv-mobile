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
        "wav2vec2-large-xlsr-53.mnn",
        "BiCodecTokenize.mnn",
        "BiCodecDetokenize.mnn"
    );

    // rwkvmobile::wav_file wav_file;
    // wav_file.load("/home/molly/rwkv-mobile/kafka.wav");
    // wav_file.resample(16000);

    // std::vector<int> global_tokens;
    // std::vector<int> semantic_tokens;
    // sparktts.tokenize_audio(wav_file.samples, global_tokens, semantic_tokens);

    runtime.run_spark_tts_streaming("他们小心翼翼地调整电路，确保每个部件都正确连接，红灯、绿灯、黄灯依次亮起，仿佛在讲述一个关于交通规则的故事。", "", argv[4], "output.wav");

    runtime.run_spark_tts_streaming("他们小心翼翼地调整电路，确保每个部件都正确连接，红灯、绿灯、黄灯依次亮起，仿佛在讲述一个关于交通规则的故事。", "", argv[4], "output.wav");

    runtime.release();

    return 0;
}
