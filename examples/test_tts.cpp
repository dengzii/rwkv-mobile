#include <iostream>
#include <unistd.h>
#include <fstream>

#include "commondef.h"
#include "runtime.h"
#include "c_api.h"

#include "frontend.h"

#define ENSURE_SUCCESS_OR_LOG_EXIT(x, msg) if (x != rwkvmobile::RWKV_SUCCESS) { std::cout << msg << std::endl; return 1; }

int main(int argc, char **argv) {
    // set stdout to be unbuffered
    setvbuf(stdout, NULL, _IONBF, 0);
    // if (argc != 6) {
    //     std::cerr << "Usage: " << argv[0] << " <model_file> <encoder_file> <tokenizer_file> <wav_file> <backend>" << std::endl;
    //     return 1;
    // }
    rwkvmobile::runtime runtime;
    runtime.init("llama.cpp");
    runtime.load_tokenizer("assets/b_rwkv_vocab_v20230424_tts.txt");

    rwkvmobile::frontend frontend;
    frontend.load_speech_tokenizer("speech_tokenizer_v2.onnx");
    frontend.load_campplus("campplus.onnx");

    std::string tts_text = "[laughter]有时候，看着小孩子们的天真行为[laughter]，我们总会会心一笑。";
    tts_text = frontend.normalize_text(tts_text);
    std::string instruction_text = "请用正常的语速说。";
    std::vector<int> tts_tokens = runtime.tokenizer_encode(tts_text);
    std::vector<int> prompt_tokens = runtime.tokenizer_encode(instruction_text + "<|endofprompt|>");

    frontend.process_zeroshot(tts_tokens, prompt_tokens, "jfk.wav", 24000);

    return 0;
}
