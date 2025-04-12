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
    runtime.load_model("test.gguf");

    rwkvmobile::frontend frontend;
    frontend.load_speech_tokenizer("speech_tokenizer_v2.onnx");
    frontend.load_campplus("campplus.onnx");
    frontend.load_flow_encoder("flow_encoder.fp16.onnx");
    frontend.load_flow_decoder_estimator("flow.decoder.estimator.fp32.onnx");

    std::string tts_text = "有时候，看着小孩子们的天真行为，我们总会会心一笑。";
    tts_text = frontend.normalize_text(tts_text);
    std::string instruction_text = "请用正常的语速说。";
    std::vector<int> tts_tokens = runtime.tokenizer_encode(tts_text);
    std::vector<int> prompt_tokens = runtime.tokenizer_encode(instruction_text + "<|endofprompt|>");

    int min_len, max_len;
    std::vector<int> llm_tokens = frontend.get_llm_tokens(tts_tokens, prompt_tokens, min_len, max_len);
    std::vector<int> speech_tokens;
    std::vector<std::vector<float>> speech_features;
    std::vector<float> speech_embedding;
    frontend.process_zeroshot("jfk.wav", speech_tokens, speech_features, speech_embedding, 24000);

    // float *logits = nullptr;
    // runtime.eval_logits(llm_tokens, logits);
    // const int speech_vocab_size = 6562;
    // const int speech_vocab_offset = 65548;

    // std::vector<int> decoded_tokens;
    // for (int i = 0; i < max_len; i++) {
    //     int token_id = frontend.speech_token_sampler(logits, speech_vocab_size, decoded_tokens, (i < min_len));
    //     runtime.free_logits_if_allocated(logits);
    //     if (token_id == speech_vocab_size - 1) {
    //         break;
    //     }
    //     decoded_tokens.push_back(token_id);
    //     runtime.eval_logits(token_id + speech_vocab_offset, logits);
    // }
    std::cout << "speech_tokens: ";
    for (int i = 0; i < speech_tokens.size(); i++) {
        std::cout << speech_tokens[i] << ", ";
    }
    std::cout << std::endl;
    std::vector<int> decoded_tokens = {1490, 4299, 4299, 4299, 4299, 4299, 4299, 4137, 2724, 2591, 380, 1451, 5968, 5322, 888, 227, 4860, 4863, 5053, 4973, 5000, 5090, 1299, 3888, 5835, 3651, 2112, 2139, 165, 5841, 5998, 834, 3147, 4526, 4672, 4677, 8, 1946, 3402, 6075, 1466, 2249, 4191, 5652, 2395, 4314, 101, 1479, 56, 867, 2220, 2760, 4661, 4468, 3669, 5322, 4523, 4479, 4398, 5402, 4678, 538, 1133, 6419, 5802, 5095, 32, 35, 386, 4850, 2554, 1707, 5838, 3648, 3645, 1944, 4299, 1393, 2915, 2124, 2612, 4554, 6378, 2493, 623, 1322, 2132, 2841, 2847, 4600, 4606, 2682, 2838, 2342, 647, 5404, 5405, 295, 4994, 5696, 1268, 1295, 5173, 5726, 4678, 4517, 5834, 3413, 1226, 2029, 4299, 4299, 4299, 4299, 4299, 4299, 4299};
    decoded_tokens.insert(decoded_tokens.begin(), speech_tokens.begin(), speech_tokens.end());
    std::cout << "decoded_tokens: ";
    for (int i = 0; i < decoded_tokens.size(); i++) {
        std::cout << decoded_tokens[i] << ", ";
    }
    std::cout << std::endl;
    frontend.speech_token_to_wav(decoded_tokens, speech_features, speech_embedding);

    runtime.release();

    return 0;
}
