#include <iostream>
#include <unistd.h>
#include <fstream>

#include "commondef.h"
#include "runtime.h"
#include "c_api.h"

#include "cosyvoice.h"

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

    rwkvmobile::cosyvoice cosyvoice;
    cosyvoice.load_speech_tokenizer("speech_tokenizer_v2.onnx");
    cosyvoice.load_campplus("campplus.onnx");
    cosyvoice.load_flow_encoder("flow_encoder.fp16.onnx");
    cosyvoice.load_flow_decoder_estimator("flow.decoder.estimator.fp32.onnx");
    cosyvoice.load_hift_generator("hift.onnx");

    std::string tts_text = "Make America great again!";
    tts_text = cosyvoice.normalize_text(tts_text);
    std::string instruction_text = "请用正常的语速说。";
    std::vector<int> tts_tokens = runtime.tokenizer_encode(tts_text);
    std::vector<int> prompt_tokens = runtime.tokenizer_encode(instruction_text + "<|endofprompt|>");
    std::cout << "tts_tokens: ";
    for (int i = 0; i < tts_tokens.size(); i++) {
        std::cout << tts_tokens[i] << ", ";
    }
    std::cout << std::endl << std::endl;
    std::cout << "prompt_tokens: ";
    for (int i = 0; i < prompt_tokens.size(); i++) {
        std::cout << prompt_tokens[i] << ", ";
    }
    std::cout << std::endl << std::endl;

    int min_len, max_len;
    std::vector<int> llm_tokens = cosyvoice.get_llm_tokens(tts_tokens, prompt_tokens, min_len, max_len);
    std::vector<int> speech_tokens;
    std::vector<std::vector<float>> speech_features;
    std::vector<float> speech_embedding;
    cosyvoice.process_zeroshot("Trump.wav", speech_tokens, speech_features, speech_embedding, 24000);

    float *logits = nullptr;
    runtime.eval_logits(llm_tokens, logits);
    const int speech_vocab_size = 6562;
    const int speech_vocab_offset = 65548;

    std::vector<int> decoded_tokens;
    for (int i = 0; i < max_len; i++) {
        int token_id = cosyvoice.speech_token_sampler(logits, speech_vocab_size, decoded_tokens, (i < min_len));
        runtime.free_logits_if_allocated(logits);
        if (token_id == speech_vocab_size - 1) {
            break;
        }
        decoded_tokens.push_back(token_id);
        runtime.eval_logits(token_id + speech_vocab_offset, logits);
    }
    std::cout << "speech_tokens: ";
    for (int i = 0; i < speech_tokens.size(); i++) {
        std::cout << speech_tokens[i] << ", ";
    }
    std::cout << std::endl << std::endl;
    std::cout << "decoded_tokens: ";
    for (int i = 0; i < decoded_tokens.size(); i++) {
        std::cout << decoded_tokens[i] << ", ";
    }
    std::cout << std::endl;
    decoded_tokens.insert(decoded_tokens.begin(), speech_tokens.begin(), speech_tokens.end());
    cosyvoice.speech_token_to_wav(decoded_tokens, speech_features, speech_embedding, "test.wav");

    runtime.release();

    return 0;
}
