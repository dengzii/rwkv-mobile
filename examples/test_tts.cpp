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
    rwkvmobile::frontend frontend;
    frontend.load_speech_tokenizer("speech_tokenizer_v2.onnx");
    frontend.load_campplus("campplus.onnx");
    frontend.process_zeroshot("Hello, world!", "Hello, world!", "jfk.wav", 24000);


    return 0;
}
