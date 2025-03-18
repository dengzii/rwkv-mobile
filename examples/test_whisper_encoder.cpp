#include <iostream>
#include <unistd.h>
#include <fstream>

#include "commondef.h"
#include "runtime.h"
#include "c_api.h"
#include "whisper.h"
#include "half.hpp"

#define ENSURE_SUCCESS_OR_LOG_EXIT(x, msg) if (x != rwkvmobile::RWKV_SUCCESS) { std::cout << msg << std::endl; return 1; }

void callback(const char *msg, const int) {
    std::cout << msg;
}

int main(int argc, char **argv) {
    // set stdout to be unbuffered
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0] << " <model_file> <encoder_file> <tokenizer_file> <wav_file> <backend>" << std::endl;
        return 1;
    }

    rwkvmobile_runtime_t runtime = rwkvmobile_runtime_init_with_name(argv[5]);
    rwkvmobile_runtime_load_tokenizer(runtime, argv[3]);
    rwkvmobile_runtime_load_model(runtime, argv[1]);
    rwkvmobile_runtime_load_whisper_encoder(runtime, argv[2]);
    rwkvmobile_runtime_set_eos_token(runtime, "\x17");
    rwkvmobile_runtime_set_bos_token(runtime, "\x16");
    rwkvmobile_runtime_set_token_banned(runtime, {0}, 1);
    rwkvmobile_runtime_set_user_role(runtime, "");

    rwkvmobile_runtime_set_audio_prompt(runtime, argv[4]);

    rwkvmobile_runtime_eval_chat(runtime, "", 100, callback, 0);

    while (rwkvmobile_runtime_is_generating(runtime)) {
        sleep(1);
    }

    rwkvmobile_runtime_release_whisper_encoder(runtime);

    rwkvmobile_runtime_release(runtime);
    return 0;
}
