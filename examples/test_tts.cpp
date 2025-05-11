#include <iostream>
#include <unistd.h>
#include <fstream>
#include <chrono>

#include "commondef.h"
#include "runtime.h"
#include "c_api.h"
#include "cosyvoice.h"
#include "logger.h"

#define ENSURE_SUCCESS_OR_LOG_EXIT(x, msg) if (x != rwkvmobile::RWKV_SUCCESS) { std::cout << msg << std::endl; return 1; }

int main(int argc, char **argv) {
    // set stdout to be unbuffered
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc != 6 && argc != 7) {
        std::cerr << "Usage: " << argv[0] << " <model_file> <backend> <encoder_path> <tokenizer_file> <wav_file> [prompt_speech_text]" << std::endl;
        return 1;
    }
    rwkvmobile::runtime runtime;
    runtime.init(argv[2]);
    runtime.load_model(argv[1]);

    std::string encoder_path = argv[3];

    runtime.cosyvoice_load_models(
        encoder_path + "speech_tokenizer_v2.onnx",
        encoder_path + "campplus.onnx",
        encoder_path + "flow_encoder.fp16.onnx",
        encoder_path + "flow_decoder_estimator.ncnn.bin",
        encoder_path + "hift.onnx",
        encoder_path + std::string(argv[4])
    );

    runtime.tts_register_text_normalizer("date-zh.fst");
    runtime.tts_register_text_normalizer("number-zh.fst");
    runtime.tts_register_text_normalizer("phone-zh.fst");

    std::string tts_text = "在中国驻美国大使馆3日举办的使馆开放日暨甘肃省推介活动上，中国驻美大使谢锋表示，中国不愿打关税战，但不怕打。如果美方准备要谈，就应当拿出平等、尊重、互惠的态度。\n\n"
"  谢锋说，美国从国际贸易中受益颇丰，既享受了来自全球物美价廉的商品，又在金融、科技、服务等高附加值领域占据明显优势。2022年美资企业在华销售额比中资企业在美销售额多出4000多亿美元。中美经贸合作总体是平衡的、双赢的。滥施关税损人害己，干扰企业正常生产经营和民众生活消费，引发全球金融市场剧烈波动，破坏世界经济长期稳定增长。\n\n"
"  谢锋强调，无论国际风云如何变幻，中国将继续以高质量发展、高水平开放的确定性应对外部不确定性。中国是全球第二大消费市场，拥有最大规模中等收入群体，已成为150多个国家和地区的主要贸易伙伴。今年一季度中国经济迎来开门红，国内生产总值（GDP）同比增长5.4%，出口顶压逆势增长6.9%，国际市场朋友圈越来越大。";

//     // std::string instruction_text = "请用高昂的语气说";
    std::string instruction_text = ""; // empty string means no instruction

    runtime.set_cache_dir("cache_dir");

    std::string prompt_speech_text = "";
    if (argc == 7) {
        prompt_speech_text = argv[6];
    }

    runtime.run_tts(tts_text, instruction_text, prompt_speech_text, argv[5], "test.wav");

    runtime.release();

    return 0;
}
