if (WIN32)
    add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
    add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
endif()

add_executable(gen gen.cpp)
target_link_libraries(gen PUBLIC rwkv_mobile_internal)

add_executable(test_chat test_chat.cpp)
target_link_libraries(test_chat PUBLIC rwkv_mobile_internal)

add_executable(test_chat_resume test_chat_resume.cpp ../src/c_api.cpp)
target_link_libraries(test_chat_resume PUBLIC rwkv_mobile_internal)

add_executable(test_chat_c_api test_chat_c_api.cpp ../src/c_api.cpp)
target_link_libraries(test_chat_c_api PUBLIC rwkv_mobile_internal)

add_executable(test_othello test_othello.cpp ../src/c_api.cpp)
target_link_libraries(test_othello PUBLIC rwkv_mobile_internal)

add_executable(test_model_reloading test_model_reloading.cpp ../src/c_api.cpp)
target_link_libraries(test_model_reloading PUBLIC rwkv_mobile_internal)

add_executable(test_clear_states test_clear_states.cpp)
target_link_libraries(test_clear_states PUBLIC rwkv_mobile_internal)

add_executable(test_sudoku test_sudoku.cpp ../src/c_api.cpp)
target_link_libraries(test_sudoku PUBLIC rwkv_mobile_internal)

add_executable(test_embedding test_embedding.cpp ../src/c_api.cpp)
target_link_libraries(test_embedding PUBLIC rwkv_mobile_internal)

if (ENABLE_VISION)
    add_executable(test_vision_encoder test_vision_encoder.cpp ../src/c_api.cpp)
    target_link_libraries(test_vision_encoder PUBLIC rwkv_mobile_internal)

    add_executable(test_vision_encoder_with_adapter test_vision_encoder_with_adapter.cpp ../src/c_api.cpp)
    target_link_libraries(test_vision_encoder_with_adapter PUBLIC rwkv_mobile_internal)
endif()

if (ENABLE_WHISPER)
    add_executable(test_whisper_encoder test_whisper_encoder.cpp ../src/c_api.cpp)
    target_link_libraries(test_whisper_encoder PUBLIC rwkv_mobile_internal)
endif()

add_executable(test_soc_detect test_soc_detect.cpp ../src/c_api.cpp)
target_link_libraries(test_soc_detect PUBLIC rwkv_mobile_internal)

add_executable(simple_benchmark simple_benchmark.cpp)
target_link_libraries(simple_benchmark PUBLIC rwkv_mobile_internal)

if (ENABLE_TTS)
    add_executable(test_tts test_tts.cpp)
    target_link_libraries(test_tts PUBLIC rwkv_mobile_internal)
endif()