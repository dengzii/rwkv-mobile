add_executable(gen gen.cpp)
target_link_libraries(gen PUBLIC rwkv_mobile_internal)

add_executable(test_chat test_chat.cpp)
target_link_libraries(test_chat PUBLIC rwkv_mobile_internal)

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

if (ENABLE_VISION)
    add_executable(test_vision_encoder test_vision_encoder.cpp ../src/c_api.cpp)
    target_link_libraries(test_vision_encoder PUBLIC rwkv_mobile_internal)
endif()

if (ENABLE_WHISPER)
    add_executable(test_whisper_encoder test_whisper_encoder.cpp ../src/c_api.cpp)
    target_link_libraries(test_whisper_encoder PUBLIC rwkv_mobile_internal)
endif()
