cmake_minimum_required(VERSION 3.25)

project(rwkv-mobile CXX)

set(CMAKE_CXX_STANDARD 20)

include(FetchContent)

option(ENABLE_NCNN_BACKEND "Enable NCNN backend" ON)
option(ENABLE_WEBRWKV_BACKEND "Enable WebRWKV backend" ON)
option(ENABLE_LLAMACPP_BACKEND "Enable llama.cpp backend" ON)
option(ENABLE_QNN_BACKEND "Enable QNN backend" OFF)

option(ENABLE_VISION "Enable vision" ON)
option(ENABLE_WHISPER "Enable whisper" ON)

option(BUILD_EXAMPLES "Build examples" ON)
option(BUILD_STATIC_LIB "Build static library" OFF)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

option(ADDRESS_SANITIZER "Enable AddressSanitizer" OFF)

if (ADDRESS_SANITIZER)
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
    # set(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
endif()

set(RWKV_MOBILE_SRCS
    src/runtime.cpp
    src/sampler.cpp
    src/tokenizer.cpp
    src/logger.cpp
)

if (ENABLE_WEBRWKV_BACKEND)
    set(RWKV_MOBILE_SRCS ${RWKV_MOBILE_SRCS} backends/web_rwkv/web_rwkv_backend.cpp)

    if (NOT IOS)
        FetchContent_Declare(
            Corrosion
            GIT_REPOSITORY https://github.com/corrosion-rs/corrosion.git
            GIT_TAG v0.5
        )
        FetchContent_MakeAvailable(Corrosion)

        FetchContent_Declare(
            web_rwkv_ffi
            GIT_REPOSITORY https://github.com/MollySophia/web-rwkv-ffi.git
            GIT_TAG main
        )
        FetchContent_MakeAvailable(web_rwkv_ffi)

        corrosion_import_crate(MANIFEST_PATH ${web_rwkv_ffi_SOURCE_DIR}/Cargo.toml
            CRATE_TYPES staticlib)
    endif()

endif()

if (ENABLE_NCNN_BACKEND)
    set(RWKV_MOBILE_SRCS ${RWKV_MOBILE_SRCS} backends/ncnn/ncnn_rwkv_backend.cpp)

    FetchContent_Declare(
        ncnn
        GIT_REPOSITORY https://github.com/Tencent/ncnn
        GIT_TAG 20240820
    )
    include(disable_unused_ncnn_layers)
    option(NCNN_BUILD_TOOLS "" OFF)
    option(NCNN_BUILD_EXAMPLES "" OFF)
    option(NCNN_BUILD_TESTS "" OFF)
    option(NCNN_BUILD_BENCHMARK "" OFF)
    option(NCNN_DISABLE_EXCEPTION "" OFF)
    FetchContent_MakeAvailable(ncnn)
endif()

if (ENABLE_LLAMACPP_BACKEND)
    set(RWKV_MOBILE_SRCS ${RWKV_MOBILE_SRCS} backends/llama_cpp/llama_cpp_backend.cpp)
    if (ENABLE_VISION)
        set(RWKV_MOBILE_SRCS ${RWKV_MOBILE_SRCS}
            src/multimodal/vision/clip.cpp
            src/multimodal/vision/llava.cpp
        )
    endif()

    if (ENABLE_WHISPER)
        set(RWKV_MOBILE_SRCS ${RWKV_MOBILE_SRCS}
            src/multimodal/whisper/whisper.cpp
        )
    endif()

    FetchContent_Declare(
        llama_cpp
        GIT_REPOSITORY https://github.com/mollysophia/llama.cpp
        GIT_TAG rwkv-v7
    )

    option(LLAMA_STANDALONE "" OFF)
    set(BUILD_SHARED_LIBS_DEFAULT OFF)
    FetchContent_MakeAvailable(llama_cpp)
endif()

if (ENABLE_QNN_BACKEND)
    file(GLOB QNN_SRCS backends/qnn/rwkv-qualcomm/Utils/*.cpp backends/qnn/rwkv-qualcomm/WrapperUtils/*.cpp backends/qnn/rwkv-qualcomm/PAL/src/common/*.cpp
                        backends/qnn/rwkv-qualcomm/PAL/src/linux/*.cpp)
    set(RWKV_MOBILE_SRCS ${RWKV_MOBILE_SRCS} ${QNN_SRCS})
    set(RWKV_MOBILE_SRCS ${RWKV_MOBILE_SRCS} backends/qnn/qnn_backend.cpp)
endif()

add_library(rwkv_mobile_internal STATIC ${RWKV_MOBILE_SRCS})
target_include_directories(rwkv_mobile_internal PUBLIC src)

if (ENABLE_WEBRWKV_BACKEND)
    if (APPLE)
        set(WEBRWKV_EXTRA_LIBS "-framework QuartzCore -framework Metal -framework CoreGraphics -framework CoreFoundation -lobjc -liconv")
    elseif (WIN32)
        set(WEBRWKV_EXTRA_LIBS ws2_32 opengl32 d3d12 d3dcompiler userenv kernel32 user32 ntdll bcrypt)
    endif()
    target_compile_definitions(rwkv_mobile_internal PUBLIC ENABLE_WEBRWKV)
    target_include_directories(rwkv_mobile_internal PUBLIC backends/web_rwkv ${web_rwkv_ffi_SOURCE_DIR}/include)
    target_link_libraries(rwkv_mobile_internal PUBLIC web_rwkv_ffi ${WEBRWKV_EXTRA_LIBS})
endif()

if (ENABLE_NCNN_BACKEND)
    target_compile_definitions(rwkv_mobile_internal PUBLIC ENABLE_NCNN)
    target_include_directories(rwkv_mobile_internal PUBLIC backends/ncnn)
    target_link_libraries(rwkv_mobile_internal PUBLIC "$<LINK_LIBRARY:WHOLE_ARCHIVE,ncnn>")
endif()

if (ENABLE_LLAMACPP_BACKEND)
    if (APPLE AND GGML_ACCELERATE)
        find_library(ACCELERATE_FRAMEWORK Accelerate)
        if (ACCELERATE_FRAMEWORK)
            set(LLAMACPP_EXTRA_LIBS ${LLAMACPP_EXTRA_LIBS} ${ACCELERATE_FRAMEWORK})
        endif()
    endif()

    if (GGML_METAL)
        find_library(FOUNDATION_LIBRARY Foundation REQUIRED)
        find_library(METAL_FRAMEWORK    Metal      REQUIRED)
        find_library(METALKIT_FRAMEWORK MetalKit   REQUIRED)
        add_compile_definitions(GGML_USE_METAL)

        set(LLAMACPP_EXTRA_LIBS ${LLAMACPP_EXTRA_LIBS}
            ${FOUNDATION_LIBRARY}
            ${METAL_FRAMEWORK}
            ${METALKIT_FRAMEWORK}
        )

        set(LLAMACPP_EXTRA_LIBS ${LLAMACPP_EXTRA_LIBS} $<TARGET_OBJECTS:ggml-metal> $<TARGET_OBJECTS:ggml-blas>)
    endif()

    if (ENABLE_VISION)
        target_include_directories(rwkv_mobile_internal PUBLIC src/multimodal/vision)
        target_compile_definitions(rwkv_mobile_internal PUBLIC ENABLE_VISION)
    endif()

    if (ENABLE_WHISPER)
        target_include_directories(rwkv_mobile_internal PUBLIC src/multimodal/whisper)
        target_compile_definitions(rwkv_mobile_internal PUBLIC ENABLE_WHISPER)
    endif()

    target_compile_definitions(rwkv_mobile_internal PUBLIC ENABLE_LLAMACPP)
    target_include_directories(rwkv_mobile_internal PUBLIC backends/llama_cpp ${llama_cpp_SOURCE_DIR}/include ${llama_cpp_SOURCE_DIR}/src ${llama_cpp_SOURCE_DIR}/ggml/include)
    target_link_libraries(rwkv_mobile_internal PUBLIC $<TARGET_OBJECTS:llama> $<TARGET_OBJECTS:ggml> $<TARGET_OBJECTS:ggml-base> $<TARGET_OBJECTS:ggml-cpu> ${LLAMACPP_EXTRA_LIBS})
endif()

if (ENABLE_QNN_BACKEND)
    target_compile_definitions(rwkv_mobile_internal PUBLIC ENABLE_QNN)
    target_include_directories(rwkv_mobile_internal PUBLIC backends/qnn
                               backends/qnn/rwkv-qualcomm
                               backends/qnn/rwkv-qualcomm/Utils
                               backends/qnn/rwkv-qualcomm/PAL/include
                               backends/qnn/rwkv-qualcomm/WrapperUtils
                               backends/qnn/rwkv-qualcomm/QNN/)
endif()

if (BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if (NOT WIN32)
    set(CMAKE_LINK_LIBRARY_USING_WHOLE_ARCHIVE "LINKER:--whole-archive" "<LINK_ITEM>" "LINKER:--no-whole-archive")
    set(CMAKE_LINK_LIBRARY_USING_WHOLE_ARCHIVE_SUPPORTED TRUE)
endif()

if (DEFINED ANDROID_NDK)
    find_library(log-lib log)
    find_library(android-lib android)
    target_link_libraries(rwkv_mobile_internal PUBLIC ${log-lib} ${android-lib})
endif()

if (BUILD_STATIC_LIB)
    add_library(rwkv_mobile STATIC src/c_api.cpp ${RWKV_MOBILE_SRCS})
    target_include_directories(rwkv_mobile PUBLIC src)
    if (ENABLE_WEBRWKV_BACKEND)
        target_compile_definitions(rwkv_mobile PUBLIC ENABLE_WEBRWKV)
        target_include_directories(rwkv_mobile PUBLIC backends/web_rwkv ${web_rwkv_ffi_SOURCE_DIR}/include)
        target_link_libraries(rwkv_mobile PRIVATE web_rwkv_ffi ${WEBRWKV_EXTRA_LIBS})
    endif()

    if (ENABLE_NCNN_BACKEND)
        target_compile_definitions(rwkv_mobile PUBLIC ENABLE_NCNN)
        target_include_directories(rwkv_mobile PUBLIC backends/ncnn)
    endif()
else()
    add_library(rwkv_mobile SHARED src/c_api.cpp)
    target_link_libraries(rwkv_mobile PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,rwkv_mobile_internal>")
    if (ENABLE_WEBRWKV_BACKEND)
        target_link_libraries(rwkv_mobile PRIVATE web_rwkv_ffi ${WEBRWKV_EXTRA_LIBS})
    endif()

    # if (ENABLE_NCNN_BACKEND)
    #     target_link_libraries(rwkv_mobile PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,ncnn>")
    # endif()
endif()