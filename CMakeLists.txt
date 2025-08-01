cmake_minimum_required(VERSION 3.25)

project(rwkv-mobile C CXX)

if(APPLE)
  include(CheckOBJCSourceCompiles)
  enable_language(OBJC)
endif()

set(CMAKE_CXX_STANDARD 17)

include(FetchContent)

option(ENABLE_NCNN_BACKEND "Enable NCNN backend" ON)
option(ENABLE_WEBRWKV_BACKEND "Enable WebRWKV backend" ON)
option(ENABLE_LLAMACPP_BACKEND "Enable llama.cpp backend" ON)
option(ENABLE_QNN_BACKEND "Enable QNN backend" OFF)
option(ENABLE_MNN_BACKEND "Enable MNN backend" ON)
option(ENABLE_COREML_BACKEND "Enable CoreML backend" OFF)

option(ENABLE_VISION "Enable vision" OFF)
option(ENABLE_WHISPER "Enable whisper" OFF)
option(ENABLE_TTS "Enable TTS" OFF)

option(BUILD_EXAMPLES "Build examples" ON)
option(BUILD_STATIC_LIB "Build static library" OFF)

if (ENABLE_TTS AND NOT ENABLE_MNN_BACKEND)
    message(FATAL_ERROR "TTS requires MNN backend")
endif()

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

option(ADDRESS_SANITIZER "Enable AddressSanitizer" OFF)

if (ANDROID)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g0")
endif()

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
    src/soc_detect.cpp
)

if (WIN32)
    set(CMAKE_LINK_LIBRARY_USING_WHOLE_ARCHIVE "/WHOLEARCHIVE:<LIBRARY>")
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    cmake_host_system_information(RESULT CPU_NUMBER_OF_LOGICAL_CORES QUERY NUMBER_OF_LOGICAL_CORES)
    add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/MP${CPU_NUMBER_OF_LOGICAL_CORES}>")
else()
    set(CMAKE_LINK_LIBRARY_USING_WHOLE_ARCHIVE "LINKER:--whole-archive" "<LINK_ITEM>" "LINKER:--no-whole-archive")
endif()
set(CMAKE_LINK_LIBRARY_USING_WHOLE_ARCHIVE_SUPPORTED TRUE)

set(BUILD_SHARED_LIBS_DEFAULT OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (ENABLE_WEBRWKV_BACKEND)
    set(RWKV_MOBILE_SRCS ${RWKV_MOBILE_SRCS} src/backends/web_rwkv/web_rwkv_backend.cpp)

    if (IOS)
        set(Rust_CARGO_TARGET "aarch64-apple-ios")
    endif()

    FetchContent_Declare(
        Corrosion
        GIT_REPOSITORY https://github.com/corrosion-rs/corrosion.git
        GIT_TAG v0.5
    )
    FetchContent_MakeAvailable(Corrosion)

    FetchContent_Declare(
        web_rwkv_ffi
        GIT_REPOSITORY https://github.com/MollySophia/web-rwkv-ffi
        GIT_TAG test
    )
    FetchContent_MakeAvailable(web_rwkv_ffi)

    corrosion_import_crate(MANIFEST_PATH ${web_rwkv_ffi_SOURCE_DIR}/Cargo.toml
        CRATE_TYPES staticlib)

endif()

if (ENABLE_NCNN_BACKEND)
    set(RWKV_MOBILE_SRCS ${RWKV_MOBILE_SRCS} src/backends/ncnn/ncnn_rwkv_backend.cpp)

    FetchContent_Declare(
        ncnn
        GIT_REPOSITORY https://github.com/Tencent/ncnn
        GIT_TAG master
    )
    # include(disable_unused_ncnn_layers)
    option(NCNN_BUILD_TOOLS "" OFF)
    option(NCNN_BUILD_EXAMPLES "" OFF)
    option(NCNN_BUILD_TESTS "" OFF)
    option(NCNN_BUILD_BENCHMARK "" OFF)
    option(NCNN_DISABLE_EXCEPTION "" OFF)
    FetchContent_MakeAvailable(ncnn)
endif()

if (ENABLE_MNN_BACKEND)
    set(RWKV_MOBILE_SRCS ${RWKV_MOBILE_SRCS} src/backends/mnn/mnn_rwkv_backend.cpp)

    FetchContent_Declare(
        mnn
        GIT_REPOSITORY https://github.com/alibaba/MNN
        GIT_TAG a739ea5870a4a45680f0e36ba9662ca39f2f4eec
    )

    if (ANDROID)
        option(MNN_ARM82 "" ON)
    else()
        # TODO: fix iOS MNN_ARM82
        option(MNN_ARM82 "" OFF)
    endif()

    if (ANDROID)
        option(MNN_SUPPORT_BF16 "" ON)
        option(MNN_OPENCL "" ON)
        option(MNN_VULKAN "" ON)
    endif()

    if (APPLE)
        option(MNN_METAL "" ON)
        option(MNN_COREML "" ON)
    endif()

    option(MNN_LOW_MEMORY "" ON)
    option(MNN_SUPPORT_TRANSFORMER_FUSE "" ON)
    option(MNN_BUILD_TOOLS "" OFF)
    option(MNN_BUILD_SHARED_LIBS "" OFF)
    option(MNN_SEP_BUILD "" OFF)
    option(MNN_USE_SYSTEM_LIB "" ON)
    option(MNN_BUILD_TOOLS "" OFF)
    FetchContent_MakeAvailable(mnn)
endif()

if (ENABLE_LLAMACPP_BACKEND)
    set(RWKV_MOBILE_SRCS ${RWKV_MOBILE_SRCS}
        src/backends/llama_cpp/llama_cpp_backend.cpp
            src/embedding/rwkv_embedding.cpp
    )
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
        GIT_REPOSITORY https://github.com/ggml-org/llama.cpp
        GIT_TAG c8ade30036139e32108fee53d8b7164dbfda4bee
    )

    option(LLAMA_STANDALONE "" OFF)
    option(LLAMA_BUILD_TESTS "" OFF)
    option(GGML_LLAMAFILE "" OFF)
    if (BUILD_STATIC_LIB)
        option(BUILD_SHARED_LIBS "" OFF)
        option(GGML_BACKEND_DL "" OFF)
    endif()

    if (APPLE)
        option(GGML_METAL "" ON)
    endif()

    FetchContent_MakeAvailable(llama_cpp)
endif()

if (ENABLE_QNN_BACKEND)
    if (WIN32)
        file(GLOB QNN_SRCS src/backends/qnn/rwkv-qualcomm/WrapperUtils/*.cpp src/backends/qnn/rwkv-qualcomm/PAL/src/common/*.cpp
                        src/backends/qnn/rwkv-qualcomm/PAL/src/windows/*.cpp)
        set(RWKV_MOBILE_SRCS ${RWKV_MOBILE_SRCS} src/backends/qnn/rwkv-qualcomm/Utils/DataUtil.cpp
                src/backends/qnn/rwkv-qualcomm/Utils/DynamicLoadUtil.cpp
                src/backends/qnn/rwkv-qualcomm/Utils/IOTensor.cpp
                src/backends/qnn/rwkv-qualcomm/Utils/Utils.cpp
                src/backends/qnn/rwkv-qualcomm/Utils/ClientBuffer.cpp
                src/backends/qnn/rwkv-qualcomm/Utils/dlwrap.cpp
                src/backends/qnn/rwkv-qualcomm/Utils/RpcMem.cpp)
    else()
        file(GLOB QNN_SRCS src/backends/qnn/rwkv-qualcomm/Utils/*.cpp src/backends/qnn/rwkv-qualcomm/WrapperUtils/*.cpp src/backends/qnn/rwkv-qualcomm/PAL/src/common/*.cpp
                        src/backends/qnn/rwkv-qualcomm/PAL/src/linux/*.cpp)
    endif()
    set(RWKV_MOBILE_SRCS ${RWKV_MOBILE_SRCS} ${QNN_SRCS})
    set(RWKV_MOBILE_SRCS ${RWKV_MOBILE_SRCS} src/backends/qnn/qnn_backend.cpp)
endif()

if (ENABLE_WHISPER OR ENABLE_TTS)
    set(RWKV_MOBILE_SRCS ${RWKV_MOBILE_SRCS}
        src/multimodal/utils/audio.cpp
    )

    if (ENABLE_TTS)
        set(RWKV_MOBILE_SRCS ${RWKV_MOBILE_SRCS}
            src/multimodal/tts/sparktts.cpp
            src/multimodal/tts/frontend_utils.cpp
        )

        if (NOT WIN32)
            include(kaldifst)
        endif()
    endif()

    message(CHECK_START "Fetching Eigen3")
    FetchContent_Declare(
        Eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG master
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE
    )
    # note: To disable eigen tests,
    # you should put this code in a add_subdirectory to avoid to change
    # BUILD_TESTING for your own project too since variables are directory
    # scoped
    set(BUILD_TESTING OFF)
    set(EIGEN_BUILD_TESTING OFF)
    set(EIGEN_MPL2_ONLY ON)
    set(EIGEN_BUILD_PKGCONFIG OFF)
    set(EIGEN_BUILD_DOC OFF)
    FetchContent_MakeAvailable(Eigen)

    FetchContent_Declare(
        kaldi_native_fbank
        GIT_REPOSITORY https://github.com/csukuangfj/kaldi-native-fbank/
        GIT_TAG v1.21.1
    )
    option(KALDI_NATIVE_FBANK_BUILD_PYTHON "" OFF)
    option(KALDI_NATIVE_FBANK_BUILD_TESTS "" OFF)
    set(BUILD_SHARED_LIBS_BACKUP ${BUILD_SHARED_LIBS})
    set(BUILD_SHARED_LIBS OFF)
    FetchContent_MakeAvailable(kaldi_native_fbank)
    set(BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS_BACKUP})
endif()

if (ENABLE_COREML_BACKEND)
    set(RWKV_MOBILE_SRCS ${RWKV_MOBILE_SRCS} src/backends/coreml/coreml_rwkv_backend.cpp)
endif()

if (WIN32)
    add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
    add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
endif()

add_library(rwkv_mobile_internal STATIC ${RWKV_MOBILE_SRCS})
target_include_directories(rwkv_mobile_internal PUBLIC src)

if (WIN32)
    target_compile_definitions(rwkv_mobile_internal PRIVATE _DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)
endif()

if (ENABLE_WHISPER OR ENABLE_TTS)
    target_compile_definitions(rwkv_mobile_internal PUBLIC ENABLE_TTS)
    target_link_libraries(rwkv_mobile_internal PUBLIC Eigen3::Eigen)
    if (ENABLE_TTS)
        if (NOT WIN32)
            target_link_libraries(rwkv_mobile_internal PUBLIC
                $<TARGET_OBJECTS:kaldi-native-fbank-core>
                $<TARGET_OBJECTS:kaldifst_core>
                $<TARGET_OBJECTS:fst>
            )

            target_include_directories(rwkv_mobile_internal PUBLIC
                ${kaldi_native_fbank_SOURCE_DIR}
                ${openfst_SOURCE_DIR}/src/include
                ${kaldifst_SOURCE_DIR}
            )
        else()
            target_link_libraries(rwkv_mobile_internal PUBLIC
                $<TARGET_OBJECTS:kaldi-native-fbank-core>
            )
            target_include_directories(rwkv_mobile_internal PUBLIC
                ${kaldi_native_fbank_SOURCE_DIR}
            )
        endif()
    endif()
    target_include_directories(rwkv_mobile_internal PUBLIC ${Eigen_SOURCE_DIR}
        src/multimodal/tts src/multimodal/utils)
endif()

if (ENABLE_WEBRWKV_BACKEND)
    if (APPLE)
        set(WEBRWKV_EXTRA_LIBS "-framework QuartzCore -framework Metal -framework CoreGraphics -framework CoreFoundation -lobjc -liconv")
    elseif (WIN32)
        set(WEBRWKV_EXTRA_LIBS ws2_32 opengl32 d3d12 d3dcompiler userenv kernel32 user32 ntdll bcrypt propsys RuntimeObject)
    endif()
    target_compile_definitions(rwkv_mobile_internal PUBLIC ENABLE_WEBRWKV)
    target_include_directories(rwkv_mobile_internal PUBLIC src/backends/web_rwkv ${web_rwkv_ffi_SOURCE_DIR}/include)
    if (IOS)
        target_compile_options(rwkv_mobile_internal PUBLIC "-I${web_rwkv_ffi_SOURCE_DIR}/include")
    endif()
    target_link_libraries(rwkv_mobile_internal PUBLIC web_rwkv_ffi ${WEBRWKV_EXTRA_LIBS})
endif()

if (ENABLE_NCNN_BACKEND)
    target_compile_definitions(rwkv_mobile_internal PUBLIC ENABLE_NCNN)
    target_include_directories(rwkv_mobile_internal PUBLIC src/backends/ncnn)
    target_link_libraries(rwkv_mobile_internal PUBLIC "$<LINK_LIBRARY:WHOLE_ARCHIVE,ncnn>")
endif()

if (ENABLE_COREML_BACKEND)
    find_library(FOUNDATION_FRAMEWORK Foundation)
    find_library(COREML_FRAMEWORK CoreML)

    if (COREML_FRAMEWORK)
        message(STATUS "CoreML framework found")
    endif()

    add_library(rwkv_coreml STATIC
        src/backends/coreml/rwkv-coreml.h
        src/backends/coreml/rwkv-coreml.mm
        src/backends/coreml/rwkv-coreml-impl.h
        src/backends/coreml/rwkv-coreml-impl.m
        src/backends/coreml/rwkv-coreml-stateful-impl.h
        src/backends/coreml/rwkv-coreml-stateful-impl.m
    )

    target_include_directories(rwkv_coreml PUBLIC src)
    target_link_libraries(rwkv_coreml PUBLIC ${FOUNDATION_FRAMEWORK} ${COREML_FRAMEWORK})
    set_target_properties(rwkv_coreml PROPERTIES
        COMPILE_FLAGS "-fobjc-arc"
        XCODE_ATTRIBUTE_CLANG_ENABLE_OBJC_ARC YES
    )
    set_target_properties(rwkv_coreml PROPERTIES FOLDER "libs")

    target_link_libraries(rwkv_mobile_internal PUBLIC rwkv_coreml)

    target_compile_definitions(rwkv_mobile_internal PUBLIC ENABLE_COREML)
    target_include_directories(rwkv_mobile_internal PUBLIC src/backends/coreml)

    # add_executable(test_coreml src/backends/coreml/test_coreml.cpp)
    # target_link_libraries(test_coreml PUBLIC rwkv_coreml)
endif()

if (ENABLE_LLAMACPP_BACKEND)
    if (APPLE AND GGML_ACCELERATE)
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
        target_include_directories(rwkv_mobile_internal PUBLIC src/multimodal/whisper src/multimodal/utils)
        target_compile_definitions(rwkv_mobile_internal PUBLIC ENABLE_WHISPER)
    endif()

    target_compile_definitions(rwkv_mobile_internal PUBLIC ENABLE_LLAMACPP)
    target_include_directories(rwkv_mobile_internal PUBLIC src/backends/llama_cpp ${llama_cpp_SOURCE_DIR}/include ${llama_cpp_SOURCE_DIR}/src ${llama_cpp_SOURCE_DIR}/ggml/include)
    target_link_libraries(rwkv_mobile_internal PUBLIC $<TARGET_OBJECTS:llama> $<TARGET_OBJECTS:ggml> $<TARGET_OBJECTS:ggml-base> $<TARGET_OBJECTS:ggml-cpu> ${LLAMACPP_EXTRA_LIBS})
endif()

if (ENABLE_QNN_BACKEND)
    target_compile_definitions(rwkv_mobile_internal PUBLIC ENABLE_QNN)
    target_include_directories(rwkv_mobile_internal PUBLIC src/backends/qnn
                               src/backends/qnn/rwkv-qualcomm
                               src/backends/qnn/rwkv-qualcomm/Utils
                               src/backends/qnn/rwkv-qualcomm/PAL/include
                               src/backends/qnn/rwkv-qualcomm/WrapperUtils
                               src/backends/qnn/rwkv-qualcomm/QNN/)
    if (WIN32)
        target_link_libraries(rwkv_mobile_internal PRIVATE Shlwapi Shell32)
    endif()
endif()

if (ENABLE_MNN_BACKEND)
    target_compile_definitions(rwkv_mobile_internal PUBLIC ENABLE_MNN)
    target_link_libraries(rwkv_mobile_internal PUBLIC
        $<LINK_LIBRARY:WHOLE_ARCHIVE,MNN>
    )
    target_include_directories(rwkv_mobile_internal PUBLIC src/backends/mnn ${mnn_SOURCE_DIR}/include)
endif()

if (BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if (DEFINED ANDROID_NDK)
    find_library(log-lib log)
    find_library(android-lib android)
    target_link_libraries(rwkv_mobile_internal PUBLIC ${log-lib} ${android-lib})
endif()

if (BUILD_STATIC_LIB)
    add_library(rwkv_mobile_static STATIC src/c_api.cpp)
    target_include_directories(rwkv_mobile_static PUBLIC src)
    target_link_libraries(rwkv_mobile_static PRIVATE rwkv_mobile_internal)

    set (STATIC_LIBS_LIST $<TARGET_FILE:rwkv_mobile_static> $<TARGET_FILE:rwkv_mobile_internal>)

    if (ENABLE_WEBRWKV_BACKEND)
        target_link_libraries(rwkv_mobile_static PRIVATE web_rwkv_ffi ${WEBRWKV_EXTRA_LIBS})
        set (STATIC_LIBS_LIST ${STATIC_LIBS_LIST} ${CMAKE_BINARY_DIR}/Release/cargo/build/${Rust_CARGO_TARGET}/release/libweb_rwkv_ffi.a)
    endif()

    if (ENABLE_LLAMACPP_BACKEND)
        set (STATIC_LIBS_LIST ${STATIC_LIBS_LIST} $<TARGET_FILE:llama> $<TARGET_FILE:ggml> $<TARGET_FILE:ggml-base> $<TARGET_FILE:ggml-cpu>)
        if (GGML_METAL)
            target_link_libraries(rwkv_mobile_static PRIVATE ${LLAMACPP_EXTRA_LIBS})
            set (STATIC_LIBS_LIST ${STATIC_LIBS_LIST} $<TARGET_FILE:ggml-metal> $<TARGET_FILE:ggml-blas>)
        endif()
    endif()

    # if (ENABLE_NCNN_BACKEND)
    #     set (STATIC_LIBS_LIST ${STATIC_LIBS_LIST} $<TARGET_FILE:ncnn>)
    # endif()

    if (ENABLE_COREML_BACKEND)
        set (STATIC_LIBS_LIST ${STATIC_LIBS_LIST} $<TARGET_FILE:rwkv_coreml>)
    endif()

    if (ENABLE_WHISPER OR ENABLE_TTS)
        set (STATIC_LIBS_LIST ${STATIC_LIBS_LIST}
            $<TARGET_FILE:kaldi-native-fbank-core>
            $<TARGET_FILE:kaldifst_core>
            $<TARGET_FILE:fst>
        )
        if (IOS)
            set (STATIC_LIBS_LIST ${STATIC_LIBS_LIST} $<TARGET_FILE:MNN>)
        endif()
    endif()

    add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/librwkv_mobile.a
        COMMAND libtool -static -o ${CMAKE_BINARY_DIR}/librwkv_mobile.a ${STATIC_LIBS_LIST}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Combining static libraries for rwkv_mobile"
    )
    add_custom_target(rwkv_mobile ALL DEPENDS ${CMAKE_BINARY_DIR}/librwkv_mobile.a)
else()
    add_library(rwkv_mobile SHARED src/c_api.cpp)
    target_link_libraries(rwkv_mobile PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,rwkv_mobile_internal>")
    if (WIN32)
        target_compile_definitions(rwkv_mobile PRIVATE _DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)
    endif()
    if (ANDROID)
        target_link_options(rwkv_mobile PRIVATE "-static-openmp")
    endif()
    if (ENABLE_WEBRWKV_BACKEND)
        target_link_libraries(rwkv_mobile PRIVATE web_rwkv_ffi ${WEBRWKV_EXTRA_LIBS})
    endif()
endif()
