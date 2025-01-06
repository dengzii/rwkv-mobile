cmake_minimum_required(VERSION 3.25)

project(rwkv-mobile CXX)

set(CMAKE_CXX_STANDARD 17)

include(FetchContent)

option(ENABLE_NCNN_BACKEND "Enable NCNN backend" ON)
option(ENABLE_WEBRWKV_BACKEND "Enable WebRWKV backend" ON)

option(BUILD_EXAMPLES "Build examples" ON)
option(BUILD_STATIC_LIB "Build static library" OFF)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

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
            GIT_TAG molly
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

add_library(rwkv_mobile_internal ${RWKV_MOBILE_SRCS})
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
    target_link_libraries(rwkv_mobile_internal PUBLIC ncnn)
endif()

if (BUILD_EXAMPLES)
    add_executable(gen examples/gen.cpp)
    target_link_libraries(gen PUBLIC rwkv_mobile_internal)

    add_executable(test_chat examples/test_chat.cpp)
    target_link_libraries(test_chat PUBLIC rwkv_mobile_internal)
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
    if (ENABLE_WEBRWKV_BACKEND)
        target_include_directories(rwkv_mobile PUBLIC backends/web_rwkv ${web_rwkv_ffi_SOURCE_DIR}/include src)
        target_link_libraries(rwkv_mobile PRIVATE web_rwkv_ffi ${WEBRWKV_EXTRA_LIBS})
    endif()

    if (ENABLE_NCNN_BACKEND)
        target_link_libraries(rwkv_mobile PRIVATE ncnn)
    endif()
else()
    add_library(rwkv_mobile SHARED src/c_api.cpp)
    target_link_libraries(rwkv_mobile PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,rwkv_mobile_internal>")
    if (ENABLE_WEBRWKV_BACKEND)
        target_link_libraries(rwkv_mobile PRIVATE web_rwkv_ffi ${WEBRWKV_EXTRA_LIBS})
    endif()

    if (ENABLE_NCNN_BACKEND)
        target_link_libraries(rwkv_mobile PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,ncnn>")
    endif()
endif()