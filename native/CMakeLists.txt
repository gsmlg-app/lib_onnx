cmake_minimum_required(VERSION 3.15)
project(lib_onnx_bridge C)

set(CMAKE_C_STANDARD 11)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime/include
)

add_library(lib_onnx_bridge SHARED
    src/lib_onnx_bridge.c
)

if(APPLE)
    find_library(COREML_FRAMEWORK CoreML)
    target_link_libraries(lib_onnx_bridge
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime/lib/macos_arm64/libonnxruntime.dylib
        ${COREML_FRAMEWORK}
        pthread
    )
    set_target_properties(lib_onnx_bridge PROPERTIES
        INSTALL_RPATH "@loader_path/../third_party/onnxruntime/lib/macos_arm64;@loader_path/."
    )
elseif(ANDROID)
    target_link_libraries(lib_onnx_bridge
        onnxruntime
        log
        pthread
    )
else()
    target_link_libraries(lib_onnx_bridge
        onnxruntime
        pthread
    )
endif()
