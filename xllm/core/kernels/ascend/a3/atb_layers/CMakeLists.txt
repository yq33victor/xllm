cmake_minimum_required(VERSION 3.26)
project(xLLM)

include_directories(
    ${CMAKE_SOURCE_DIR}/xllm/core/kernels/ascend/a3/atb_layers/
    ${CMAKE_SOURCE_DIR}/third_party/
    ${CMAKE_SOURCE_DIR}/third_party/spdlog/include
    ${CMAKE_SOURCE_DIR}/xllm/core/kernels/ascend/a3/atb_layers/core/include
    $ENV{ATB_HOME_PATH}/include
    $ENV{ASCEND_HOME_PATH}/include
)

link_directories(
    $ENV{ASCEND_HOME_PATH}/lib64
    $ENV{ATB_HOME_PATH}/lib
    $ENV{ASCEND_TOOLKIT_HOME}/lib64)

add_subdirectory(pytorch)
add_subdirectory(core)
add_subdirectory(operations)
add_subdirectory(models)
