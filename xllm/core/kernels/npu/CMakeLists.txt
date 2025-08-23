cmake_minimum_required(VERSION 3.26)
project(xLLM)

include_directories(
    ${CMAKE_SOURCE_DIR}/xllm/core/kernels/npu
    ${CMAKE_SOURCE_DIR}/third_party/
    ${CMAKE_SOURCE_DIR}/third_party/spdlog/include
    $ENV{ATB_HOME_PATH}/include
    $ENV{ASCEND_HOME_PATH}/include
)

link_directories(
    $ENV{ASCEND_HOME_PATH}/lib64
    $ENV{ATB_HOME_PATH}/lib
    $ENV{ASCEND_TOOLKIT_HOME}/lib64)

if(USE_A3)
    add_subdirectory(a3)
else()
    add_subdirectory(a2)
endif()
add_subdirectory(xllm_ops)