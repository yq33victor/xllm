include(cc_library)

if(USE_MLU)
  list(APPEND ATTN_HDRS mlu/attention.h)
  list(APPEND ATTN_HDRS deepseek_v2_attention.h)
  list(APPEND ATTN_SRCS mlu/attention.cpp)
  list(APPEND ATTN_SRCS deepseek_v2_attention.cpp)
elseif(USE_CUDA)
  list(APPEND ATTN_HDRS gpu/attention.h)
  list(APPEND ATTN_HDRS gpu/flashinfer_workspace.h)
  list(APPEND ATTN_SRCS gpu/attention.cpp)
  list(APPEND ATTN_SRCS gpu/flashinfer_workspace.cpp)
  # support deepseek_v2
endif()

cc_library(
  NAME
    common_layers
  HDRS
    qwen3_attention.h
    fuse_norm.h
    rotary_embedding.h
    fused_moe.h
    dense_mlp.h
    qwen3_decoder_layer.h
    qwen3_moe_decoder_layer.h
    linear_impl.h
    linear.h
    word_embedding_impl.h
    layer_utils.h
    indexer.h
    ${ATTN_HDRS}
  SRCS
    qwen3_attention.cpp
    fuse_norm.cpp
    rotary_embedding.cpp
    fused_moe.cpp
    dense_mlp.cpp
    qwen3_decoder_layer.cpp
    qwen3_moe_decoder_layer.cpp
    linear_impl.cpp
    layer_utils.cpp
    indexer.cpp
    ${ATTN_SRCS}
  DEPS
    "-Wl,--whole-archive"
    "-Wl,--no-whole-archive"
    :kv_cache
    :prefix_cache
    :block
    :parallel_state
    :state_dict
    :model
    :kernels
    glog::glog
    gflags::gflags
    torch
    :platform
)

add_subdirectory(tests)
