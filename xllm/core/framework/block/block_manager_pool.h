/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <memory>
#include <vector>

#include "block_manager.h"
#include "framework/model/model_input_params.h"
#include "framework/request/request.h"
#include "framework/request/sequence.h"

namespace xllm {

class BlockManagerPool {
 public:
  struct Options {
    PROPERTY(uint32_t, num_blocks) = 0;
    PROPERTY(uint32_t, host_num_blocks) = 0;
    PROPERTY(int32_t, block_size) = 0;
    PROPERTY(bool, enable_prefix_cache) = true;
    PROPERTY(bool, enable_disagg_pd) = false;
    PROPERTY(bool, enable_cache_upload) = false;
    PROPERTY(bool, enable_kvcache_store) = false;
  };

  explicit BlockManagerPool(const Options& options, int32_t dp_size = 1);

  ~BlockManagerPool() = default;

  BlockManager* get_block_manager(Sequence* sequence, bool is_host);

  bool allocate(Sequence* sequence);
  bool allocate(std::vector<Sequence*>& sequences);
  bool allocate(Sequence* sequence, size_t num_tokens);

  uint32_t pre_allocate(Sequence* sequence);

  // Try to allocate blocks with num_tokens,
  // return {} if not enough blocks
  std::vector<Block> allocate(size_t num_tokens, int32_t& dp_rank);

  void deallocate(Request* request);
  void deallocate(std::vector<Sequence*>& sequences);
  void deallocate(Sequence* sequence);

  void allocate_shared(Sequence* sequence);
  void cache(Sequence* sequence);

  std::shared_ptr<std::vector<std::vector<CacheBlockInfo>>> get_copy_in_cache_block_infos();
  std::shared_ptr<std::vector<std::vector<CacheBlockInfo>>> get_copy_out_cache_block_infos();
  void reset_copy_content();

  void get_merged_kvcache_event(KvCacheEvent* event) const;
  float get_gpu_cache_usage_perc() const;

  std::vector<size_t> num_blocks_in_prefix_cache() const;
  std::vector<size_t> num_free_blocks() const;
  std::vector<size_t> num_used_blocks() const;
  double kv_cache_utilization() const;

  // get the options for the block manager
  const Options& options() const { return options_; }

 private:
  int32_t get_manager_with_max_free_blocks() const;
  int32_t get_dp_rank(Sequence* sequence) const;

  void allocate_host_shared(Sequence* sequence);
  void cache_host(Sequence* sequence);

 private:
  std::vector<std::shared_ptr<BlockManager>> block_managers_;
  std::vector<std::shared_ptr<BlockManager>> host_block_managers_;

  // the options for the block manager
  Options options_;

  // CacheBlockInfo per step
  std::shared_ptr<std::vector<std::vector<CacheBlockInfo>>> copy_in_cache_block_infos_;
  std::shared_ptr<std::vector<std::vector<CacheBlockInfo>>> copy_out_cache_block_infos_;
  std::vector<std::vector<Block>> evict_host_blocks_;
};

}  // namespace xllm
