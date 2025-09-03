/* Copyright 2025 The xLLM Authors. All Rights Reserved.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include "block_manager.h"
#include "kv_cache/kv_cache_event.h"

namespace xllm {

class BlockManagerImpl : public std::enable_shared_from_this<BlockManagerImpl>,
                        public BlockManager {
 public:
  explicit BlockManagerImpl(const Options& options);
  virtual ~BlockManagerImpl() {
    CHECK_EQ(num_free_blocks_, free_blocks_.size() - 1)
        << "Not all blocks have been freed";
  };

  // Try to allocate blocks with num_blocks,
  // return {} if not enough blocks
  std::vector<Block> allocate(size_t num_blocks) override;

  void deallocate(const Slice<Block>& blocks) override;

  // allocate shared blocks when enable prefix cache
  std::vector<Block> allocate_shared(
      const Slice<int32_t>& tokens_ids,
      const Slice<Block>& existed_shared_blocks = {}) override;

  // cache blocks when enable prefix cache
  void cache(const Slice<int32_t>& token_ids,
             std::vector<Block>& blocks) override;

  void get_merged_kvcache_event(KvCacheEvent* event) const override;

  size_t num_blocks_in_prefix_cache() const override {
    if (options_.enable_prefix_cache()) {
      CHECK(prefix_cache_);
      return prefix_cache_->num_blocks();
    }
    return 0;
  }

  // free blocks num
  size_t num_free_blocks() const override { return num_free_blocks_; }

  // used blocks num
  size_t num_used_blocks() const override { return num_used_blocks_; }

  // current kv cache utilization.
  double kv_cache_utilization() const override {
    return static_cast<double>(num_used_blocks_) / num_total_blocks();
  }

  float get_gpu_cache_usage_perc() const override {
    return 1.0 - num_free_blocks_ * 1.0 / num_total_blocks();
  }

  // call BlockManager to free block used by Block.
  void free(int32_t block_id) override;

  // allocate a list of blocks
  // std::vector<Block> allocate(uint32_t n_blocks) override;

  // allocate a block
  Block allocate() override;

  // total blocks num
  size_t num_total_blocks() const override { return free_blocks_.size() - 1; }

 private:
  // check if has enough slots, if not, try to evict some blocks
  // from the prefix cache
  bool has_enough_blocks(uint32_t num_blocks);

 private:
  // prefix cache
  std::unique_ptr<PrefixCache> prefix_cache_;

  // reserved block id for padding
  Block padding_block_;

  // number of used blocks
  std::atomic<size_t> num_used_blocks_{0};

  // free block count
  size_t num_free_blocks_ = 0;

  // block size
  size_t block_size_ = 0;

  // free block list
  std::vector<int32_t> free_blocks_;
};

}  // namespace xllm
