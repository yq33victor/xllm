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

#include "block_manager_impl.h"

namespace xllm {

class ConcurrentBlockManagerImpl : public std::enable_shared_from_this<ConcurrentBlockManagerImpl>,
                                  public BlockManagerImpl {
 public:
  explicit ConcurrentBlockManagerImpl(const Options& options);
  virtual ~ConcurrentBlockManagerImpl() = default;

  // Try to allocate blocks with num_blocks,
  // return {} if not enough blocks
  std::vector<Block> allocate(size_t num_blocks) override;

  void deallocate(const Slice<Block>& blocks) override;

  // try to share blocks among sequences with the same prefix
  std::vector<Block> allocate_shared(
      const Slice<int32_t>& tokens_ids,
      const Slice<Block>& existed_shared_blocks = {}) override;

  // cache the blocks
  void cache(const Slice<int32_t>& token_ids,
             std::vector<Block>& blocks) override;

  // get the number of blocks in the prefix cache
  size_t num_blocks_in_prefix_cache() const override;

  // get the number of free blocks in the block allocator
  size_t num_free_blocks() const override;

  // get the block utilization.
  double kv_cache_utilization() const override;

 private:
  // mutex for disagg prefill/decode mode
  mutable std::mutex mutex_;
};

}  // namespace xllm
