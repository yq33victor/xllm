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

#include "block.h"

#include <glog/logging.h>
#include <string.h>

#include <cstdint>

#include "block_manager.h"

namespace xllm {
Block::Block(int32_t id, std::shared_ptr<BlockManager> manager)
    : id_(id), ref_count_(new uint32_t(1)), manager_(manager) {
  // get the block size from the manager
  size_ = manager_ == nullptr ? 0 : manager_->block_size();
}

Block::~Block() {
  // decrease reference count
  dec_ref_count();
}

// copy constructor
Block::Block(const Block& other)
    : id_(other.id_),
      size_(other.size_),
      ref_count_(other.ref_count_),
      manager_(other.manager_) {
  memcpy(hash_value_, other.hash_value_, MURMUR_HASH3_VALUE_LEN);

  token_ids_.reserve(other.token_ids_.size());
  token_ids_.assign(other.token_ids_.begin(), other.token_ids_.end());

  // increase reference count
  inc_ref_count();
}

// copy assignment
Block& Block::operator=(const Block& other) {
  if (this != &other) {
    dec_ref_count();

    id_ = other.id_;
    size_ = other.size_;
    manager_ = other.manager_;
    ref_count_ = other.ref_count_;

    memcpy(hash_value_, other.hash_value_, MURMUR_HASH3_VALUE_LEN);

    token_ids_.reserve(other.token_ids_.size());
    token_ids_.assign(other.token_ids_.begin(), other.token_ids_.end());

    inc_ref_count();
  }
  return *this;
}

Block::Block(Block&& other) noexcept
    : id_(other.id_),
      size_(other.size_),
      ref_count_(other.ref_count_),
      manager_(other.manager_) {
  memcpy(hash_value_, other.hash_value_, MURMUR_HASH3_VALUE_LEN);
  token_ids_.swap(other.token_ids_);

  // reset other without adjusting the reference count
  other.id_ = -1;
  other.size_ = 0;
  other.ref_count_ = nullptr;
  other.manager_ = nullptr;
}

Block& Block::operator=(Block&& other) noexcept {
  if (this != &other) {
    dec_ref_count();

    id_ = other.id_;
    size_ = other.size_;
    manager_ = other.manager_;
    ref_count_ = other.ref_count_;

    memcpy(hash_value_, other.hash_value_, MURMUR_HASH3_VALUE_LEN);
    token_ids_.swap(other.token_ids_);

    other.id_ = -1;
    other.size_ = 0;
    other.ref_count_ = nullptr;
    other.manager_ = nullptr;
  }

  return *this;
}

void Block::inc_ref_count() {
  if (ref_count_ != nullptr) {
    ++(*ref_count_);
  }
}

void Block::dec_ref_count() {
  if (ref_count_ != nullptr && --(*ref_count_) == 0) {
    // release the reference count memory
    delete ref_count_;
    // return the block id to the manager
    if (manager_ != nullptr) {
      manager_->free(id_);
    }

    token_ids_.clear();
  }
}

}  // namespace xllm