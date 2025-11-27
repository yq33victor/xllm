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

#include <algorithm>
#include <vector>

namespace xllm {

// TODO: refactor/rename
class MasterCoordinator {
 public:
  MasterCoordinator(int32_t serve_model_num)
      : serve_model_num_(serve_model_num) {
    CHECK(serve_model_num > 0) << "serve_model_num must be greater than 0";
    estimate_kv_mem.resize(serve_model_num);
  }

  void set_estimate_kv_mem(int32_t model_idx, int64_t kv_mem) {
    LOG(ERROR) << "MasterCoordinator: model_idx = " << model_idx
               << ", kv_mem = " << kv_mem;
    estimate_kv_mem[model_idx] = kv_mem;
  }

  int64_t plan_mem() {
    auto min_mem =
        std::min_element(estimate_kv_mem.begin(), estimate_kv_mem.end());
    LOG(ERROR) << "MasterCoordinator: plan mem = "
               << (*min_mem / serve_model_num_);
    return (*min_mem / serve_model_num_);
  }

 private:
  int32_t serve_model_num_ = 0;
  std::vector<int64_t> estimate_kv_mem;
};

}  // namespace xllm