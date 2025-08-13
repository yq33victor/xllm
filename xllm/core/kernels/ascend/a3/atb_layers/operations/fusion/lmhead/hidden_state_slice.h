/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef ATB_SPEED_MODELS_COMMON_LAYER_HIDDEN_STATE_SLICE_H
#define ATB_SPEED_MODELS_COMMON_LAYER_HIDDEN_STATE_SLICE_H

#include "atb/atb_infer.h"
#include "atb_speed/utils/operation_util.h"
#include "operations/fusion/linear/linear_parallel.h"

namespace atb_speed {
namespace common {
/// A struct defines `HiddenStateSlice` operation's parameters.
struct HiddenStateSliceParam {
    /// The rank of this device in the tensor parallelism communication domain in lmhead.
    int rank = 0;
    /// The size of the tensor parallelism communication domain in lmhead.
    int world_size = 1;
};

/// Create `HiddenStateSlice` graph operation.
/// \param param `HiddenStateSlice`'s parameters, see `HiddenStateSliceParam` for more details.
/// \param operation The address pointer to the `HiddenStateSlice` operation.
/// Operation's Inputs:
/// Name                   | Dtype | Shape |
/// ---------------------- | ----- | ----- |
/// in_hidden_states       | float16/float/int8/bool/int32/uint32/bf16 | [all_token_size, vocab_size] |
/// Operation's Outputs:
/// Name                   | Dtype | Shape |
/// ---------------------- | ----- | ----- |
/// output                 | float16/float/int8/bool/int32/uint32/bf16 | [token_size, vocab_size] |
atb::Status HiddenStateSlice(const HiddenStateSliceParam &param, atb::Operation **operation);
}  // namespace common
}  // namespace atb_speed
#endif
