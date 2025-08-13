/*
* Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
*
*  * Licensed under the Apache License, Version 2.0 (the "License");
*  * you may not use this file except in compliance with the License.
*  * You may obtain a copy of the License at
*  *
*  * http://www.apache.org/licenses/LICENSE-2.0
*  *
*  * Unless required by applicable law or agreed to in writing, software
*  * distributed under the License is distributed on an "AS IS" BASIS,
*  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  * See the License for the specific language governing permissions and
*  * limitations under the License.
*  */
#ifndef ATB_SPEED_MODELS_ALL2ALL_MATMUL_OPERATION_H
#define ATB_SPEED_MODELS_ALL2ALL_MATMUL_OPERATION_H
#include <atb/atb_infer.h>
#include "atb_speed/utils/operation_util.h"
#include "atb_speed/log.h"
#include "operations/fusion/linear/linear.h"
#include "operations/fusion/linear/linear_parallel.h"
#include "operations/fusion/norm/norm_linear.h"

namespace atb_speed {
namespace common {
struct All2AllMatmulParam {
    bool transpose = true;
    bool supportSwiGLU = true;
    int32_t topk = 2;
    int gmmQuantType = 0;
    uint32_t numOfExperts = 8;
    uint32_t numOfDeviceExperts = 8;
    std::vector<int> moeLinearQuantType = {};
    bool hasBias = false;
    bool isBF16 = false;
    bool gateUpTransposeB = false;
    bool downTransposeB = false;
    int packQuantType = atb_speed::common::PackQuantType::ALL_FP;
    int denseQuantType = atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED;

    std::vector<int32_t> deviceExpert = {0, 1, 2, 3, 4, 5, 6, 7};
    int expertParallelDegree = 0;
    bool enableFusedRouting = false;

    std::string backend = "hccl";

    bool hasMoeEp = false;
    int moeEpRank = 0;
    int moeEpSize = 1;
    std::string moeEpDomain = "";
    std::string moeEpRankTableFile = "";

    bool hasMlpTp = false;
    int mlpTpRank = 0;
    int mlpTpSize = 1;
    std::string mlpTpDomain = "";
    std::string mlpTpRankTableFile = "";
};

atb::Status CreateAll2AllMatmulOperation(const All2AllMatmulParam &param, atb::Operation **operation);
}
} // namespace atb_speed
#endif