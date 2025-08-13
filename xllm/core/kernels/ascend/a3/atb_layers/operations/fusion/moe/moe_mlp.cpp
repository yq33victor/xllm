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
#include "moe_mlp.h"
#include <atb/atb_infer.h>
#include <memory>
#include <algorithm>
#include "operations/fusion/moe/integrated_gmm.h"
#include "operations/fusion/moe/ep/expert_filter.h"
#include "operations/aclnn/ops/finalize_routing_operation.h"
#include "operations/aclnn/ops/moe_init_routing_operation.h"
#include "operations/aclnn/ops/moe_init_routing_quant_operation.h"
#include "operations/aclnn/ops/moe_compute_expert_tokens_operation.h"
#include "operations/aclnn/ops/moetoken_unpermute_operation.h"
#include "operations/aclnn/ops/inplacemasked_filltensor_operation.h"
#include "operations/aclnn/ops/grouped_matmul_operation.h"
#include "atb_speed/base/event_manager.h"
#include "operations/aclnn/ops/moe_distribute_combine_operation.h"
#include "operations/aclnn/ops/moe_distribute_dispatch_operation.h"

namespace atb_speed {
namespace common {

static const uint64_t NUM2 = 2;
static const uint64_t NUM3 = 3;
static const uint64_t NUM4 = 4;
static const uint64_t NUM5 = 5;

int CalcUpGmmQuantType(const MoeMlpParam &param)
{
    int gmmQuantType = 0;
    int tempQuantType = atb_speed::common::GetLinearQuantType(
        param.denseQuantType == atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED ?
            param.packQuantType : param.denseQuantType,
        param.moeLinearQuantType[IntegratedGmmIdx::MOE_MLP_GATE_IDX], false);
    switch (tempQuantType) {
        case LinearQuantType::NO_QUANT:
            gmmQuantType = GmmQuantType::NONE;
            break;
        case LinearQuantType::LINEAR_W8A8_DYNAMIC_QUANT:
        case LinearQuantType::LINEAR_W8A8_DYNAMIC_DEQUANT:
            gmmQuantType = GmmQuantType::W8A8_TOKEN;
            break;
        case LinearQuantType::W8A16:
            gmmQuantType = GmmQuantType::W8A16_CHANNEL;
            break;
        case LinearQuantType::W4A16:
            gmmQuantType = GmmQuantType::W4A16_CHANNEL;
            break;
        case LinearQuantType::W4A8:
            gmmQuantType = GmmQuantType::W4A8_GROUP;
            break;
        default:
            gmmQuantType = GmmQuantType::W8A8_CHANNEL;
            break;
    }
    return gmmQuantType;
}

bool IsGMMSwigluQuant(const int gmmQuantType, const MoeMlpParam &param)
{
    return gmmQuantType == GmmQuantType::W8A8_TOKEN && param.enableGMMSwigluQuant;
}

std::map<std::string, std::vector<std::string>> GetMoeMlpInTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> moeMlpInTensorCandidates = {
        {"default", {
            "in_hiddenstates", "in_mlp_gateup_weight_expert", "in_mlp_gateup_bias_expert",
            "in_mlp_gateup_descale_expert", "in_mlp_gateup_offset_expert", "in_mlp_gateup_scale_expert",
            "in_mlp_gateup_compress_idx_expert", "in_mlp_down_weight_expert",
            "in_mlp_down_bias_expert", "in_mlp_down_descale_expert", "in_mlp_down_offset_expert",
            "in_mlp_down_scale_expert", "in_mlp_down_compress_idx_expert", "in_expert_array", "in_selected_experts",
            "in_expert_weight"},
        },
        {"ep", {
            "in_zero_hot", "in_start_expert_idx", "in_device_expert_count", "in_padding_idx"}
        }
    };
    return moeMlpInTensorCandidates;
}

std::map<std::string, std::vector<std::string>> GetMoeMlpInterTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> moeMlpInterTensorCandidates = {
        {"default", {
            "intermediate_idx", "intermediate_weight_idx", "intermediate_dummy_zero",
            "intermediate_dummy_one", "intermediate_rev_idx", "intermediate_group_list",
            "intermediate_sorted_hiddenstates", "intermediate_rev_sorted_hiddenstates",
            "intermediate_matmul_gate_up_out", "intermediate_swish_out", "intermediate_mlp_out",
            "intermediate_mlp_out_weighted", "intermediate_sorted_weight"}
        },
        {"default_w8a8_token", {
            "intermediate_idx", "intermediate_weight_idx", "intermediate_dummy_zero",
            "intermediate_dummy_one", "intermediate_rev_idx", "intermediate_group_list",
            "intermediate_sorted_hiddenstates", "intermediate_rev_sorted_hiddenstates",
            "intermediate_mlp_out", "intermediate_mlp_out_weighted", "intermediate_sorted_weight"}
        },
        {"enableFusedRouting", {
            "intermediate_idx", "intermediate_group_list", "intermediate_sorted_hiddenstates",
            "intermediate_matmul_gate_up_out", "intermediate_swish_out", "intermediate_mlp_out"}
        },
        {"disable_mc2", {
            "intermediate_group_list_int64"}
        },
        {"enable_mc2", {
            "intermediate_ep_recv_counts", "intermediate_tp_recv_counts", "intermediate_gmm0_deqscale",
            "intermediate_expand_expert_weight"}
        },
        {"enableFusedRouting_w8a8_token", {
            "intermediate_idx", "intermediate_group_list", "intermediate_sorted_hiddenstates", "intermediate_mlp_out"}
        },
        {"disable_swiglu", {
            "intermediate_matmul_gate_out", "intermediate_matmul_up_out", "intermediate_swish_out_internal"}
        },
        {"enable_init_quant", {
            "intermediate_tokens_before_capacity", "intermediate_sorted_hiddenstates_dequant_scale"}
        },
        {"enable_swiglu_quant", {
            "intermedaite_swiglu_dequant_scale"}
        },
        {"ep", {
            "intermediate_group_list_sliced", "intermediate_expert_weight"}
        },
        {"dynanmic_ep", {
            "intermediate_selected_experts"}
        },
        {"gating_dp", {
            "intermediate_group_list_sliced", "intermediate_expert_weight", "intermediate_selected_experts"}
        },
    };
    return moeMlpInterTensorCandidates;
}

atb::Status AddIntermediateTensor(const MoeMlpParam &param, std::vector<std::string>& interTensorList)
{
    auto moeMlpInterTensorCandidates = GetMoeMlpInterTensorCandidates();
    bool isGMMSwigluQuant = IsGMMSwigluQuant(CalcUpGmmQuantType(param), param);
    if (param.enableFusedRouting) {
        AddTensorToList(moeMlpInterTensorCandidates, param.enableMoeDistribute ?
            "enable_mc2" : "disable_mc2", interTensorList);
        AddTensorToList(moeMlpInterTensorCandidates, isGMMSwigluQuant ? "enableFusedRouting_w8a8_token" :
            "enableFusedRouting", interTensorList);
    } else {
        AddTensorToList(moeMlpInterTensorCandidates, isGMMSwigluQuant ?
            "default_w8a8_token" : "default", interTensorList);
    }
    if (param.enableInitQuant && !param.enableMoeDistribute) {
        AddTensorToList(moeMlpInterTensorCandidates, "enable_init_quant", interTensorList);
    }
    if (param.enableSwigluQuant) {
        AddTensorToList(moeMlpInterTensorCandidates, "enable_swiglu_quant", interTensorList);
    }
    if (!param.supportSwiGLU) {
        AddTensorToList(moeMlpInterTensorCandidates, "disable_swiglu", interTensorList);
    }
    if (param.hasMoeEp && !param.enableMoeDistribute) {
        if (!param.enableGatingDp) {
            AddTensorToList(moeMlpInterTensorCandidates, "ep", interTensorList);
            if (!param.shiftedTopK) {
                AddTensorToList(moeMlpInterTensorCandidates, "dynanmic_ep", interTensorList);
            }
        } else {
            AddTensorToList(moeMlpInterTensorCandidates, "gating_dp", interTensorList);
        }
    }
    return atb::NO_ERROR;
}

std::map<std::string, uint32_t> ConstructTensorMap(
    const MoeMlpParam &param,
    uint32_t &inTensorNum, uint32_t &outTensorNum, uint32_t &interTensorNum)
{
    auto moeMlpInTensorCandidates = GetMoeMlpInTensorCandidates();
    std::vector<std::string> inTensorList = {};
    std::vector<std::string> interTensorList = {};
    std::vector<std::string> outTensorList = {"out_moe_mlp_result"};
    AddTensorToList(moeMlpInTensorCandidates, "default", inTensorList);
    if (param.hasMoeEp) {
        AddTensorToList(moeMlpInTensorCandidates, "ep", inTensorList);
    }

    AddIntermediateTensor(param, interTensorList);

    if (param.enableExpertCumSumOutput) {
        if (param.enableFusedRouting && !param.enableMoeDistribute) {
            interTensorList.erase(std::remove(interTensorList.begin(), interTensorList.end(),
                "intermediate_group_list_int64"), interTensorList.end());
            outTensorList.push_back("intermediate_group_list_int64");
        } else {
            interTensorList.erase(std::remove(interTensorList.begin(), interTensorList.end(),
                "intermediate_group_list"), interTensorList.end());
            outTensorList.push_back("intermediate_group_list");
        }
    }
    inTensorNum = inTensorList.size();
    outTensorNum = outTensorList.size();
    interTensorNum = interTensorList.size();
    return GetTensorMap(inTensorList, outTensorList, interTensorList);
}

atb::Status CreateExpertFilter(
    std::map<std::string, uint32_t> &tensorMap, const MoeMlpParam &param,
    atb::GraphParam &opGraph, size_t &nodeId)
{
    atb::Node &expertFilterNode = opGraph.nodes.at(nodeId++);
    atb_speed::common::ExpertFilterParam expertFilterParam;
    expertFilterParam.numOfExperts = param.numOfExperts;
    expertFilterParam.deviceExpert = param.deviceExpert;
    expertFilterParam.shiftedTopK = param.shiftedTopK;
    expertFilterParam.isBF16 = param.isBF16;
    expertFilterParam.enableGatingDp = param.enableGatingDp;
    atb_speed::common::CreateExpertFilterOperation(expertFilterParam, &expertFilterNode.operation);

    expertFilterNode.inTensorIds = {GetTensorIdx(tensorMap, "in_selected_experts"),
                                    GetTensorIdx(tensorMap, "in_expert_weight"),
                                    GetTensorIdx(tensorMap, "in_start_expert_idx"),
                                    GetTensorIdx(tensorMap, "in_device_expert_count"),
                                    GetTensorIdx(tensorMap, "in_zero_hot")};

    if (param.shiftedTopK && !param.enableGatingDp) {
        expertFilterNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_expert_weight")};
    } else {
        expertFilterNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_selected_experts"),
                                         GetTensorIdx(tensorMap, "intermediate_expert_weight")};
    }
    ATB_SPEED_LOG_DEBUG("InitRouting calculation success");
    return atb::NO_ERROR;
}

// Step 1: hidden state permutation
atb::Status CreateInitRoutingQuant(
    std::map<std::string, uint32_t> &tensorMap, const MoeMlpParam &param,
    atb::GraphParam &opGraph, size_t &nodeId)
{
    atb::Node &initRoutingNode = opGraph.nodes.at(nodeId++);
    atb_speed::common::MoeInitRoutingQuantParam initRoutingParam;
    initRoutingParam.topkNum = param.topk;
    initRoutingParam.expertNum = param.numOfExperts;
    int gmmQuantType = CalcUpGmmQuantType(param);
    if (gmmQuantType == GmmQuantType::W4A8_GROUP) {
        initRoutingParam.expertTokensCoutOrCumsumFlag = 2; // 2 : W4A8_GROUP_Mutmal 不适用累加和形式
    }
    initRoutingNode.operation = new atb_speed::common::MoeInitRoutingQuantOperation("MoeInitRoutingQuantOperation" +
        std::to_string(gmmQuantType), initRoutingParam);
    initRoutingNode.inTensorIds = {GetTensorIdx(tensorMap, "in_hiddenstates"),
                                   GetTensorIdx(tensorMap, param.hasMoeEp && !param.shiftedTopK ? \
                                   "intermediate_selected_experts" : (param.enableGatingDp ? \
                                   "intermediate_selected_experts" : "in_selected_experts"))};
    initRoutingNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_sorted_hiddenstates"),
                               GetTensorIdx(tensorMap, "intermediate_idx"),
                               GetTensorIdx(tensorMap, "intermediate_group_list"),
                               GetTensorIdx(tensorMap, "intermediate_tokens_before_capacity"),
                               GetTensorIdx(tensorMap, "intermediate_sorted_hiddenstates_dequant_scale")};
    ATB_SPEED_LOG_DEBUG("InitRouting calculation success");
    return atb::NO_ERROR;
}

atb::Status CreateInitRouting(
    std::map<std::string, uint32_t> &tensorMap, const MoeMlpParam &param,
    atb::GraphParam &opGraph, size_t &nodeId)
{
    atb::Node &initRoutingNode = opGraph.nodes.at(nodeId++);
    atb_speed::common::MoeInitRoutingParam initRoutingParam;
    initRoutingParam.topkNum = param.topk;
    initRoutingParam.expertNum = param.numOfExperts;
    int gmmQuantType = CalcUpGmmQuantType(param);
    if (gmmQuantType == GmmQuantType::W4A8_GROUP) {
        initRoutingParam.expertTokensCoutOrCumsumFlag = 2; // 2 : W4A8_GROUP_Mutmal 不适用累加和形式
    }
    initRoutingNode.operation = new atb_speed::common::MoeInitRoutingOperation("MoeInitRoutingOperation" +
        std::to_string(gmmQuantType), initRoutingParam);
    initRoutingNode.inTensorIds = {GetTensorIdx(tensorMap, "in_hiddenstates"),
                                   GetTensorIdx(tensorMap, param.hasMoeEp && !param.shiftedTopK ? \
                                   "intermediate_selected_experts" : (param.enableGatingDp ? \
                                   "intermediate_selected_experts" : "in_selected_experts"))};
    initRoutingNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_sorted_hiddenstates"),
                               GetTensorIdx(tensorMap, "intermediate_idx"),
                               GetTensorIdx(tensorMap, "intermediate_group_list")};
    ATB_SPEED_LOG_DEBUG("InitRouting calculation success");
    return atb::NO_ERROR;
}

atb::Status CreateComputeExpertSlice(
    std::map<std::string, uint32_t> &tensorMap, const MoeMlpParam &param,
    atb::GraphParam &opGraph, size_t &nodeId)
{
    atb::Node &sliceNode = opGraph.nodes.at(nodeId++);
    atb::infer::SliceParam sliceParam;
    sliceParam.offsets.resize(1);
    sliceParam.offsets[0] = 0;
    sliceParam.size.resize(1);
    sliceParam.size[0] = param.numOfDeviceExperts;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(sliceParam, &sliceNode.operation));
    sliceNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_group_list")};
    sliceNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_group_list_sliced")};
    ATB_SPEED_LOG_DEBUG("sliceNode calculation success");
    return atb::NO_ERROR;
}

atb::Status CreateCast(std::map<std::string, uint32_t> &tensorMap, const MoeMlpParam &param,
    atb::GraphParam &opGraph, size_t &nodeId)
{
    atb::Node &castNode = opGraph.nodes.at(nodeId++);
    atb::infer::ElewiseParam castParam;
    castParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_CAST;
    castParam.outTensorType = ACL_INT64;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(castParam, &castNode.operation));
    castNode.inTensorIds = {GetTensorIdx(tensorMap, param.hasMoeEp ? \
        "intermediate_group_list_sliced" : "intermediate_group_list")};
    castNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_group_list_int64")};
    ATB_SPEED_LOG_DEBUG("Cast calculation success");
    return atb::NO_ERROR;
}

atb::Status CreateGating(
    std::map<std::string, uint32_t> &tensorMap, const MoeMlpParam &param, std::shared_ptr<int64_t> batchDimPtr,
    atb::GraphParam &opGraph, size_t &nodeId)
{
    atb::Node &gatingNode = opGraph.nodes.at(nodeId++);
    CHECK_PARAM_NE(param.topk, 0);
    CHECK_PARAM_NE(param.numOfExperts, 0);
    atb::infer::GatingParam gatingParam;
    gatingParam.topkExpertNum = param.topk;
    gatingParam.cumSumNum = param.numOfExperts;
    gatingParam.cumSumInt64 = true;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(gatingParam, &gatingNode.operation));
    gatingNode.inTensorIds = {
        GetTensorIdx(tensorMap, "in_selected_experts"), GetTensorIdx(tensorMap, "in_expert_array")};
    gatingNode.outTensorIds = {
        GetTensorIdx(tensorMap, "intermediate_idx"), GetTensorIdx(tensorMap, "intermediate_group_list"),
        GetTensorIdx(tensorMap, "intermediate_weight_idx")};
    gatingNode.inTensorReshapeFuncs.resize(gatingNode.inTensorIds.size());
    gatingNode.inTensorReshapeFuncs[0] = [batchDimPtr](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 1;
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
    };
    ATB_SPEED_LOG_DEBUG("Gating calculation success");
    return atb::NO_ERROR;
}

atb::Status CreateGather0(std::map<std::string, uint32_t> &tensorMap, atb::GraphParam &opGraph, size_t &nodeId)
{
    atb::Node &gatherNode0 = opGraph.nodes.at(nodeId++);
    atb::infer::GatherParam gatherParam;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(gatherParam, &gatherNode0.operation));
    gatherNode0.inTensorIds = {GetTensorIdx(tensorMap, "in_hiddenstates"), GetTensorIdx(tensorMap, "intermediate_idx")};
    gatherNode0.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_sorted_hiddenstates")};
    ATB_SPEED_LOG_DEBUG("Gather0 calculation success");
    return atb::NO_ERROR;
}

// Step 2: grouped matmul calculation & activation
atb::Status CreateGmm(std::map<std::string, uint32_t> &tensorMap,
    atb::GraphParam &opGraph, const MoeMlpParam &param, size_t &nodeId)
{
    atb::Node &gmmNode = opGraph.nodes.at(nodeId++);
    atb_speed::common::IntegratedGmmParam gmmParam;
    gmmParam.hasBias = (param.packQuantType == atb_speed::common::PackQuantType::ALL_W4A8) ? \
        true : param.hasBias;
    gmmParam.isUp = true;
    gmmParam.moeLinearQuantType = param.moeLinearQuantType;
    gmmParam.packQuantType = param.packQuantType;
    gmmParam.transposeB = param.gateUpTransposeB;
    gmmParam.downTransposeB = param.downTransposeB;
    gmmParam.outDataType = param.isBF16 ? ACL_BF16 : ACL_FLOAT16;
    gmmParam.quantGroupSize = param.quantGroupSize;
    gmmParam.enableGMMSwigluQuant = param.enableGMMSwigluQuant;
    gmmParam.enableAtlasGMMFused = param.enableAtlasGMMFused;
    if (param.enableInitQuant || (param.enableMoeDistribute &&
        (param.packQuantType == atb_speed::common::PackQuantType::ALL_W8A8_DYNAMIC \
            || param.packQuantType == atb_speed::common::PackQuantType::ALL_W4A8))) {
        gmmParam.skipQuant = true;
    }
    if (param.enableCVOverlap) {gmmParam.enableCVOverlap = true;}
    CHECK_OPERATION_STATUS_RETURN(CreateIntegratedGmmOperation(gmmParam, &gmmNode.operation));
    gmmNode.inTensorIds = {};
    gmmNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "intermediate_sorted_hiddenstates"));
    gmmNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_mlp_gateup_weight_expert"));
    gmmNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_mlp_gateup_bias_expert"));
    gmmNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_mlp_gateup_descale_expert"));
    gmmNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_mlp_gateup_offset_expert"));
    gmmNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_mlp_gateup_scale_expert"));
    gmmNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_mlp_gateup_compress_idx_expert"));
    if (param.enableFusedRouting && !param.enableMoeDistribute) {
        gmmNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "intermediate_group_list_int64"));
    } else {
        gmmNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "intermediate_group_list"));
    }
    bool isGMMSwigluQuant = IsGMMSwigluQuant(CalcUpGmmQuantType(param), param);
    if (isGMMSwigluQuant) {
        gmmNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_mlp_down_weight_expert"));
        gmmNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_mlp_down_bias_expert"));
        gmmNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_mlp_down_descale_expert"));
        gmmNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_mlp_down_offset_expert"));
        gmmNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_mlp_down_scale_expert"));
        gmmNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_mlp_down_compress_idx_expert"));
    }
    gmmNode.outTensorIds = {GetTensorIdx(tensorMap, isGMMSwigluQuant ? "intermediate_mlp_out" :
        "intermediate_matmul_gate_up_out")};
    if (gmmParam.skipQuant) {
        gmmNode.inTensorIds.push_back(GetTensorIdx(tensorMap, param.enableMoeDistribute ? "intermediate_gmm0_deqscale"
            : "intermediate_sorted_hiddenstates_dequant_scale"));
    }
    ATB_SPEED_LOG_DEBUG("GMM calculation success");
    return atb::NO_ERROR;
}

atb::Status CreateActivation(
    std::map<std::string, uint32_t> &tensorMap, atb::GraphParam &opGraph, size_t &nodeId)
{
    atb::Node &swishNode = opGraph.nodes.at(nodeId++);
    atb::infer::ActivationParam activationParam;
    activationParam.activationType = atb::infer::ActivationType::ACTIVATION_SWIGLU_FORWARD;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(activationParam, &swishNode.operation));
    swishNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_matmul_gate_up_out")};
    swishNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_swish_out")};
    ATB_SPEED_LOG_DEBUG("Activation calculation success");
    return atb::NO_ERROR;
}

atb::Status CreateActivationQuant(
    std::map<std::string, uint32_t> &tensorMap, atb::GraphParam &opGraph, size_t &nodeId)
{
    atb::Node &swigluQuantNode = opGraph.nodes.at(nodeId++);
    atb::infer::SwigluQuantParam swigluQuantParam;
    swigluQuantParam.quantType = atb::infer::SwigluQuantParam::QuantType::QUANT_TYPE_PER_TOKEN;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(swigluQuantParam, &swigluQuantNode.operation));
    swigluQuantNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_matmul_gate_up_out")};
    swigluQuantNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_swish_out"),
                                    GetTensorIdx(tensorMap, "intermedaite_swiglu_dequant_scale")};
    ATB_SPEED_LOG_DEBUG("ActivationQuant calculation success");
    return atb::NO_ERROR;
}

atb::Status CreateSplit(std::map<std::string, uint32_t> &tensorMap, atb::GraphParam &opGraph, size_t &nodeId)
{
    atb::Node &splitNode = opGraph.nodes.at(nodeId++);
    atb::infer::SplitParam splitParam = {1, 2};
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(splitParam, &splitNode.operation));
    splitNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_matmul_gate_up_out")};
    splitNode.outTensorIds = {
        GetTensorIdx(tensorMap, "intermediate_matmul_gate_out"), GetTensorIdx(tensorMap, "intermediate_matmul_up_out")};
    ATB_SPEED_LOG_DEBUG("Split calculation success");
    return atb::NO_ERROR;
}

atb::Status CreateActivationO(std::map<std::string, uint32_t> &tensorMap, atb::GraphParam &opGraph, size_t &nodeId)
{
    atb::Node &swishNodeO = opGraph.nodes.at(nodeId++);
    atb::infer::ActivationParam activationParam;
    activationParam.activationType = atb::infer::ActivationType::ACTIVATION_SWISH;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(activationParam, &swishNodeO.operation));
    swishNodeO.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_matmul_gate_out")};
    swishNodeO.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_swish_out_internal")};
    ATB_SPEED_LOG_DEBUG("Activation calculation success");
    return atb::NO_ERROR;
}

atb::Status CreateElewiseMul(std::map<std::string, uint32_t> &tensorMap, atb::GraphParam &opGraph, size_t &nodeId)
{
    atb::Node &mulNode = opGraph.nodes.at(nodeId++);
    atb::infer::ElewiseParam elewiseParam;
    elewiseParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(elewiseParam, &mulNode.operation));
    mulNode.inTensorIds = {
        GetTensorIdx(tensorMap, "intermediate_swish_out_internal"),
        GetTensorIdx(tensorMap, "intermediate_matmul_up_out")};
    mulNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_swish_out")};
    ATB_SPEED_LOG_DEBUG("ElewiseMul0 calculation success");
    return atb::NO_ERROR;
}

atb::Status CreateGmm1(std::map<std::string, uint32_t> &tensorMap,
    atb::GraphParam &opGraph, const MoeMlpParam &param, size_t &nodeId)
{
    atb::Node &gmmDownNode = opGraph.nodes.at(nodeId++);
    atb_speed::common::IntegratedGmmParam gmmParam;
    gmmParam.hasBias = param.hasBias;
    gmmParam.isUp = false;
    gmmParam.moeLinearQuantType = param.moeLinearQuantType;
    gmmParam.packQuantType = param.packQuantType;
    gmmParam.transposeB = param.downTransposeB;
    if (param.packQuantType == atb_speed::common::PackQuantType::ALL_W4A8) {
        gmmParam.hasBias = true;
    }
    gmmParam.outDataType = param.isBF16 ? ACL_BF16 : ACL_FLOAT16;
    if (param.enableSwigluQuant) {gmmParam.skipQuant = true;}
    CHECK_OPERATION_STATUS_RETURN(CreateIntegratedGmmOperation(gmmParam, &gmmDownNode.operation));
    gmmDownNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_mlp_out")};
    gmmDownNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_swish_out"),
                               GetTensorIdx(tensorMap, "in_mlp_down_weight_expert"),
                               GetTensorIdx(tensorMap, "in_mlp_down_bias_expert"),
                               GetTensorIdx(tensorMap, "in_mlp_down_descale_expert"),
                               GetTensorIdx(tensorMap, "in_mlp_down_offset_expert"),
                               GetTensorIdx(tensorMap, "in_mlp_down_scale_expert"),
                               GetTensorIdx(tensorMap, "in_mlp_down_compress_idx_expert")};
    if (param.enableFusedRouting && !param.enableMoeDistribute) {
        gmmDownNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "intermediate_group_list_int64"));
    } else {
        gmmDownNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "intermediate_group_list"));
    }
    if (param.enableSwigluQuant) {
        gmmDownNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "intermedaite_swiglu_dequant_scale"));
    }

    ATB_SPEED_LOG_DEBUG("GMM calculation success");
    return atb::NO_ERROR;
}

// Step 3: hidden state reduction
atb::Status CreateMoeTokenUnpermute(
    std::map<std::string, uint32_t> &tensorMap, const MoeMlpParam &param, atb::GraphParam &opGraph,  size_t &nodeId)
{
    atb::Node &unpermuteNode = opGraph.nodes.at(nodeId++);
    unpermuteNode.operation = new atb_speed::common::MoeTokenUnpermuteOperation("MoeTokenUnpermuteNode");
    unpermuteNode.outTensorIds = {GetTensorIdx(tensorMap, "out_moe_mlp_result")};
    unpermuteNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_mlp_out"),
                                 GetTensorIdx(tensorMap, "intermediate_idx"),
                                // shiftedTopK原地写
                                 GetTensorIdx(tensorMap, param.hasMoeEp && !param.shiftedTopK ? \
                                 "intermediate_expert_weight" : (param.enableGatingDp) ? \
                                 "intermediate_expert_weight" : "in_expert_weight")};
    ATB_SPEED_LOG_DEBUG("UnpermuteNode calculation success");
    return atb::NO_ERROR;
}

atb::Status CreateMoeDistributeDispatch(
    std::map<std::string, uint32_t> &tensorMap, const MoeMlpParam &param,
    atb::GraphParam &opGraph, size_t &nodeId)
{
    atb::Node &moeDistributeDispatchNode = opGraph.nodes.at(nodeId++);
    atb_speed::common::MoeDistributeDispatchParam dispatchParam;
    dispatchParam.epRankId = param.moeEpRank;
    dispatchParam.epRankSize = param.moeEpSize;
    dispatchParam.epCommName = param.moeEpDomain;
    dispatchParam.maxDecodeDpTokenSize = param.maxDecodeDpTokenSize;
    dispatchParam.moeExpertNum = param.numOfExperts;
    dispatchParam.localMoeExpertNum = param.numOfDeviceExperts;
    dispatchParam.sharedExpertRankNum = \
        static_cast<int64_t>(std::max(param.moeEpSize - static_cast<int>(param.numOfExperts) + \
            static_cast<int>(param.numOfRedundantExpert), 0));
    dispatchParam.topk = param.topk;

    if (param.packQuantType == atb_speed::common::PackQuantType::ALL_W8A8_DYNAMIC) {
        dispatchParam.quantMode = 2; // 2: 量化模式2
        dispatchParam.isQuant = true;
        dispatchParam.quantSmooth = false;
    } else {
        dispatchParam.quantMode = 0; // 0不量化
        dispatchParam.isQuant = false;
        dispatchParam.quantSmooth = false;
    }

    moeDistributeDispatchNode.operation = new atb_speed::common::MoeDistributeDispatchOperation(
                std::string("MoeDistributeDispatch") + std::to_string(param.packQuantType), dispatchParam);
    moeDistributeDispatchNode.inTensorIds = {GetTensorIdx(tensorMap, "in_hiddenstates"),
                                   GetTensorIdx(tensorMap, "in_selected_experts"),
                                   GetTensorIdx(tensorMap, "in_expert_weight"),
                                   GetTensorIdx(tensorMap, "in_padding_idx")};

    moeDistributeDispatchNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_sorted_hiddenstates"),
                                    GetTensorIdx(tensorMap, "intermediate_gmm0_deqscale"),
                                    GetTensorIdx(tensorMap, "intermediate_idx"),
                                    GetTensorIdx(tensorMap, "intermediate_group_list"),
                                    GetTensorIdx(tensorMap, "intermediate_ep_recv_counts"),
                                    GetTensorIdx(tensorMap, "intermediate_tp_recv_counts"),
                                    GetTensorIdx(tensorMap, "intermediate_expand_expert_weight"),
                                    };
    ATB_SPEED_LOG_DEBUG("MoeDistributeDispatch calculation success");
    return atb::NO_ERROR;
}

atb::Status CreateMoeDistributeCombine(
    std::map<std::string, uint32_t> &tensorMap, const MoeMlpParam &param,
    atb::GraphParam &opGraph, size_t &nodeId)
{
    atb::Node &moeDistributeCombineNode = opGraph.nodes.at(nodeId++);
    atb_speed::common::MoeDistributeCombineParam combineParam;
    combineParam.epRankId = param.moeEpRank;
    combineParam.epRankSize = param.moeEpSize;
    combineParam.epCommName = param.moeEpDomain;
    combineParam.moeExpertNum = param.numOfExperts;
    combineParam.localMoeExpertNum = param.numOfDeviceExperts;
    combineParam.maxDecodeDpTokenSize = param.maxDecodeDpTokenSize;
    combineParam.sharedExpertRankNum = \
        static_cast<int64_t>(std::max(param.moeEpSize - static_cast<int>(param.numOfExperts) + \
            static_cast<int>(param.numOfRedundantExpert), 0));
    combineParam.topk = param.topk;
    moeDistributeCombineNode.operation = new atb_speed::common::MoeDistributeCombineOperation("MoeDistributeCombine",
                                                                               combineParam);

    moeDistributeCombineNode.outTensorIds = {GetTensorIdx(tensorMap, "out_moe_mlp_result")};
    moeDistributeCombineNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_mlp_out"),
                                 GetTensorIdx(tensorMap, "in_selected_experts"),
                                 GetTensorIdx(tensorMap, "intermediate_idx"),
                                 GetTensorIdx(tensorMap, "intermediate_ep_recv_counts"),
                                 GetTensorIdx(tensorMap, "in_expert_weight"),
                                 GetTensorIdx(tensorMap, "intermediate_tp_recv_counts"),
                                 GetTensorIdx(tensorMap, "intermediate_expand_expert_weight"),
                                 };

    ATB_SPEED_LOG_DEBUG("MoeDistributeDispatch calculation success");
    return atb::NO_ERROR;
}

// Op5 - Gather1
atb::Status CreateGather1(std::map<std::string, uint32_t> &tensorMap,
    std::shared_ptr<int64_t> batchDimPtr, atb::GraphParam &opGraph, size_t &nodeId)
{
    atb::Node &gatherNode1 = opGraph.nodes.at(nodeId++);
    atb::infer::GatherParam gatherParam;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(gatherParam, &gatherNode1.operation));
    gatherNode1.inTensorIds = {GetTensorIdx(tensorMap, "in_expert_weight"),
                               GetTensorIdx(tensorMap, "intermediate_weight_idx")};
    gatherNode1.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_sorted_weight")};
    gatherNode1.inTensorReshapeFuncs.resize(gatherNode1.inTensorIds.size());
    gatherNode1.inTensorReshapeFuncs[0] = [batchDimPtr](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 1;
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
    };
    ATB_SPEED_LOG_DEBUG("Gather1 calculation success");
    return atb::NO_ERROR;
}

// Op6 - ElewiseMul1
atb::Status CreateElewiseMul1(std::map<std::string, uint32_t> &tensorMap,
    std::shared_ptr<int64_t> batchDimPtr, atb::GraphParam &opGraph, size_t &nodeId)
{
    atb::Node &weightMulNode = opGraph.nodes.at(nodeId++);
    atb::infer::ElewiseParam elewiseParam;
    elewiseParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(elewiseParam, &weightMulNode.operation));
    weightMulNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_mlp_out"),
                                 GetTensorIdx(tensorMap, "intermediate_sorted_weight")};
    weightMulNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_mlp_out_weighted")};
    weightMulNode.inTensorReshapeFuncs.resize(weightMulNode.inTensorIds.size());
    weightMulNode.inTensorReshapeFuncs[1] = [batchDimPtr](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2; // 2:number of dimensions of the new shape
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = 1;
    };
    ATB_SPEED_LOG_DEBUG("ElewiseMul1 calculation success");
    return atb::NO_ERROR;
}

// Op7 - Argsort
atb::Status CreateArgsort(std::map<std::string, uint32_t> &tensorMap, atb::GraphParam &opGraph, size_t &nodeId)
{
    atb::Node &argsortNode = opGraph.nodes.at(nodeId++);
    atb::infer::GatingParam gatingParam;
    gatingParam.topkExpertNum = 1;
    gatingParam.cumSumNum = 0;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(gatingParam, &argsortNode.operation));
    argsortNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_weight_idx"),
                               GetTensorIdx(tensorMap, "in_expert_array")};
    argsortNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_dummy_zero"),
                                GetTensorIdx(tensorMap, "intermediate_dummy_one"),
                                GetTensorIdx(tensorMap, "intermediate_rev_idx")};
    ATB_SPEED_LOG_DEBUG("Argsort calculation success");
    return atb::NO_ERROR;
}

// Op8 - Gather2
atb::Status CreateGather2(std::map<std::string, uint32_t> &tensorMap, atb::GraphParam &opGraph, size_t &nodeId)
{
    atb::Node &gatherNode2 = opGraph.nodes.at(nodeId++);
    atb::infer::GatherParam gatherParam;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(gatherParam, &gatherNode2.operation));
    gatherNode2.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_mlp_out_weighted"),
                               GetTensorIdx(tensorMap, "intermediate_rev_idx")};
    gatherNode2.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_rev_sorted_hiddenstates")};
    ATB_SPEED_LOG_DEBUG("Cather2 calculation success");
    return atb::NO_ERROR;
}

// Op9 - Reduction
atb::Status CreateReduction(std::map<std::string, uint32_t> &tensorMap,
    const MoeMlpParam &param, std::shared_ptr<int64_t> batchDimPtr, atb::GraphParam &opGraph, size_t &nodeId)
{
    CHECK_PARAM_NE(param.topk, 0);
    atb::Node &reduceNode = opGraph.nodes.at(nodeId++);
    atb::infer::ReduceParam reduceParam;
    reduceParam.reduceType = atb::infer::ReduceParam::ReduceType::REDUCE_SUM;
    reduceParam.axis = {1};
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(reduceParam, &reduceNode.operation));
    reduceNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_rev_sorted_hiddenstates")};
    reduceNode.outTensorIds = {GetTensorIdx(tensorMap, "out_moe_mlp_result")};
    reduceNode.inTensorReshapeFuncs.resize(reduceNode.inTensorIds.size());
    reduceNode.inTensorReshapeFuncs[0] = [batchDimPtr, param](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 3; // 3:number of dimensions of the new shape
        newShape.dims[0] = oldShape.dims[0] / param.topk;
        newShape.dims[1] = param.topk;
        newShape.dims[2] = oldShape.dims[1]; // 2:the third dimension of the new shape
    };
    ATB_SPEED_LOG_DEBUG("Reduction calculation success");
    return atb::NO_ERROR;
}

int64_t CalculateNodeSize(const MoeMlpParam &param)
{
    int64_t nodeCount = 10;
    if (!param.supportSwiGLU) {
        nodeCount += NUM2;
    }
    if (param.enableMoeDistribute) {
        nodeCount -= NUM5;
    } else {
        if (param.enableFusedRouting) {
            nodeCount -= NUM4;
        }
        if (param.hasMoeEp) {
            nodeCount += NUM2;
        }
    }
    bool isGMMSwigluQuant = IsGMMSwigluQuant(CalcUpGmmQuantType(param), param);
    nodeCount -= isGMMSwigluQuant ? 2 : 0; // 2: activationNode, gmm1Node
    if (param.enableCVOverlap) {
        nodeCount += NUM4;
    }
    return nodeCount;
}

atb::Status CreateActivationBlock(std::map<std::string, uint32_t> &tensorMap,
    const MoeMlpParam &param, size_t &nodeId, atb::GraphParam &opGraph)
{
    if (param.supportSwiGLU) {
        if (param.enableSwigluQuant) {
            CHECK_OPERATION_STATUS_RETURN(CreateActivationQuant(tensorMap, opGraph, nodeId));
        } else {
            CHECK_OPERATION_STATUS_RETURN(CreateActivation(tensorMap, opGraph, nodeId));
        }
    } else {
        CHECK_OPERATION_STATUS_RETURN(CreateSplit(tensorMap, opGraph, nodeId));
        CHECK_OPERATION_STATUS_RETURN(CreateActivationO(tensorMap, opGraph, nodeId));
        CHECK_OPERATION_STATUS_RETURN(CreateElewiseMul(tensorMap, opGraph, nodeId));
    }

    ATB_SPEED_LOG_DEBUG("ActivationBlock calculation success");
    return atb::NO_ERROR;
}

atb::Status CreateRecord(const MoeMlpParam &param, atb::GraphParam &opGraph, size_t &nodeId,
                         atb_speed::EventAction eventAction, const std::string &cvKey)
{
    if (param.enableCVOverlap) {
        atb::Node &recordNode = opGraph.nodes.at(nodeId++);
        recordNode.inTensorIds = {};
        recordNode.outTensorIds = {};
        CHECK_OPERATION_STATUS_RETURN(atb_speed::EventManager::GetInstance().RecordEvent(
            recordNode.operation,
            eventAction,
            cvKey));
        ATB_SPEED_LOG_DEBUG("Record event success");
    }
    return atb::NO_ERROR;
}

atb::Status CreateWait(const MoeMlpParam &param, atb::GraphParam &opGraph, size_t &nodeId,
                       atb_speed::EventAction eventAction, const std::string &cvKey)
{
    if (param.enableCVOverlap) {
        atb::Node &waitNode = opGraph.nodes.at(nodeId++);
        waitNode.inTensorIds = {};
        waitNode.outTensorIds = {};
        CHECK_OPERATION_STATUS_RETURN(atb_speed::EventManager::GetInstance().WaitEvent(
            waitNode.operation,
            eventAction,
            cvKey));
        ATB_SPEED_LOG_DEBUG("Wait event success");
    }
    return atb::NO_ERROR;
}

atb::Status CreateMoeMlpOperation(const MoeMlpParam &param, atb::Operation **operation)
{
    std::shared_ptr<int64_t> batchDimPtr = std::make_shared<int64_t>(0);
    atb::GraphParam opGraph;
    opGraph.name = "MoeMlp";
    bool isGMMSwigluQuant = IsGMMSwigluQuant(CalcUpGmmQuantType(param), param);
    std::map<std::string, uint32_t> tensorMap = ConstructTensorMap(
        param, opGraph.inTensorNum, opGraph.outTensorNum, opGraph.internalTensorNum);
    ATB_SPEED_LOG_DEBUG("opGraph.inTensorNum " << opGraph.inTensorNum);
    ATB_SPEED_LOG_DEBUG("opGraph.outTensorNum " << opGraph.outTensorNum);
    ATB_SPEED_LOG_DEBUG("opGraph.internalTensorNum" << opGraph.internalTensorNum);

    uint64_t nodeCount = CalculateNodeSize(param);
    opGraph.nodes.resize(nodeCount);
    size_t nodeId = 0;
    if (param.hasMoeEp && !param.enableMoeDistribute) {
        CHECK_OPERATION_STATUS_RETURN(CreateExpertFilter(tensorMap, param, opGraph, nodeId));
    }

    if (param.enableMoeDistribute) {
        CHECK_OPERATION_STATUS_RETURN(CreateMoeDistributeDispatch(tensorMap, param, opGraph, nodeId));
        CHECK_OPERATION_STATUS_RETURN(CreateGmm(tensorMap, opGraph, param, nodeId));
        if (!isGMMSwigluQuant) {
            CHECK_OPERATION_STATUS_RETURN(CreateActivationBlock(tensorMap, param, nodeId, opGraph));
            CHECK_OPERATION_STATUS_RETURN(CreateGmm1(tensorMap, opGraph, param, nodeId));
        }
        CHECK_OPERATION_STATUS_RETURN(CreateMoeDistributeCombine(tensorMap, param, opGraph, nodeId));
    } else if (param.enableFusedRouting) {
        if (param.enableCVOverlap) {
            CHECK_OPERATION_STATUS_RETURN(CreateWait(
                param, opGraph, nodeId, atb_speed::EventAction::POP, atb_speed::common::CUBE_CONTROL));
        }
        if (param.enableInitQuant) {
            CHECK_OPERATION_STATUS_RETURN(CreateInitRoutingQuant(tensorMap, param, opGraph, nodeId));
        } else {
            CHECK_OPERATION_STATUS_RETURN(CreateInitRouting(tensorMap, param, opGraph, nodeId));
        }
        if (param.hasMoeEp) {
            CHECK_OPERATION_STATUS_RETURN(CreateComputeExpertSlice(tensorMap, param, opGraph, nodeId));
        }
        CHECK_OPERATION_STATUS_RETURN(CreateCast(tensorMap, param, opGraph, nodeId));
        CHECK_OPERATION_STATUS_RETURN(CreateGmm(tensorMap, opGraph, param, nodeId));
        if (param.enableCVOverlap) {
            CHECK_OPERATION_STATUS_RETURN(CreateWait(
                param, opGraph, nodeId, atb_speed::EventAction::POP, atb_speed::common::VECTOR_CONTROL));
            CHECK_OPERATION_STATUS_RETURN(CreateRecord(
                param, opGraph, nodeId, atb_speed::EventAction::POP, atb_speed::common::CUBE_CONTROL));
        }
        if (!isGMMSwigluQuant) {
            CHECK_OPERATION_STATUS_RETURN(CreateActivationBlock(tensorMap, param, nodeId, opGraph));
            CHECK_OPERATION_STATUS_RETURN(CreateGmm1(tensorMap, opGraph, param, nodeId));
        }
        CHECK_OPERATION_STATUS_RETURN(CreateMoeTokenUnpermute(tensorMap, param, opGraph, nodeId));

        if (param.enableCVOverlap) {
            CHECK_OPERATION_STATUS_RETURN(CreateWait(
                param, opGraph, nodeId, atb_speed::EventAction::POP, atb_speed::common::CUBE_CONTROL));
        }
    } else {
        CHECK_OPERATION_STATUS_RETURN(CreateGating(tensorMap, param, batchDimPtr, opGraph, nodeId));
        CHECK_OPERATION_STATUS_RETURN(CreateGather0(tensorMap, opGraph, nodeId));
        CHECK_OPERATION_STATUS_RETURN(CreateGmm(tensorMap, opGraph, param, nodeId));
        if (!isGMMSwigluQuant) {
            CHECK_OPERATION_STATUS_RETURN(CreateActivationBlock(tensorMap, param, nodeId, opGraph));
            CHECK_OPERATION_STATUS_RETURN(CreateGmm1(tensorMap, opGraph, param, nodeId));
        }
        CHECK_OPERATION_STATUS_RETURN(CreateGather1(tensorMap, batchDimPtr, opGraph, nodeId));
        CHECK_OPERATION_STATUS_RETURN(CreateElewiseMul1(tensorMap, batchDimPtr, opGraph, nodeId));
        CHECK_OPERATION_STATUS_RETURN(CreateArgsort(tensorMap, opGraph, nodeId));
        CHECK_OPERATION_STATUS_RETURN(CreateGather2(tensorMap, opGraph, nodeId));
        CHECK_OPERATION_STATUS_RETURN(CreateReduction(tensorMap, param, batchDimPtr, opGraph, nodeId));
    }
    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                    atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        if (param.enableExpertCumSumOutput) {
            outTensorDescs.at(1) = atb::TensorDesc{};
            outTensorDescs.at(1).format = ACL_FORMAT_ND;
            outTensorDescs.at(1).shape.dimNum = 1;
            outTensorDescs.at(1).dtype = ACL_INT64;
            if (param.enableMoeDistribute) {
                outTensorDescs.at(1).shape.dims[0] = param.numOfDeviceExperts;
            } else if (param.enableFusedRouting) {
                if (param.hasMoeEp) {
                    outTensorDescs.at(1).shape.dims[0] = param.numOfDeviceExperts;
                } else {
                    outTensorDescs.at(1).shape.dims[0] = param.numOfExperts;
                }
            } else {
                outTensorDescs.at(1).shape.dims[0] = param.numOfExperts;
            }
        }

        return atb::NO_ERROR;
    };
    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
}
}
} // namespace atb_speed