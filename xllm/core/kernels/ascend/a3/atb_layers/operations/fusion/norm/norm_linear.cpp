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
#include "atb_speed/log.h"
#include "operations/aclnn/ops/add_rms_norm_quant_operation.h"
#include "operations/aclnn/ops/add_rms_norm_dynamic_quant_operation.h"
#include "operations/aclnn/ops/rms_norm_operation.h"
#include "operations/aclnn/utils/utils.h"
#include "operations/fusion/utils.h"
#include "operations/fusion/norm/norm_linear.h"

namespace atb_speed {
namespace common {

template <typename NormParamType>
bool UseNormQuant(const NormLinearParam<NormParamType> &param)
{
    if (param.fusionLinearParam.quantType == LinearQuantType::LINEAR_W8A8_DEQUANT || \
        param.fusionLinearParam.quantType == LinearQuantType::LINEAR_W8A8_SC_DEQUANT || \
        (param.fusionLinearParam.quantType == LinearQuantType::LINEAR_W8A8_DYNAMIC_DEQUANT && \
         param.enableAddNorm)) {
        return true;
    } else {
        return false;
    }
}

template <typename NormParamType>
bool UseAddRmsNormQuant(const NormLinearParam<NormParamType> &param)
{
    // 算子支持310P、A2、A3机型
    return (Is310P() || IsA2()) && param.enableAddNorm &&
        param.fusionLinearParam.quantType == LinearQuantType::LINEAR_W8A8_DEQUANT;
}

template <typename NormParamType>
bool UseAddRmsNormDynamicQuant(const NormLinearParam<NormParamType> &param)
{
    // 算子支持A2、A3机型
    return IsA2() && param.enableAddNorm &&
        param.fusionLinearParam.quantType == LinearQuantType::LINEAR_W8A8_DYNAMIC_DEQUANT;
}

std::map<std::string, std::vector<std::string>> GetInTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> normInTensorCandidates = {
        {"default", {
            "in_input", "in_norm_weight", "in_norm_bias", "in_norm_new_weight", "in_norm_new_bias",
            "in_linear_weight", "in_scale", "in_offset", "in_descale", "in_bias", "in_compress_idx"}
        },
        {"add_norm", {"in_residual_input"}},
        {"add_rmsnorm_quant", {"in_scale_fill", "in_offset_fill"}},
        {"lora", {"in_seq_len_cum_sum", "in_linear_lora_a", "in_linear_lora_b"}},
        {"lora_with_mask", {"in_im_mask"}}
    };
    return normInTensorCandidates;
}

std::map<std::string, std::vector<std::string>> GetIntermediateTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> normIntermediateTensorCandidates = {
        {"default", {"intermediate_norm"}},
        {"addrmsnormquant", {"y2_out"}},
        {"addrmsnormdynamicquant", {"y2_out", "scale1_out", "scale2_out"}},
        {"useless", {"intermediate_rstd"}}
    };
    return normIntermediateTensorCandidates;
}

std::map<std::string, std::vector<std::string>> GetOutTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> normOutTensorCandidates = {
        {"default", {"out_linear"}},
        {"add_norm", {"out_add"}},
    };
    return normOutTensorCandidates;
}

template <typename NormParamType>
std::map<std::string, uint32_t> ConstructNormTensorMap(
    const NormLinearParam<NormParamType> &param,
    uint32_t &inTensorNum, uint32_t &outTensorNum, uint32_t &internalTensorNum)
{
    auto normInTensorCandidates = GetInTensorCandidates();
    auto normIntermediateTensorCandidates = GetIntermediateTensorCandidates();
    auto normOutTensorCandidates = GetOutTensorCandidates();

    std::vector<std::string> inTensorList = {};
    std::vector<std::string> intermediateTensorList = {};
    std::vector<std::string> outTensorList = {};

    // 添加默认的Tensor
    AddTensorToList(normInTensorCandidates, "default", inTensorList);
    if (!param.skipNorm) {
        AddTensorToList(normIntermediateTensorCandidates, "default", intermediateTensorList);
    }

    // 添加add norm特性的Tensor、添加AddRmsNormQuant特性的Tensor
    if (param.enableAddNorm) {
        AddTensorToList(normInTensorCandidates, "add_rmsnorm_quant", inTensorList);
        AddTensorToList(normInTensorCandidates, "add_norm", inTensorList);
    }

    // 添加lora特性的Tensor
    if (param.fusionLinearParam.supportLora) {
        if (param.fusionLinearParam.useImMask) {
            AddTensorToList(normInTensorCandidates, "lora_with_mask", inTensorList);
        }
        AddTensorToList(normInTensorCandidates, "lora", inTensorList);
    }

    // 添加outTensor
    AddTensorToList(normOutTensorCandidates, "default", outTensorList);
    if (param.enableAddNorm) {
        AddTensorToList(normOutTensorCandidates, "add_norm", outTensorList);
    }
    if (UseAddRmsNormQuant(param)) {
        AddTensorToList(normIntermediateTensorCandidates, "addrmsnormquant", intermediateTensorList);
    }
    if (UseAddRmsNormDynamicQuant(param)) {
        AddTensorToList(normIntermediateTensorCandidates, "addrmsnormdynamicquant", intermediateTensorList);
    }
    if (param.enableAclnnRmsNorm) {
        AddTensorToList(normIntermediateTensorCandidates, "useless", intermediateTensorList);
    }
    inTensorNum = inTensorList.size();
    outTensorNum = outTensorList.size();
    internalTensorNum = intermediateTensorList.size();

    return GetTensorMap(inTensorList, outTensorList, intermediateTensorList);
}

template <typename NormParamType>
int64_t InsertNorm(
    atb::GraphParam &opGraph,
    const NormLinearParam<NormParamType> &param,
    std::map<std::string, uint32_t> &tensorMap)
{
    bool useNormQuant = UseNormQuant(param);
    bool useAddRmsNormQuant = UseAddRmsNormQuant(param);
    bool useAddRmsNormDynamicQuant = UseAddRmsNormDynamicQuant(param);
    atb::Node normNode;
    if (param.enableAddNorm) {
        normNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_residual_input"));
    }
    normNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_norm")};
    if (param.enableAclnnRmsNorm) {
        normNode.outTensorIds.push_back(GetTensorIdx(tensorMap, "intermediate_rstd"));
    }
    if (useAddRmsNormQuant || useAddRmsNormDynamicQuant) {
        normNode.outTensorIds.push_back(GetTensorIdx(tensorMap, "y2_out"));
        normNode.outTensorIds.push_back(GetTensorIdx(tensorMap, "out_add"));
        if (useAddRmsNormDynamicQuant) {
            normNode.outTensorIds.push_back(GetTensorIdx(tensorMap, "scale1_out"));
            normNode.outTensorIds.push_back(GetTensorIdx(tensorMap, "scale2_out"));
        }
    } else if (param.enableAddNorm) {
        normNode.outTensorIds.push_back(GetTensorIdx(tensorMap, "out_add"));
    }
    if (useNormQuant) {  // activation quant
        normNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_input"));
        normNode.inTensorIds.push_back(param.isAntiOutlier ? \
            GetTensorIdx(tensorMap, "in_norm_new_weight") : GetTensorIdx(tensorMap, "in_norm_weight"));
        if (!useAddRmsNormQuant && !useAddRmsNormDynamicQuant) {
            normNode.inTensorIds.push_back(param.isAntiOutlier ? \
                GetTensorIdx(tensorMap, "in_norm_new_bias") : GetTensorIdx(tensorMap, "in_norm_bias"));
        }
        if (useAddRmsNormQuant) {
            normNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_scale_fill"));
            normNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_offset_fill"));
        } else if (!useAddRmsNormDynamicQuant) {
            normNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_scale"));
            normNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_offset"));
        }
        if (useAddRmsNormQuant) { // aclnn接口
            normNode.operation = new atb_speed::common::AddRmsNormQuantOperation(
                "AddRmsNormQuantOperation", param.normParamType.normParam.epsilon);
        } else if (useAddRmsNormDynamicQuant) { // aclnn接口
            normNode.operation = new atb_speed::common::AddRmsNormDynamicQuantOperation(
                "AddRmsNormDynamicQuantOperation", param.normParamType.normParam.epsilon);
        } else { // ATB接口
            CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(param.normQuantParamType, &normNode.operation));
        }
    } else {  // activation no quant
        if (param.enableAclnnRmsNorm) {
            normNode.operation = new atb_speed::common::RmsNormOperation("RmsNorm", 1e-5);
        } else {
            CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(param.normParamType, &normNode.operation));
        }
        normNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_input"));
        normNode.inTensorIds.push_back(param.isAntiOutlier ? \
            GetTensorIdx(tensorMap, "in_norm_new_weight") : GetTensorIdx(tensorMap, "in_norm_weight"));
        if (param.normHasBias) {
            normNode.inTensorIds.push_back(param.isAntiOutlier ? \
                GetTensorIdx(tensorMap, "in_norm_new_bias") : GetTensorIdx(tensorMap, "in_norm_bias"));
        }
    }
    opGraph.nodes.push_back(normNode);
    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status NormLinear(const NormLinearParam<NormParamType> &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.name = "NormLinear";

    std::map<std::string, uint32_t> tensorMap = ConstructNormTensorMap(
        param, opGraph.inTensorNum, opGraph.outTensorNum, opGraph.internalTensorNum);

    // 校验（当前不支持PostNorm）
    bool normIsPostNorm = static_cast<int>(param.normParamType.layerType) == \
        static_cast<int>(atb::infer::RmsNormParam::RmsNormType::RMS_NORM_POSTNORM) || \
        static_cast<int>(param.normParamType.layerType) == \
        static_cast<int>(atb::infer::LayerNormParam::LayerNormType::LAYER_NORM_POSTNORM);
    bool normQuantIsPostNorm = static_cast<int>(param.normQuantParamType.layerType) == \
        static_cast<int>(atb::infer::RmsNormParam::RmsNormType::RMS_NORM_POSTNORM) || \
        static_cast<int>(param.normQuantParamType.layerType) == \
        static_cast<int>(atb::infer::LayerNormParam::LayerNormType::LAYER_NORM_POSTNORM);
    if (normIsPostNorm || (UseNormQuant(param) && normQuantIsPostNorm)) {
        ATB_SPEED_LOG_ERROR("Common Op NormLinear not support POSTNORM");
        return atb::ERROR_INTERNAL_ERROR;
    }

    if (!param.skipNorm) {
        CHECK_OPERATION_STATUS_RETURN(InsertNorm(opGraph, param, tensorMap));
    }

    atb::Node linearNode;
    atb_speed::common::FusionLinearParam linearParam = param.fusionLinearParam;
    // 如果是LINEAR_W8A8_DYNAMIC_DEQUANT且AddRmsNormQuant为false, 则设为LinearQuantType::LINEAR_W8A8_DYNAMIC_QUANT
    if (linearParam.quantType == LinearQuantType::LINEAR_W8A8_DYNAMIC_DEQUANT && !param.enableAddNorm) {
        linearParam.quantType = LinearQuantType::LINEAR_W8A8_DYNAMIC_QUANT;
    }
    CHECK_OPERATION_STATUS_RETURN(FusionLinear(linearParam, &linearNode.operation));
    std::vector<std::string> linearInTensor = {
        param.skipNorm ? "in_input" : "intermediate_norm", "in_linear_weight", "in_scale", "in_offset",
        "in_descale", "in_bias", "in_compress_idx"
    };
    if (UseAddRmsNormDynamicQuant(param)) {
        linearInTensor.push_back("scale1_out");
    }
    if (param.fusionLinearParam.supportLora) {
        if (param.fusionLinearParam.useImMask) {
            linearInTensor.push_back("in_im_mask");
        }
        linearInTensor.push_back("in_seq_len_cum_sum");
        linearInTensor.push_back("in_linear_lora_a");
        linearInTensor.push_back("in_linear_lora_b");
    }
    linearNode.inTensorIds = GetTensorIdxList(tensorMap, linearInTensor);
    linearNode.outTensorIds = {GetTensorIdx(tensorMap, "out_linear")};
    opGraph.nodes.push_back(linearNode);

    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(opGraph, operation));
    return atb::NO_ERROR;
}

LinearQuantType GetLinearQuantType(
    const int &packQuantType, const int &linearType, bool hasNorm, const int &linearDesc
)
{
    if (linearType == atb_speed::common::LinearType::FP || \
        linearDesc == LinearDesc::FLOAT16_DESC || \
        linearDesc == LinearDesc::BFLOAT16_DESC) {
        return atb_speed::common::LinearQuantType::NO_QUANT;
    } else if (packQuantType == atb_speed::common::ALL_W4A16 || \
               packQuantType == atb_speed::common::ALL_W4A16_ANTI || \
               packQuantType == atb_speed::common::MIX_W4A16 || \
               packQuantType == atb_speed::common::MIX_W4A16_ANTI || \
               linearDesc == LinearDesc::W4A16_DESC
            ) {
        return atb_speed::common::LinearQuantType::W4A16;
    } else if (packQuantType == atb_speed::common::ALL_W8A16 || \
               packQuantType == atb_speed::common::ALL_W8A16_ANTI || \
               packQuantType == atb_speed::common::MIX_W8A16 || \
               packQuantType == atb_speed::common::MIX_W8A16_ANTI || \
               linearDesc == LinearDesc::W8A16_DESC
    ) {
        return atb_speed::common::LinearQuantType::W8A16;
    } else if (
        packQuantType == atb_speed::common::ALL_W8A8_DYNAMIC || \
        packQuantType == atb_speed::common::MIX_W8A8_DYNAMIC || \
        linearDesc == LinearDesc::W8A8_DYNAMIC_DESC
    ) {
        return hasNorm ? LinearQuantType::LINEAR_W8A8_DYNAMIC_DEQUANT : LinearQuantType::LINEAR_W8A8_DYNAMIC_QUANT;
    } else if (packQuantType == atb_speed::common::ALL_W4A8 || \
        packQuantType == atb_speed::common::ALL_W4A8_ANTI || \
        packQuantType == atb_speed::common::MIX_W4A8 || \
        packQuantType == atb_speed::common::MIX_W4A8_ANTI || \
        linearDesc == LinearDesc::W4A8_DESC
    ) {
        return atb_speed::common::LinearQuantType::W4A8;
    } else {
        if (packQuantType == atb_speed::common::ALL_W8A8SC || \
            packQuantType == atb_speed::common::MIX_W8A8SC || \
            packQuantType == atb_speed::common::ALL_W8A8SC_ANTI || \
            packQuantType == atb_speed::common::MIX_W8A8SC_ANTI || \
            linearDesc == LinearDesc::W8A8SC_DESC
        ) {
            return hasNorm ? LinearQuantType::LINEAR_W8A8_SC_DEQUANT : LinearQuantType::LINEAR_W8A8_SC_QUANT;
        } else {
            return hasNorm ? LinearQuantType::LINEAR_W8A8_DEQUANT : LinearQuantType::LINEAR_W8A8_QUANT;
        }
    }
}

template bool UseNormQuant(const NormLinearParam<atb::infer::RmsNormParam> &param);
template bool UseAddRmsNormQuant(const NormLinearParam<atb::infer::RmsNormParam> &param);
template bool UseAddRmsNormDynamicQuant(const NormLinearParam<atb::infer::RmsNormParam> &param);
template std::map<std::string, uint32_t> ConstructNormTensorMap(
    const NormLinearParam<atb::infer::RmsNormParam> &param,
    uint32_t &inTensorNum, uint32_t &outTensorNum, uint32_t &internalTensorNum);
template int64_t InsertNorm(
    atb::GraphParam &opGraph, const NormLinearParam<atb::infer::RmsNormParam> &param,
    std::map<std::string, uint32_t> &tensorMap);
template atb::Status NormLinear(const NormLinearParam<atb::infer::RmsNormParam> &param, atb::Operation **operation);

template bool UseNormQuant(const NormLinearParam<atb::infer::LayerNormParam> &param);
template std::map<std::string, uint32_t> ConstructNormTensorMap(
    const NormLinearParam<atb::infer::LayerNormParam> &param,
    uint32_t &inTensorNum, uint32_t &outTensorNum, uint32_t &internalTensorNum);
template int64_t InsertNorm(
    atb::GraphParam &opGraph, const NormLinearParam<atb::infer::LayerNormParam> &param,
    std::map<std::string, uint32_t> &tensorMap);
template atb::Status NormLinear(const NormLinearParam<atb::infer::LayerNormParam> &param, atb::Operation **operation);

} // namespace common
} // namespace atb_speed