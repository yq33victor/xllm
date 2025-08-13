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
#include <set>
#include "nlohmann/json.hpp"
#include "atb/atb_infer.h"
#include "atb_speed/log.h"
#include "models/moe/model/decoder_model.h"

#include <atb/types.h>

namespace atb_speed {
namespace moe {

constexpr size_t MOE_LINEAR_TYPE_LENGTH = 4;

void MoeModelParam::ParseParam(const nlohmann::json &paramJson)
{
    atb_speed::base::ModelParam::ParseParam(paramJson);
    ParseQuantParams(paramJson);
    ParseAttnParallelParams(paramJson);
    ParseParallelParams(paramJson);
    ParseInteOpParams(paramJson);
    CheckParallelParamValid();
    if (paramJson.contains("numOfExperts")) {
        numOfExperts = atb_speed::base::FetchJsonParam<uint32_t>(paramJson, "numOfExperts");
    }
    if (paramJson.contains("numOfDeviceExperts")) {
        numOfDeviceExperts = atb_speed::base::FetchJsonParam<uint32_t>(paramJson, "numOfDeviceExperts");
    }
    if (paramJson.contains("expertParallelDegree")) {
        this->expertParallelDegree = paramJson["expertParallelDegree"].get<int>();
    }
    if (paramJson.contains("routingMethod")) {
        this->routingMethod = paramJson["routingMethod"].get<std::string>();
    }
    if (paramJson.contains("processLogits")) {
        this->processLogits = paramJson["processLogits"].get<std::string>();
    }
    if (paramJson.contains("normHasBias")) {
        this->normHasBias = paramJson["normHasBias"].get<bool>();
    }
    if (paramJson.contains("firstKDenseReplace")) {
        this->firstKDenseReplace = atb_speed::base::FetchJsonParam<int>(paramJson, "firstKDenseReplace");
    }
    if (paramJson.contains("numOfSharedExperts")) {
        this->numOfSharedExperts = atb_speed::base::FetchJsonParam<int>(paramJson, "numOfSharedExperts");
    }
    if (paramJson.contains("hasSharedExpert")) {
        this->hasSharedExpert = atb_speed::base::FetchJsonParam<bool>(paramJson, "hasSharedExpert");
    }
    if (paramJson.contains("hasSharedExpertGate")) {
        this->hasSharedExpertGate = atb_speed::base::FetchJsonParam<bool>(paramJson, "hasSharedExpertGate");
    }
    if (paramJson.contains("maskStartIdx")) {
        maskStartIdx = atb_speed::base::FetchJsonParam<int>(paramJson, "maskStartIdx");
    }
    if (paramJson.contains("numOfSelectedExperts")) {
        for (auto item : paramJson["numOfSelectedExperts"]) {
            this->numOfSelectedExperts.push_back(item.get<int>());
        }
    }
    if (paramJson.contains("deviceExpert")) {
        for (auto item : paramJson["deviceExpert"]) {
            deviceExpert.push_back(atb_speed::base::FetchJsonParam<int32_t>(item, "deviceExpert", true));
        }
    }
    if (paramJson.contains("enableGMMSwigluQuant")) {
        this->enableGMMSwigluQuant = paramJson["enableGMMSwigluQuant"].get<bool>();
    }
    if (paramJson.contains("enableDpOut")) {
        this->enableDpOut = paramJson["enableDpOut"].get<bool>();
    }
    if (paramJson.contains("lmHeadLocalTp")) {
        this->lmHeadLocalTp = paramJson["lmHeadLocalTp"].get<bool>();
    }
}

void MoeModelParam::ParseInteOpParams(const nlohmann::json &paramJson)
{
    if (paramJson.contains("enableFusedRouting")) {
        this->enableFusedRouting = paramJson["enableFusedRouting"].get<bool>();
    }
    if (paramJson.contains("enableInitQuant")) {
        this->enableInitQuant = paramJson["enableInitQuant"].get<bool>();
    }
    if (paramJson.contains("enableSwigluQuant")) {
        this->enableSwigluQuant = paramJson["enableSwigluQuant"].get<bool>();
    }
}

void MoeModelParam::ParseQuantParams(const nlohmann::json &paramJson)
{
    for (auto item : paramJson["moeLinearQuantType"]) {
        this->moeLinearQuantType.push_back(item.get<std::vector<int>>());
    }
    for (auto item : paramJson["mlpLinearQuantType"]) {
        this->mlpLinearQuantType.push_back(item.get<std::vector<int>>());
    }
    for (auto item : paramJson["moeLinearTransposeType"]) {
        this->moeLinearTransposeType.push_back(item.get<std::vector<int>>());
    }
    for (auto item : paramJson["mlpLinearTransposeType"]) {
        this->mlpLinearTransposeType.push_back(item.get<std::vector<int>>());
    }
}

void MoeModelParam::ParseAttnParallelParams(const nlohmann::json &paramJson)
{
    if (paramJson.contains("hasAttnTp")) {
        hasAttnTp = atb_speed::base::FetchJsonParam<bool>(paramJson, "hasAttnTp");
    }
    if (paramJson.contains("attnTpRank")) {
        attnTpRank = atb_speed::base::FetchJsonParam<int>(paramJson, "attnTpRank");
    }
    if (paramJson.contains("attnTpSize")) {
        attnTpSize = CheckPositive(atb_speed::base::FetchJsonParam<int>(paramJson, "attnTpSize"));
    }
    if (paramJson.contains("attnTpDomain")) {
        attnTpDomain = atb_speed::base::FetchJsonParam<std::string>(paramJson, "attnTpDomain");
    }
    if (paramJson.contains("hasAttnOprojTp")) {
        hasAttnOprojTp = atb_speed::base::FetchJsonParam<bool>(paramJson, "hasAttnOprojTp");
    }
    if (paramJson.contains("attnOprojTpRank")) {
        attnOprojTpRank = atb_speed::base::FetchJsonParam<int>(paramJson, "attnOprojTpRank");
    }
    if (paramJson.contains("attnOprojTpSize")) {
        attnOprojTpSize = CheckPositive(atb_speed::base::FetchJsonParam<int>(paramJson, "attnOprojTpSize"));
    }
    if (paramJson.contains("attnOprojTpDomain")) {
        attnOprojTpDomain = atb_speed::base::FetchJsonParam<std::string>(paramJson, "attnOprojTpDomain");
    }
    if (paramJson.contains("attnOprojPrefetch")) {
        attnOprojPrefetch = atb_speed::base::FetchJsonParam<bool>(paramJson, "attnOprojPrefetch");
    }
    if (paramJson.contains("hasAttnDp")) {
        hasAttnDp = atb_speed::base::FetchJsonParam<bool>(paramJson, "hasAttnDp");
    }
    if (paramJson.contains("attnDpRank")) {
        attnDpRank = atb_speed::base::FetchJsonParam<int>(paramJson, "attnDpRank");
    }
    if (paramJson.contains("attnDpSize")) {
        attnDpSize = CheckPositive(atb_speed::base::FetchJsonParam<int>(paramJson, "attnDpSize"));
    }
    if (paramJson.contains("attnDpDomain")) {
        attnDpDomain = atb_speed::base::FetchJsonParam<std::string>(paramJson, "attnDpDomain");
    }
}

void MoeModelParam::ParseParallelParams(const nlohmann::json &paramJson)
{
    if (paramJson.contains("hasMlpTp")) {
        hasMlpTp = atb_speed::base::FetchJsonParam<bool>(paramJson, "hasMlpTp");
    }
    if (paramJson.contains("mlpTpRank")) {
        mlpTpRank = atb_speed::base::FetchJsonParam<int>(paramJson, "mlpTpRank");
    }
    if (paramJson.contains("mlpTpSize")) {
        mlpTpSize = CheckPositive(atb_speed::base::FetchJsonParam<int>(paramJson, "mlpTpSize"));
    }
    if (paramJson.contains("mlpTpDomain")) {
        mlpTpDomain = atb_speed::base::FetchJsonParam<std::string>(paramJson, "mlpTpDomain");
    }
    if (paramJson.contains("hasMoeEp")) {
        hasMoeEp = atb_speed::base::FetchJsonParam<bool>(paramJson, "hasMoeEp");
    }
    if (paramJson.contains("moeEpRank")) {
        moeEpRank = atb_speed::base::FetchJsonParam<int>(paramJson, "moeEpRank");
    }
    if (paramJson.contains("moeEpSize")) {
        moeEpSize = CheckPositive(atb_speed::base::FetchJsonParam<int>(paramJson, "moeEpSize"));
    }
    if (paramJson.contains("moeEpDomain")) {
        moeEpDomain = atb_speed::base::FetchJsonParam<std::string>(paramJson, "moeEpDomain");
    }
    if (paramJson.contains("hasMoeTp")) {
        hasMoeTp = atb_speed::base::FetchJsonParam<bool>(paramJson, "hasMoeTp");
    }
    if (paramJson.contains("moeTpRank")) {
        moeTpRank = atb_speed::base::FetchJsonParam<int>(paramJson, "moeTpRank");
    }
    if (paramJson.contains("moeTpSize")) {
        moeTpSize = CheckPositive(atb_speed::base::FetchJsonParam<int>(paramJson, "moeTpSize"));
    }
    if (paramJson.contains("moeTpDomain")) {
        moeTpDomain = atb_speed::base::FetchJsonParam<std::string>(paramJson, "moeTpDomain");
    }
    if (paramJson.contains("lmHeadTpRank")) {
        lmHeadTpRank = atb_speed::base::FetchJsonParam<int>(paramJson, "lmHeadTpRank");
    }
    if (paramJson.contains("lmHeadTpSize")) {
        lmHeadTpSize = CheckPositive(atb_speed::base::FetchJsonParam<int>(paramJson, "lmHeadTpSize"));
    }
    if (paramJson.contains("lmHeadTpDomain")) {
        lmHeadTpDomain = atb_speed::base::FetchJsonParam<std::string>(paramJson, "lmHeadTpDomain");
    }
    if (paramJson.contains("maxDecodeDpTokenSize")) {
        maxDecodeDpTokenSize = atb_speed::base::FetchJsonParam<int>(paramJson, "maxDecodeDpTokenSize");
    }
}

void MoeModelParam::CheckParallelParamValid()
{
    if (attnTpRank >= attnTpSize) {
        throw std::runtime_error("attnTpSize must be greater than attnTpRank, please check.");
    }
    if (attnDpRank >= attnDpSize) {
        throw std::runtime_error("attnDpSize must be greater than attnDpRank, please check.");
    }
    if (mlpTpRank >= mlpTpSize) {
        throw std::runtime_error("mlpTpSize must be greater than mlpTpRank, please check.");
    }
    if (moeEpRank >= moeEpSize) {
        throw std::runtime_error("moeEpSize must be greater than moeEpRank, please check.");
    }
}

void MoeModelParam::PrintParam()
{
    atb_speed::base::ModelParam::PrintParam();
    ATB_SPEED_LOG_DEBUG(", numOfExperts: " << this->numOfExperts
                  << ", expertParallelDegree: " << this->expertParallelDegree
                  << ", numOfSelectedExperts:" << this->numOfSelectedExperts
                  << ", routingMethod: " << this->routingMethod
                  << ", processLogits" << this->processLogits
                  << ", normHasBias: " << this->normHasBias
                  << ", enableFusedRouting: " << this->enableFusedRouting
                  << ", moeLinearQuantType: " << this->moeLinearQuantType
                  << ", mlpLinearQuantType: " << this->mlpLinearQuantType
                  << ", moeLinearTransposeType: " << this->moeLinearTransposeType
                  << ", mlpLinearTransposeType: " << this->mlpLinearTransposeType);
}

void MoeModelParam::CheckRoutingMethodValid()
{
    std::set<std::string> supportRoutingMethods = {"softMaxTopK", "integratedSoftmaxTopK", "deviceLimited", "noAuxTc"};
    if (supportRoutingMethods.find(this->routingMethod) == supportRoutingMethods.end()) {
        std::stringstream ss;
        ss << "The routing method " << this->routingMethod << " is not valid." << std::endl;
        ATB_SPEED_LOG_ERROR(ss.str());
        throw std::runtime_error(ss.str());
    }
}

void MoeModelParam::CheckProcessLogitsValid()
{
    std::set<std::string> supportProcessLogits = {"none", "normalization", "scaling", "normScaling"};
    if (supportProcessLogits.find(this->processLogits) == supportProcessLogits.end()) {
        std::stringstream ss;
        ss << "The process logits method" << this->processLogits << " is not valid." << std::endl;
        ATB_SPEED_LOG_ERROR(ss.str());
        throw std::runtime_error(ss.str());
    }
}

void MoeModelParam::CheckParam()
{
    CheckLinearParamsSufficient(this->moeLinearQuantType, this->numHiddenLayers, MOE_LINEAR_TYPE_LENGTH);
    CheckLinearParamsSufficient(this->mlpLinearQuantType, this->numHiddenLayers, MOE_LINEAR_TYPE_LENGTH);
    CheckLinearParamsSufficient(this->moeLinearTransposeType, this->numHiddenLayers, MOE_LINEAR_TYPE_LENGTH);
    CheckLinearParamsSufficient(this->mlpLinearTransposeType, this->numHiddenLayers, MOE_LINEAR_TYPE_LENGTH);
    CheckRoutingMethodValid();
}

MoeDecoderModel::MoeDecoderModel(const std::string &param) : atb_speed::base::DecoderModel(param)
{
    this->param.FromString(param);
    this->inTensorCandidates["default_moe"] = {
        "expert_array_model", "expert_group_model", "one_hot_model", "zero_hot_model"};
    this->inTensorCandidates["fused_routing"] = {
        "in_final_hidden_state", "in_final_hidden_state_two", "in_final_bias"};
}

void MoeDecoderModel::ConstructInTensorMap()
{
    DecoderModel::ConstructInTensorMap();
    atb_speed::common::AssignTensorIdx(this->inTensorCandidates, "default_moe", this->inTensorMap);
}

atb::Status MoeDecoderModel::CreateLayerOperation(atb::Operation **op, uint32_t layerId)
{
    MoeLayerParam layerParam;
    this->SetLayerParam(layerParam, layerId);
    if (this->param.normType == atb_speed::base::RMS_NORM) {
        MoeDecoderLayer<atb::infer::RmsNormParam> decoderLayer(layerParam);
        CHECK_OPERATION_STATUS_RETURN(decoderLayer.BuildGraph(op));
    } else {
        MoeDecoderLayer<atb::infer::LayerNormParam> decoderLayer(layerParam);
        CHECK_OPERATION_STATUS_RETURN(decoderLayer.BuildGraph(op));
    }
    return atb::NO_ERROR;
}

void MoeDecoderModel::SetLayerParam(MoeLayerParam &layerParam, uint32_t layerId)
{
    atb_speed::base::DecoderModel::SetLayerParam(layerParam, layerId);
    layerParam.numOfExperts = this->param.numOfExperts;
    layerParam.expertParallelDegree = this->param.expertParallelDegree;
    layerParam.routingMethod = this->param.routingMethod;
    layerParam.numOfSelectedExperts = this->param.numOfSelectedExperts;
    layerParam.normHasBias = this->param.normHasBias;
    layerParam.enableFusedRouting = this->param.enableFusedRouting;
    layerParam.enableGMMSwigluQuant = this->param.enableGMMSwigluQuant;
    layerParam.enableInitQuant = this->param.enableInitQuant;
    layerParam.enableSwigluQuant = this->param.enableSwigluQuant;
    layerParam.processLogits = this->param.processLogits;
    layerParam.hasMoeEp = this->param.hasMoeEp;
    layerParam.moeLinearQuantType = this->param.moeLinearQuantType[layerId];
    layerParam.mlpLinearQuantType = this->param.mlpLinearQuantType[layerId];
    layerParam.moeLinearTransposeType = this->param.moeLinearTransposeType[layerId];
    layerParam.mlpLinearTransposeType = this->param.mlpLinearTransposeType[layerId];
}

void MoeDecoderModel::SetLayerNodeInput(atb_speed::Model::Node &layerNode, uint32_t layerId)
{
    uint32_t inTensorId = 0;
    this->SetLayerNodeDefaultInput(layerNode, layerId, inTensorId);
    this->SetLayerNodeOptionalInput(layerNode, layerId, inTensorId);
    layerNode.inTensors.at(inTensorId++) = \
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "expert_array_model"));
    layerNode.inTensors.at(inTensorId++) = \
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "expert_group_model"));
    layerNode.inTensors.at(inTensorId++) = \
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "one_hot_model"));
    layerNode.inTensors.at(inTensorId++) = \
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "zero_hot_model"));
    if (this->param.enableSpeculate || this->param.enableSplitFuse || this->param.enablePrefixCache) {
        layerNode.inTensors.at(inTensorId++) = \
            &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "q_len"));
    }
}

} // namespace moe
} // namespace atb_speed