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
#ifndef ATB_SPEED_MODELS_MOE_DECODER_LAYER_H
#define ATB_SPEED_MODELS_MOE_DECODER_LAYER_H

#include <vector>
#include "nlohmann/json.hpp"
#include "atb/atb_infer.h"
#include "atb_speed/log.h"
#include "operations/fusion/moe/sparse_moe.h"
#include "models/base/param/layer_param.h"
#include "models/base/layer/decoder_layer.h"

namespace atb_speed {
namespace moe {

/// A class that defines the parameters of base MoE layers.
class MoeLayerParam : public atb_speed::base::LayerParam {
public:
    MoeLayerParam() = default;
    virtual ~MoeLayerParam() override = default;
    void PrintParam() override;

    MoeLayerParam(const MoeLayerParam&) = default;
    MoeLayerParam& operator=(const MoeLayerParam&) = default;

    MoeLayerParam(MoeLayerParam&&) = default;
    MoeLayerParam& operator=(MoeLayerParam&&) = default;

    /// Deprecated.
    bool enableTopKSoftmax = false;
    /// A flag indicating whether matrecies need to be transpose for matrix multiplications.
    bool transpose = true;
    /// The total number of experts utilized by the model.
    uint32_t numOfExperts = 64;
    /// The number of experts loaded to the device.
    uint32_t numOfDeviceExperts = 64;
    /// The size of the expert parallelism communication domain.
    int expertParallelDegree = 0;
    /// A flag that indicates whether the norm in this layer has bias.
    bool normHasBias = false;
    /// A flag indicating whether or not to use integrated routing operators.
    bool enableFusedRouting = false;
    /// A flag indicating whether or not to use integrated GMM+Swiglu+quant operators.
    bool enableGMMSwigluQuant = false;
    /// A flag indicating whether or not to use fused atb GMM+Swiglu+quant operators instead of aclnn.
    bool enableAtlasGMMFused = false;
    /// A flag indicating whether to use the integrated routing-quant operator
    bool enableInitQuant = false;
    /// A flag indicating whether to use the integrated swiglu-quant operator
    bool enableSwigluQuant = false;
    /// The way in which the top k experts are selected.
    std::string routingMethod = "softMaxTopK";
    /// The way in which expert scores are further processed.
    std::string processLogits = "normalization";

    // ==== 并行策略参数 ====

    /// A flag indicating whether or not to use tensor parallelism in attention.
    bool hasAttnOprojTp = false;
    /// The rank of this device in the tensor parallelism communication domain in attention.
    int attnOprojTpRank = 0;
    /// The size of the tensor parallelism communication domain in attention.
    int attnOprojTpSize = 1;
    /// The communication domain of tensor parallelism in attention.
    std::string attnOprojTpDomain = "";
    /// The rankTableFile for the device in the attnTp communication domain.
    std::string attnOprojTpRankTableFile = "";
    std::string attnOprojTpBackend = "";
    bool attnOprojPrefetch = false;
    /// The rank of the current device within a lmhead TP communication domain
    int lmHeadTpRank = 0;
    /// The size of the lmhead TP communication domain that the current device is in
    int lmHeadTpSize = 1;
    /// The id of the lmhead TP communication domain
    std::string lmHeadTpDomain = "";
    /// A flag indicating whether the data collected after lmhead will again be data-wise separated
    bool lmHeadLocalTp = false;
    bool enableDpOut = false;
    /// A flag indicating whether the model utilizes expert parallelism.
    bool hasMoeEp = false;
    /// The rank of this device in the expert parallelism communication domain.
    int moeEpRank = 0;
    /// The size of the expert parallelism communication domain.
    int moeEpSize = 1;
    int maxDecodeDpTokenSize = 0;
    /// The communication domain of expert parallelism.
    std::string moeEpDomain = "";
    /// The rankTableFile for the device in the communication domain.
    std::string moeEpRankTableFile = "";
    std::string moeEpBackend = "";
    /// A flag indicating whether the model utilizes expert parallelism.
    bool hasMoeTp = false;
    /// The rank of this device in the expert parallelism communication domain.
    int moeTpRank = 0;
    /// The size of the expert parallelism communication domain.
    int moeTpSize = 1;
    /// The communication domain of expert parallelism.
    std::string moeTpDomain = "";
    /// The rankTableFile for the device in the communication domain.
    std::string moeTpRankTableFile = "";
    std::string moeTpBackend = "";
    /// The list of experts loaded on the device.
    std::vector<int32_t> deviceExpert = {};
    /// A vector that defines the quantization types of linears in MoE.
    /// It is automatically generated by MoE weight wrapper.
    std::vector<int> moeLinearQuantType = {};
    /// A vector that defines the quantization types of linears in MLP.
    /// It is automatically generated by MoE weight wrapper.
    std::vector<int> mlpLinearQuantType = {};
    /// A vector that defines the transpose types of linears in MoE.
    /// It is automatically generated by MoE weight wrapper.
    std::vector<int> moeLinearTransposeType = {};
    /// A vector that defines the transpose types of linears in MLP.
    /// It is automatically generated by MoE weight wrapper.
    std::vector<int> mlpLinearTransposeType = {};
    /// The number of experts selected for each token.
    atb::SVector<int32_t> numOfSelectedExperts = {2};
};

/// A template class for representing a MoE Layer.
/// \tparam NormType atb_speed::base::RMS_NORM or atb::infer::LayerNormParam.
template <typename NormType>
class MoeDecoderLayer : public atb_speed::base::DecoderLayer<NormType> {
public:
    explicit MoeDecoderLayer(const MoeLayerParam &param);
    ~MoeDecoderLayer() override {};

protected:
    /// A function that constructs a map from in tensor names to in tensor ids.
    void ConstructInTensorMap() override;
    /// A function that constructs a map from internal tensor names to in tensor ids.
    void ConstructInternalTensorMap() override;
    /// A function that constructs a map from layer tensor names to InsertNorm function tensor names.
    std::map<std::string, uint32_t> ConstructNormTensorMap() const;

    /// A function that update parameters of a `SparseMoeParam` based on parsed MoE layer parameters.
    /// \param sparseMoeParam An `atb_speed::common::SparseMoeParam` object that needs to be updated.
    virtual void SetSparseMoeParam(atb_speed::common::SparseMoeParam &sparseMoeParam);
    /// A function that add operations to a layer graph.
    /// \return A flag that indicates whether operations are added successfully.
    atb::Status AddOperationToGraph() override;
    
    /// A function that update parameters of a `NormLinearParam` based on parsed MoE layer parameters.
    /// \param selfNormParam An `atb_speed::common::NormLinearParam<NormType>` object that needs to be updated.
    virtual void SetSelfNormParam(atb_speed::common::NormLinearParam<NormType> &selfNormParam);
    /// A function that add a normalizaton node in a MoE layer.
    /// \return A flag that indicates whether the normalizaton node is added successfully.
    virtual atb::Status AddSelfNorm();
    /// A function that add a MoE node in a MoE layer.
    /// \return A flag that indicates whether the MoE node is added successfully.
    virtual atb::Status AddMoe();
    /// A function that add an allReduce node after a MoE node in a MoE layer.
    /// \return A flag that indicates whether the allReduce node is added successfully.
    virtual atb::Status AddMoeAllReduce();

    MoeLayerParam param;
};
}  // namespace moe
}  // namespace atb_speed
#endif  // ATB_SPEED_MODELS_MOE_DECODER_LAYER_H