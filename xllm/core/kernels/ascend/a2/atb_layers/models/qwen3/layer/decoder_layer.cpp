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

#include "models/qwen3/layer/decoder_layer.h"

namespace atb_speed {
namespace qwen {

void QwenLayerParam::PrintParam()
{
    LayerParam::PrintParam();
    std::stringstream ss;
    ss << " Layer Param: " << "enableLogN: " << this->enableLogN << ", isEmbedding: " << this->isEmbedding
       << ", enableQScale: " << this->enableQScale;
    ATB_SPEED_LOG_DEBUG(ss.str());
}

QwenDecoderLayer::QwenDecoderLayer(const QwenLayerParam &param) : base::DecoderLayer<atb::infer::RmsNormParam>(param)
{
    this->param = param;
    this->param.PrintParam();
    this->inTensorCandidates["logn_enable"] = {"kv_cache_idx"};
};

void QwenDecoderLayer::ConstructInTensorMap()
{
    this->inTensorList.clear();
    // 添加默认的Tensor
    atb_speed::common::AddTensorToList(this->inTensorCandidates, "input_norm_weight", this->inTensorList);
    atb_speed::common::AddTensorToList(this->inTensorCandidates, "attn_weight", this->inTensorList);
    atb_speed::common::AddTensorToList(this->inTensorCandidates, "post_attn_norm_weight", this->inTensorList);
    atb_speed::common::AddTensorToList(this->inTensorCandidates, "mlp_weight", this->inTensorList);
    // 添加AddRmsNormQuant特性的Tensor
    if (param.enableInterLayerAddNorm || param.enableIntraLayerAddNorm) {
        atb_speed::common::AddTensorToList(this->inTensorCandidates, "add_rmsnorm_quant", this->inTensorList);
    }
    // 添加KV cache int8特性的Tensor
    if (param.enableKvQuant) {
        atb_speed::common::AddTensorToList(this->inTensorCandidates, "kv_quant_scale", this->inTensorList);
        if (param.kvQuantHasOffset) {
            atb_speed::common::AddTensorToList(this->inTensorCandidates, "kv_quant_offset", this->inTensorList);
        }
    }
    // 添加 QKNorm 特性的Tensor
    if (param.useQKNorm) {
        atb_speed::common::AddTensorToList(this->inTensorCandidates, "qk_norm", this->inTensorList);
    }
    // 添加FA3特性的Tensor
    if (param.enableFA3) {
        atb_speed::common::AddTensorToList(this->inTensorCandidates, "fa3_quant", this->inTensorList);
    }

    // 添加lccl reduce int8特性的Tensor
    if (param.enableReduceQuant) {
        atb_speed::common::AddTensorToList(this->inTensorCandidates, "reduce_quant_attn", this->inTensorList);
        atb_speed::common::AddTensorToList(this->inTensorCandidates, "reduce_quant_mlp", this->inTensorList);
    }

    atb_speed::common::AddTensorToList(this->inTensorCandidates, "default", this->inTensorList);
    atb_speed::common::AddTensorToList(
        this->internalTensorCandidates, "default", this->intermediateTensorList);

    // 添加头压缩特性的Tensor
    if (param.enableCompressHead) {
        atb_speed::common::AddTensorToList(
            this->inTensorCandidates,
            param.positionEmbeddingType == base::PositionEmbeddingType::ALIBI ? "compress_head_alibi" : "compress_head_rope",
            this->inTensorList);
    }

    // 添加omniattention特性的Tensor
    if (param.enableOmniAttention) {
        atb_speed::common::AddTensorToList(this->inTensorCandidates, "compress_head_rope", this->inTensorList);
    }

    // 添加并行解码特性或SplitFuse或isPrefixCacheWithoutChunk的Tensor
    if (param.enableSpeculate || param.enableSplitFuse || param.isPrefixCacheWithoutChunk) {
        atb_speed::common::AddTensorToList(this->inTensorCandidates, "q_len", this->inTensorList);
    }

    // 添加lora特性的Tensor
    if (param.enableLora) {
        atb_speed::common::AddTensorToList(this->inTensorCandidates, "lora_common", this->inTensorList);
        atb_speed::common::AddTensorToList(this->inTensorCandidates, "lora_attn", this->inTensorList);
        atb_speed::common::AddTensorToList(this->inTensorCandidates, "lora_mlp", this->inTensorList);
    }

    if (param.hasAttnDp) {
        atb_speed::common::AddTensorToList(this->inTensorCandidates, "attn_dp", this->inTensorList);
    }
    // 添加AddNorm融合特性的Tensor
    if (param.enableInterLayerAddNorm && param.layerId != 0) {
        atb_speed::common::AddTensorToList(this->inTensorCandidates, "input_add_norm", this->inTensorList);
    }

    // 添加logn特性的Tensor
    if (this->param.enableLogN) {
        atb_speed::common::AddTensorToList(this->inTensorCandidates, "logn_enable", this->inTensorList);
    }
}

void QwenDecoderLayer::SetFusionAttentionParam(
    atb_speed::common::FusionAttentionParam<atb::infer::RmsNormParam> &fusionAttentionParam)
{
    DecoderLayer<atb::infer::RmsNormParam>::SetFusionAttentionParam(fusionAttentionParam);
    if (this->param.enableLogN) {
        fusionAttentionParam.pageAttentionParam.scaleType = atb::infer::PagedAttentionParam::SCALE_TYPE_LOGN;
    }
    if (this->param.isEmbedding) {
        fusionAttentionParam.selfAttentionParam.maskType =
            atb::infer::SelfAttentionParam::MaskType::MASK_TYPE_UNDEFINED;
    } else {
        fusionAttentionParam.selfAttentionParam.maskType = atb::infer::SelfAttentionParam::MaskType::MASK_TYPE_NORM;
    }
    fusionAttentionParam.enableQScale = !param.isFA && param.enableQScale;
    fusionAttentionParam.pageAttentionParam.qkScale =
        param.enableQScale ? 1.0 : fusionAttentionParam.pageAttentionParam.qkScale;
    fusionAttentionParam.selfAttentionParam.qkScale =
        param.enableQScale ? 1.0 : fusionAttentionParam.selfAttentionParam.qkScale;
    fusionAttentionParam.enableAclnnRmsNorm = param.enableAclnnRmsNorm;
    fusionAttentionParam.isPrefixCacheWithoutChunk = param.isPrefixCacheWithoutChunk;
    if (fusionAttentionParam.isPrefixCacheWithoutChunk) {
        fusionAttentionParam.selfAttentionParam.maskType = atb::infer::SelfAttentionParam::MaskType::MASK_TYPE_NORM_COMPRESS;
    }
}

void QwenDecoderLayer::SetMlpNormParam(
    atb_speed::common::MlpParam<atb::infer::RmsNormParam> &mlpParam)
{
    DecoderLayer<atb::infer::RmsNormParam>::SetMlpNormParam(mlpParam);
    mlpParam.enableAclnnRmsNorm = param.enableAclnnRmsNorm;
}

void QwenDecoderLayer::SetFusionAttentionATBSelfAttentionParam(
        atb_speed::common::FusionAttentionParam<atb::infer::RmsNormParam> &fusionAttentionParam)
{
    fusionAttentionParam.selfAttentionParam.headNum = this->param.numAttentionHeadsPerRank;
    fusionAttentionParam.selfAttentionParam.kvHeadNum = this->param.numKeyValueHeadsPerRank;
    fusionAttentionParam.selfAttentionParam.qkScale = 1.0 / sqrt(this->param.hiddenSizePerAttentionHead);
    if (this->param.isFA) {
        fusionAttentionParam.selfAttentionParam.calcType = this->param.isPrefill ? \
            atb::infer::SelfAttentionParam::CalcType::ENCODER : atb::infer::SelfAttentionParam::CalcType::DECODER;
    } else {
        fusionAttentionParam.selfAttentionParam.isTriuMask = this->param.isPrefill ? 1 : 0;
        if (param.isPrefixCacheWithoutChunk) {
            fusionAttentionParam.selfAttentionParam.calcType = atb::infer::SelfAttentionParam::CalcType::PREFIX_ENCODER;
            fusionAttentionParam.selfAttentionParam.kernelType = atb::infer::SelfAttentionParam::KernelType::KERNELTYPE_HIGH_PRECISION;
        } else {
            fusionAttentionParam.selfAttentionParam.calcType = atb::infer::SelfAttentionParam::CalcType::PA_ENCODER;
        }
    }
    if (this->param.attnBackend == atb_speed::common::OpBackend::ACLNN && param.isFA) {
        fusionAttentionParam.selfAttentionParam.calcType = atb::infer::SelfAttentionParam::CalcType::PA_ENCODER;
    }
}

std::map<unsigned int, std::vector<std::string>> QwenDecoderLayer::GetAttentionIntensor()
{
    std::map<unsigned int, std::vector<std::string>> attnInTensor = {};
    attnInTensor[common::AttnInTensorCategory::ATTN_DEFAULT] = {
        "in_hidden_states", "in_input_norm_weight", "in_input_norm_bias", "in_input_norm_new_weight",
        "in_input_norm_new_bias", "in_qkv_weight_0", "in_qkv_scale_0", "in_qkv_offset_0", "in_qkv_descale_0",
        "in_qkv_bias_0", "in_qkv_compress_idx_0", "in_qkv_weight_1", "in_qkv_scale_1", "in_qkv_offset_1",
        "in_qkv_descale_1", "in_qkv_bias_1", "in_qkv_compress_idx_1", "in_qkv_weight_2", "in_qkv_scale_2",
        "in_qkv_offset_2", "in_qkv_descale_2", "in_qkv_bias_2", "in_qkv_compress_idx_2", "in_cos_embedding",
        "in_sin_embedding", "in_seq_len", "in_k_cache", "in_v_cache", "in_attention_mask", "in_token_offset",
        "in_layer_id", "in_block_tables", "in_slots", "in_qkv_dense_weight", "in_qkv_dense_scale",
        "in_qkv_dense_offset", "in_qkv_dense_descale", "in_qkv_dense_bias", "in_qkv_dense_compress_idx"};
    if (this->param.enableCompressHead) {
        if (this->param.positionEmbeddingType == base::PositionEmbeddingType::ALIBI) {
            attnInTensor[common::AttnInTensorCategory::ATTN_COMPRESS_HEAD_ALIBI] = \
                this->inTensorCandidates["compress_head_alibi"];
        } else {
            attnInTensor[common::AttnInTensorCategory::ATTN_COMPRESS_HEAD_ROPE] = \
                this->inTensorCandidates["compress_head_rope"];
                }
    }
    if (this->param.enableOmniAttention) {
        attnInTensor[common::AttnInTensorCategory::ATTN_OMNI] = \
            this->inTensorCandidates["compress_head_rope"];
    }
    if (this->param.enableSpeculate || param.enableSplitFuse || param.isPrefixCacheWithoutChunk) {
        attnInTensor[common::AttnInTensorCategory::ATTN_SPECULATE] = this->inTensorCandidates["q_len"];
    }
    if (this->param.enableKvQuant) {
        attnInTensor[common::AttnInTensorCategory::ATTN_KV_QUANT_SCALE] = this->inTensorCandidates["kv_quant_scale"];
        if (this->param.kvQuantHasOffset) {
            attnInTensor[common::AttnInTensorCategory::ATTN_KV_QUANT_OFFSET] = \
                this->inTensorCandidates["kv_quant_offset"];
        }
    }
    if (this->param.enableFA3) {
        attnInTensor[common::AttnInTensorCategory::ATTN_FA3] = this->inTensorCandidates["fa3_quant"];
    }
    if (this->param.enableLora) {
        attnInTensor[common::AttnInTensorCategory::ATTN_LORA] = {"in_seq_len_cum_sum"};
        for (std::string tensor : this->inTensorCandidates.at("lora_attn")) {
            attnInTensor[common::AttnInTensorCategory::ATTN_LORA].push_back(tensor);
        }
    }
    if (this->param.enableReduceQuant) {
        attnInTensor[common::AttnInTensorCategory::ATTN_REDUCE_QUANT] = {};
        for (std::string tensor : this->inTensorCandidates.at("reduce_quant_attn")) {
            attnInTensor[common::AttnInTensorCategory::ATTN_REDUCE_QUANT].push_back(tensor);
        }
    }
    if (this->param.useQKNorm) {
        attnInTensor[common::AttnInTensorCategory::ATTN_QK_NORM] = this->inTensorCandidates["qk_norm"];
    }
    if (this->param.enableInterLayerAddNorm && (this->param.layerId != 0)) {
        attnInTensor[common::AttnInTensorCategory::ATTN_ADD_RMS_NORM_QUANT].push_back("in_qkv_scale_fill");
        attnInTensor[common::AttnInTensorCategory::ATTN_ADD_RMS_NORM_QUANT].push_back("in_qkv_offset_fill");
        attnInTensor[common::AttnInTensorCategory::ATTN_ADD_NORM] = {"in_last_mlp_out"};
    }
    if (this->param.enablePreFetchWeight) {
        attnInTensor[common::AttnInTensorCategory::ATTN_CMO] = {"in_mlp_weight_0"};
    }
    
    if (this->param.enableLogN) {
        attnInTensor[common::AttnInTensorCategory::ATTN_LOG_N_SCALE] = {"kv_cache_idx"};
    }
    return attnInTensor;
}

} // namespace qwen
} // namespace atb_speed