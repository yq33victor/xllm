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

 #include "operations/fusion/linear/linear.h"
 #include "operations/fusion/linear/linear_parallel.h"
 #include "operations/fusion/norm/norm_linear.h"
 #include "operations/fusion/mlp/mlp.h"
 #include "operations/fusion/moe/sparse_moe.h"
 #include "operations/aclnn/ops/inplace_nan_to_num_operation.h"
 #include "operations/fusion/moe/moe_shared_expert.h"
 #include "models/deepseekv2/operation/latent_attention.h"
 #include "atb_speed/base/event_manager.h"
 #include "models/deepseekv2/layer/decoder_layer.h"
 #include <gflags/gflags.h>

DECLARE_bool(enable_atb_comm_multiprocess);

 namespace atb_speed {
 namespace deepseekV2 {
 
 static const uint64_t STREAM1 = 1;
 static const uint64_t NUM2 = 2;
 static const float FLOAT16_MAX = 65504.0f;
 static const float FLOAT16_MIN = -65504.0f;
 
 void SetDeepseekV2LayerInTensorDefaultCandidates(
     std::map<std::string, std::vector<std::string>> &deepseekV2LayerInTensorCandidates)
 {
     deepseekV2LayerInTensorCandidates["default"] = {
             "in_hidden_states", "in_expert_array", "in_expert_group", "in_one_hot", "in_zero_hot",
             "in_final_state",
             "in_cos_table", "in_sin_table", "in_attention_mask", "in_k_cache", "in_k_rope_cache", "in_seq_len",
             "in_place_holder", "in_token_offset", "in_layer_id", "in_block_tables", "in_slots", "in_q_len"};
     deepseekV2LayerInTensorCandidates["default_weight"] = {
             "in_input_norm_weight", "in_input_norm_bias", "in_input_norm_new_weight", "in_input_norm_new_bias",
             "in_q_proj_a_weight", "in_q_proj_a_bias", "in_q_proj_a_descale", "in_q_proj_a_offset", "in_q_proj_a_scale",
             "in_q_proj_a_compress_idx", "in_q_proj_a_layernorm_weight", "in_q_proj_a_layernorm_bias",
             "in_q_proj_b_weight", "in_q_proj_b_bias", "in_q_proj_b_descale", "in_q_proj_b_offset", "in_q_proj_b_scale",
             "in_q_proj_b_compress_idx", "in_kv_proj_with_mqa_weight", "in_kv_proj_with_mqa_bias",
             "in_kv_proj_with_mqa_descale", "in_kv_proj_with_mqa_offset", "in_kv_proj_with_mqa_scale",
             "in_kv_proj_with_mqa_compress_idx", "in_kv_proj_a_layernorm_weight", "in_kv_proj_a_layernorm_bias",
             "in_k_proj_b_for_q_weight", "in_k_proj_b_for_q_bias", "in_k_proj_b_for_q_descale",
             "in_k_proj_b_for_q_offset", "in_k_proj_b_for_q_scale", "in_k_proj_b_for_q_compress_idx",
             "in_v_proj_b_for_o_weight", "in_v_proj_b_for_o_bias", "in_v_proj_b_for_o_descale",
             "in_v_proj_b_for_o_offset", "in_v_proj_b_for_o_scale", "in_v_proj_b_for_o_compress_idx",
             "in_attention_out_weight", "in_attention_out_bias", "in_attention_out_descale", "in_attention_out_offset",
             "in_attention_out_scale", "in_attention_out_compress_idx", "in_selfattention_out_norm_weight",
             "in_selfattention_out_norm_bias", "in_selfattention_out_new_norm_weight",
             "in_selfattention_out_new_norm_bias", "in_mlp_gateup_weight_shared_expert",
             "in_mlp_gateup_bias_shared_expert", "in_mlp_gateup_descale_shared_expert",
             "in_mlp_gateup_offset_shared_expert", "in_mlp_gateup_scale_shared_expert",
             "in_mlp_gateup_compress_idx_shared_expert", "in_mlp_down_weight_shared_expert",
             "in_mlp_down_bias_shared_expert", "in_mlp_down_descale_shared_expert",
             "in_mlp_down_offset_shared_expert", "in_mlp_down_scale_shared_expert",
             "in_mlp_down_compress_idx_shared_expert", "in_shared_expert_gate_weight", "in_shared_expert_gate_bias",
             "in_shared_expert_gate_descale", "in_shared_expert_gate_offset", "in_shared_expert_gate_scale",
             "in_shared_expert_gate_compress_idx", "in_block_sparse_moe_gate_weight", "in_block_sparse_moe_gate_bias",
             "in_block_sparse_moe_gate_descale", "in_block_sparse_moe_gate_offset", "in_block_sparse_moe_gate_scale",
             "in_block_sparse_moe_gate_compress_idx", "in_mlp_gateup_weight_expert", "in_mlp_gateup_bias_expert",
             "in_mlp_gateup_descale_expert", "in_mlp_gateup_offset_expert", "in_mlp_gateup_scale_expert",
             "in_mlp_gateup_compress_idx_expert", "in_mlp_down_weight_expert", "in_mlp_down_bias_expert",
             "in_mlp_down_descale_expert", "in_mlp_down_offset_expert", "in_mlp_down_scale_expert",
             "in_mlp_down_compress_idx_expert"};
 }
 
 std::map<std::string, std::vector<std::string>> GetDeepseekV2LayerInTensorCandidates()
 {
     std::map<std::string, std::vector<std::string>> deepseekV2LayerInTensorCandidates = {
         {"fa3_quant", {
             "in_q_quant_scale", "in_k_quant_scale", "in_qk_descale",
             "kv_offset", "fa3_v_quant_scale"}},
         {"parallel_input", {
             "in_attn_padding_idx", "in_attn_unpadding_idx", "in_ffn_padding_idx",
             "in_ffn_unpadding_idx", "in_lm_head_skip_padding_token_indices",
             "in_attention_padding_idx_slice", "in_start_expert_idx",
             "in_device_expert_count", "in_attention_padding_idx", "in_attention_unpadding_idx",
             "in_lty_idx", "in_moe_idx"}},
         {"decoder_weight", {
             "in_mlp_gateup_weight_shared_expert_tp", "in_mlp_gateup_bias_shared_expert_tp",
             "in_mlp_gateup_descale_shared_expert_tp", "in_mlp_gateup_offset_shared_expert_tp",
             "in_mlp_gateup_scale_shared_expert_tp", "in_mlp_gateup_compress_idx_shared_expert_tp",
             "in_mlp_down_weight_shared_expert_tp", "in_mlp_down_bias_shared_expert_tp",
             "in_mlp_down_descale_shared_expert_tp", "in_mlp_down_offset_shared_expert_tp",
             "in_mlp_down_scale_shared_expert_tp", "in_mlp_down_compress_idx_shared_expert_tp",
             "in_shared_expert_gate_weight_tp", "in_shared_expert_gate_bias_tp",
             "in_shared_expert_gate_descale_tp", "in_shared_expert_gate_offset_tp",
             "in_shared_expert_gate_scale_tp", "in_shared_expert_gate_compress_idx_tp",
             "in_block_sparse_moe_gate_weight_shuffled", "in_block_sparse_moe_gate_bias_shuffled",
             "in_block_sparse_moe_gate_descale_shuffled", "in_block_sparse_moe_gate_offset_shuffled",
             "in_block_sparse_moe_gate_scale_shuffled", "in_block_sparse_moe_gate_compress_idx_shuffled"
         }},
         {"attn_inner_sp_prefill", {"in_k_sp_gather_indices"}},
         {"attn_inner_sp_decode", {"in_seq_len_sp"}},
         {"force_load_balance", {
             "in_fake_topk"
         }},
         {"epwb", {
             "in_expert_routing_map"
         }},
     };
     SetDeepseekV2LayerInTensorDefaultCandidates(deepseekV2LayerInTensorCandidates);
     return deepseekV2LayerInTensorCandidates;
 }
 
 std::map<std::string, std::vector<std::string>> GetDeepseekV2LayerIntermediateTensorCandidates()
 {
     std::map<std::string, std::vector<std::string>> deepseekV2LayerIntermediateTensorCandidates = {
         {"default", {
             "intermediate_attention_out", "intermediate_attention_add_out", "intermediate_selfattention_norm_out",
             "intermediate_moe_out_with_shared"}},
         {"shared_expert", {
             "intermediate_shared_expert_out"}},
         {"attn_need_padding", {
             "intermediate_attention_out_padding",
             "intermediate_selfattention_norm_out_partial",
         }},
         {"attn_reduce_scatter", {
             "intermediate_attention_out_scatter"}},
         {"attn_allgather", {
             "intermediate_dp_attn_out_all_with_padding"}},
         {"ffn_reduce_scatter", {
             "intermediate_moe_out_with_shared_with_padding"}},
         {"ffn_allgather", {
             "intermediate_mlp_out_all"}},
         {"ffn_need_padding", {
             "intermediate_mlp_out"}},
         {"gatherprenorm", {
             "intermediate_selfattention_norm_out_fp32"}},
         {"hiddenstates_padding_slice", {
             "intermediate_hidden_states_padding", "intermediate_hidden_states_scatter"}},
         {"epwb", {
             "intermediate_expert_routing_map"
         }}
     };
     return deepseekV2LayerIntermediateTensorCandidates;
 }
 
 std::map<std::string, uint32_t> ConstructTensorMap(
     const DecoderLayerParam &param, uint32_t &inTensorNum, uint32_t &outTensorNum, uint32_t &internalTensorNum)
 {
     auto deepseekV2InTensorCandidates = GetDeepseekV2LayerInTensorCandidates();
     auto deepseekV2IntermediateCandidates = GetDeepseekV2LayerIntermediateTensorCandidates();
 
     std::vector<std::string> inTensorList = {};
     std::vector<std::string> intermediateTensorList = {};
     std::vector<std::string> outTensorList = {"out_decoder_layer"};
 
     if (!param.isDenseLayer && param.enableExpertCumSumOutput) {
         outTensorList.push_back("out_gmm_cumsum_list");
     }
 
     atb_speed::common::AddTensorToList(deepseekV2InTensorCandidates, "default_weight", inTensorList);
     if (param.enableFA3) {
         atb_speed::common::AddTensorToList(deepseekV2InTensorCandidates, "fa3_quant", inTensorList);
     }
 
     if (param.hasP2DWeight && param.layerId >= param.firstKDenseReplace) {
         atb_speed::common::AddTensorToList(deepseekV2InTensorCandidates, "decoder_weight", inTensorList);
     }
 
     atb_speed::common::AddTensorToList(deepseekV2InTensorCandidates, "default", inTensorList);
     if (param.enableLoadBalance) {
         atb_speed::common::AddTensorToList(deepseekV2InTensorCandidates, "force_load_balance", inTensorList);
     }
     atb_speed::common::AddTensorToList(deepseekV2InTensorCandidates, "parallel_input", inTensorList);
     if (param.enableEPWB) {
         atb_speed::common::AddTensorToList(deepseekV2InTensorCandidates, "epwb", inTensorList);
         if (param.layerId >= param.firstKDenseReplace) {
             atb_speed::common::AddTensorToList(deepseekV2IntermediateCandidates, "epwb", intermediateTensorList);
         }
     }
     atb_speed::common::AddTensorToList(deepseekV2IntermediateCandidates, "default", intermediateTensorList);
     if (param.hasSharedExpert && !param.isDenseLayer) {
         atb_speed::common::AddTensorToList(deepseekV2IntermediateCandidates, "shared_expert", intermediateTensorList);
     }
     if (!param.attnAllreduce && (param.hasAttnComm)) {
         atb_speed::common::AddTensorToList(deepseekV2IntermediateCandidates,
             "attn_need_padding", intermediateTensorList);
         if (!param.enableGatherPreNorm) {
             atb_speed::common::AddTensorToList(deepseekV2IntermediateCandidates,
                                                "hiddenstates_padding_slice", intermediateTensorList);
         }
 
         if (param.attnReduceScatter) {
             atb_speed::common::AddTensorToList(deepseekV2IntermediateCandidates,
                 "attn_reduce_scatter", intermediateTensorList);
         }
         if (param.attnAllGather) {
             atb_speed::common::AddTensorToList(deepseekV2IntermediateCandidates,
                 "attn_allgather", intermediateTensorList);
         }
     }
 
     if (param.mapping.Get(base::ATTN_INNER_SP).IsEnabled()) {
         atb_speed::common::AddTensorToList(deepseekV2InTensorCandidates,
             param.isPrefill ? "attn_inner_sp_prefill" : "attn_inner_sp_decode", inTensorList);
     }
 
     if (param.ffnAllreduce || param.hasFfnComm) {
         // 大ep场景下，开启h3p qkvdown dp时moe层不需要intermediate_mlp_out，最后一层除外
         if (!(param.enableQkvdownDp && !param.isLastLayer && param.ffnStreamNum > 1)) {
             atb_speed::common::AddTensorToList(deepseekV2IntermediateCandidates,
                 "ffn_need_padding", intermediateTensorList);
         }
     }
     if (param.ffnAllGather) {
         if (!param.enableQkvdownDp || param.isLastLayer) {
             atb_speed::common::AddTensorToList(deepseekV2IntermediateCandidates,
                 "ffn_allgather", intermediateTensorList);
         }
     }
     if (param.hasFfnComm) {
         atb_speed::common::AddTensorToList(deepseekV2IntermediateCandidates,
             "ffn_reduce_scatter", intermediateTensorList);
     }
     if (((param.attnReduceScatter || param.attnAllGather) && param.enableGatherPreNorm) || param.enableExtraOprojTp) {
         if (!param.enableQkvdownDp || param.layerId == param.firstKDenseReplace) {
             atb_speed::common::AddTensorToList(deepseekV2IntermediateCandidates,
                 "gatherprenorm", intermediateTensorList);
         }
     }
     inTensorNum = inTensorList.size();
     internalTensorNum = intermediateTensorList.size();
     outTensorNum = outTensorList.size();
     ATB_SPEED_LOG_DEBUG("ConstructTensorMap done");
     return atb_speed::common::GetTensorMap(inTensorList, outTensorList, intermediateTensorList);
 }
 
 atb::Status SetLatentAttentionInnerCommParam(
     atb_speed::deepseekV2::LatentAttentionParam<atb::infer::RmsNormParam> &latentAttentionParam,
     const DecoderLayerParam &param)
 {
     latentAttentionParam.enableExtraOprojTp = param.enableExtraOprojTp;
     latentAttentionParam.selfOutLinearInnerTensorParallelInfo = param.mapping.Get(base::ATTN_O_PROJ_TP);
     latentAttentionParam.selfOutLinearInnerTensorParallelInfoLCCL = param.mapping.Get(base::ATTN_O_PROJ_TP);
     latentAttentionParam.selfOutLinearInnerTensorParallelInfoLCCL.commDomain = "";
     latentAttentionParam.selfOutLinearInnerTensorParallelInfoLCCL.InitCommDomain("lccl");
     latentAttentionParam.attnOprojPrefetch = param.attnOprojPrefetch;
 
     if (param.attnAllreduce) {
         atb_speed::common::ParallelInfo parallelInfo = param.mapping.Get(base::ATTN_TP);
         latentAttentionParam.selfOutLinearTensorParallelInfo.rank = parallelInfo.rank;
         latentAttentionParam.selfOutLinearTensorParallelInfo.worldSize = parallelInfo.rankIds.size();
         latentAttentionParam.selfOutLinearTensorParallelInfo.backend = parallelInfo.backend;
         latentAttentionParam.selfOutLinearTensorParallelInfo.hcommInfo = parallelInfo.hcclComm;
         latentAttentionParam.selfOutLinearTensorParallelInfo.commDomain = parallelInfo.commDomain;
     }
     return atb::NO_ERROR;
 }
 
 void SetRmsNormParam(
     atb_speed::deepseekV2::LatentAttentionParam<atb::infer::RmsNormParam> &latentAttentionParam,
     const DecoderLayerParam &param)
 {
     atb::infer::RmsNormParam attenRmsNormParam;
     attenRmsNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
     attenRmsNormParam.normParam.epsilon = param.normEps;
     latentAttentionParam.normParamType = attenRmsNormParam;
 }
 
 void SetRmsNormQuantParam(
     atb_speed::deepseekV2::LatentAttentionParam<atb::infer::RmsNormParam> &latentAttentionParam,
     const DecoderLayerParam &param)
 {
     atb::infer::RmsNormParam attenRmsNormQuantParam;
     attenRmsNormQuantParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
     attenRmsNormQuantParam.normParam.epsilon = param.normEps;
     attenRmsNormQuantParam.normParam.quantType = atb::infer::QUANT_INT8;
     latentAttentionParam.normQuantParamType = attenRmsNormQuantParam;
 }
 
 void SetAttnInnerSpParam(atb_speed::deepseekV2::LatentAttentionParam<atb::infer::RmsNormParam> &latentAttentionParam,
     const DecoderLayerParam &param)
 {
     if (param.mapping.Get(base::ATTN_INNER_SP).IsEnabled()) {
         latentAttentionParam.hasAttnInnerSp = param.mapping.Get(base::ATTN_INNER_SP).IsEnabled();
         latentAttentionParam.attnSpRank = param.mapping.Get(base::ATTN_INNER_SP).rank;
         latentAttentionParam.attnSpSize = param.mapping.Get(base::ATTN_INNER_SP).rankIds.size();
         latentAttentionParam.attnSpDomain = param.mapping.Get(base::ATTN_INNER_SP).commDomain;
         latentAttentionParam.attnSpRankTableFile = "";
         latentAttentionParam.attnSpBackend = param.mapping.Get(base::ATTN_INNER_SP).backend;
         latentAttentionParam.attnSpHcclComm = param.mapping.Get(base::ATTN_INNER_SP).hcclComm;
 
         latentAttentionParam.pageAttentionParam.headNum = \
             param.numAttentionHeadsPerRank * param.mapping.Get(base::ATTN_INNER_SP).rankIds.size();
     }
 }
 
 atb::Status SetLatentAttentionParam(
     atb_speed::deepseekV2::LatentAttentionParam<atb::infer::RmsNormParam> &latentAttentionParam,
     const DecoderLayerParam &param)
 {
     SetRmsNormParam(latentAttentionParam, param);
     SetRmsNormQuantParam(latentAttentionParam, param);
 
     latentAttentionParam.isGroupedQueryAttention = param.numAttentionHeadsPerRank != param.numKeyValueHeadsPerRank;
     latentAttentionParam.isBF16 = param.isBF16;
     latentAttentionParam.attnLinearQuantType = param.attnLinearQuantType;
     latentAttentionParam.packQuantType = param.packQuantType.at(0);
     latentAttentionParam.quantGroupSize = param.quantGroupSize;
     latentAttentionParam.attnLinearTransposeType = param.attnLinearTransposeType;
     latentAttentionParam.enableLcoc = param.enableLcoc;
     latentAttentionParam.qLoraRank = param.qLoraRank;
     latentAttentionParam.headNum = param.headNum;
     latentAttentionParam.qkNopeHeadDim = param.qkNopeHeadDim;
     latentAttentionParam.qkRopeHeadDim = param.qkRopeHeadDim;
     latentAttentionParam.kvLoraRank = param.kvLoraRank;
     latentAttentionParam.rotaryType = atb_speed::common::RotaryType::ALL_ROTARY;
     latentAttentionParam.ropeParam.rotaryCoeff = 2; // 2:设置新张量形状
     latentAttentionParam.isFA = param.isFA;
     latentAttentionParam.isPrefill = param.isPrefill;
     latentAttentionParam.headDim = param.hiddenSizePerAttentionHead;
     latentAttentionParam.selfAttentionParam.headNum = param.numAttentionHeadsPerRank;
     latentAttentionParam.selfAttentionParam.kvHeadNum = param.numAttentionHeadsPerRank;
     CHECK_PARAM_GT(param.hiddenSizePerAttentionHead, 0);
     latentAttentionParam.selfAttentionParam.qkScale = param.softmaxScale;
     latentAttentionParam.selfAttentionParam.isTriuMask = param.isPrefill ? 1 : 0;
     if (param.isFA) {
         latentAttentionParam.selfAttentionParam.calcType = param.isPrefill ? \
             atb::infer::SelfAttentionParam::CalcType::ENCODER : atb::infer::SelfAttentionParam::CalcType::DECODER;
     } else {
         latentAttentionParam.selfAttentionParam.calcType = atb::infer::SelfAttentionParam::CalcType::PA_ENCODER;
     }
     latentAttentionParam.selfAttentionParam.maskType = atb::infer::SelfAttentionParam::MaskType::MASK_TYPE_NORM;
     latentAttentionParam.pageAttentionParam.headNum = param.numAttentionHeadsPerRank;
     latentAttentionParam.pageAttentionParam.kvHeadNum = 1;
     latentAttentionParam.pageAttentionParam.mlaVHeadSize = param.kvLoraRank;
     latentAttentionParam.pageAttentionParam.qkScale = param.softmaxScale;
     latentAttentionParam.pageAttentionParam.maskType = atb::infer::PagedAttentionParam::MaskType::UNDEFINED;
     latentAttentionParam.enableMlaPreprocess = param.enableMlaPreprocess;
     if (param.enableSpeculate) {
         if (param.maskfree) {
             latentAttentionParam.pageAttentionParam.maskType = \
                 atb::infer::PagedAttentionParam::MaskType::MASK_TYPE_NORM;
         } else {
             latentAttentionParam.pageAttentionParam.maskType = \
                 atb::infer::PagedAttentionParam::MaskType::MASK_TYPE_SPEC;
         }
         latentAttentionParam.pageAttentionParam.calcType = atb::infer::PagedAttentionParam::CalcType::CALC_TYPE_SPEC;
     }
     if (param.enableFA3 && param.enableKvQuantLayer) {
         latentAttentionParam.reshapeCacheParm.kvCacheCfg = \
             atb::infer::ReshapeAndCacheParam::KvCacheCfg::K_CACHE_V_CACHE_NZ;
         latentAttentionParam.pageAttentionParam.quantType = atb::infer::PagedAttentionParam::TYPE_QUANT_QKV_ONLINE;
         latentAttentionParam.pageAttentionParam.outDataType = param.isBF16 ? ACL_BF16 : ACL_FLOAT16;
     } else if (param.isNzCache) {
         latentAttentionParam.reshapeCacheParm.kvCacheCfg = \
             atb::infer::ReshapeAndCacheParam::KvCacheCfg::K_CACHE_V_CACHE_NZ;
     } else {
         latentAttentionParam.reshapeCacheParm.kvCacheCfg = \
             atb::infer::ReshapeAndCacheParam::KvCacheCfg::K_CACHE_V_CACHE;
     }
     latentAttentionParam.isNzCache = param.isNzCache;
     // This function must be called after the pageAttentionParam is set. It will change pageAttentionParam.headNum
     SetAttnInnerSpParam(latentAttentionParam, param);
     SetLatentAttentionInnerCommParam(latentAttentionParam, param);
     latentAttentionParam.enableQkvdownDp = param.enableQkvdownDp && param.layerId > param.firstKDenseReplace;
     latentAttentionParam.layerId = param.layerId;
     latentAttentionParam.firstKDenseReplace = param.firstKDenseReplace;
     latentAttentionParam.hasAttnComm = param.hasAttnComm;
     latentAttentionParam.attnTpRank = param.mapping.Get(base::ATTN_TP).rank;
     latentAttentionParam.attnTpSize = param.mapping.Get(base::ATTN_TP).rankIds.size();
     latentAttentionParam.attnTpBackend = param.mapping.Get(base::ATTN_TP).backend;
     latentAttentionParam.attnTpDomain = param.mapping.Get(base::ATTN_TP).commDomain;
     latentAttentionParam.attnTpRankTableFile = "";
     latentAttentionParam.hcclComm = param.mapping.Get(base::ATTN_TP).hcclComm;
     latentAttentionParam.ffnAllGather = param.ffnAllGather;
     latentAttentionParam.hasFfnComm = param.hasFfnComm;
     return atb::NO_ERROR;
 }
 
 int64_t SetAttention(atb::GraphParam &opGraph, const DecoderLayerParam &param,
                      std::map<std::string, uint32_t> &tensorMap)
 {
     atb::Node attentionNode;
     atb_speed::deepseekV2::LatentAttentionParam<atb::infer::RmsNormParam> latentAttentionParam;
     SetLatentAttentionParam(latentAttentionParam, param);
     CHECK_OPERATION_STATUS_RETURN(Attention(latentAttentionParam, &attentionNode.operation));
     std::vector<std::string> attnInTensorNames = {
         "in_hidden_states",
         "in_input_norm_weight", "in_input_norm_bias", "in_input_norm_new_weight", "in_input_norm_new_bias",
         "in_q_proj_a_weight", "in_q_proj_a_bias", "in_q_proj_a_descale",
         "in_q_proj_a_offset", "in_q_proj_a_scale", "in_q_proj_a_compress_idx",
         "in_q_proj_a_layernorm_weight", "in_q_proj_a_layernorm_bias",
         "in_q_proj_b_weight", "in_q_proj_b_bias", "in_q_proj_b_descale",
         "in_q_proj_b_offset", "in_q_proj_b_scale", "in_q_proj_b_compress_idx",
         "in_kv_proj_with_mqa_weight", "in_kv_proj_with_mqa_bias", "in_kv_proj_with_mqa_descale",
         "in_kv_proj_with_mqa_offset", "in_kv_proj_with_mqa_scale", "in_kv_proj_with_mqa_compress_idx",
         "in_kv_proj_a_layernorm_weight", "in_kv_proj_a_layernorm_bias",
         "in_k_proj_b_for_q_weight", "in_k_proj_b_for_q_bias", "in_k_proj_b_for_q_descale",
         "in_k_proj_b_for_q_offset", "in_k_proj_b_for_q_scale", "in_k_proj_b_for_q_compress_idx",
         "in_v_proj_b_for_o_weight", "in_v_proj_b_for_o_bias", "in_v_proj_b_for_o_descale",
         "in_v_proj_b_for_o_offset", "in_v_proj_b_for_o_scale", "in_v_proj_b_for_o_compress_idx",
         "in_attention_out_weight", "in_attention_out_bias", "in_attention_out_descale",
         "in_attention_out_offset", "in_attention_out_scale", "in_attention_out_compress_idx",
         "in_cos_table", "in_sin_table", "in_seq_len", "in_k_cache", "in_k_rope_cache",
         "in_attention_mask", "in_q_len", "in_token_offset", "in_layer_id", "in_block_tables",
         "in_slots", "in_attn_padding_idx"
     };
     if (param.enableFA3 && param.enableKvQuantLayer) {
         atb_speed::common::AddTensorToList(GetDeepseekV2LayerInTensorCandidates(), "fa3_quant", attnInTensorNames);
     }
     if (param.enableQkvdownDp && param.layerId > param.firstKDenseReplace) {
         // h3p qkvdown dp特性和sp特性同时开启时，需要与latent_attention.cpp模块中两个特性的inTensor顺序一致，
         // h3p qkvdown dp的inTensor需在sp的inTensor之前
         attnInTensorNames.push_back("in_ffn_unpadding_idx");
     }
     if (param.mapping.Get(base::ATTN_INNER_SP).IsEnabled()) {
         if (param.isPrefill) {
             attnInTensorNames.push_back("in_k_sp_gather_indices");
         } else {
             attnInTensorNames.push_back("in_seq_len_sp");
         }
     }
     attentionNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, attnInTensorNames);
     attentionNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_attention_out"});
     opGraph.nodes.push_back(attentionNode);
     ATB_SPEED_LOG_DEBUG("Attention calculation success");
     return atb::NO_ERROR;
 }
 
 atb::Status SetPadding(atb::GraphParam &opGraph, std::map<std::string, uint32_t> tensorMap)
 {
     atb::Node gatherNode;
     atb::infer::GatherParam gatherParam;
     CHECK_OPERATION_STATUS_RETURN(CreateOperation(gatherParam, &gatherNode.operation));
 
     gatherNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_attention_out",
                                                                              "in_attn_padding_idx"});
     gatherNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_attention_out_padding"});
 
     opGraph.nodes.push_back(gatherNode);
     ATB_SPEED_LOG_DEBUG("create SetPadding");
     return atb::NO_ERROR;
 }
 
 atb::Status SetAttnReduceScatter(atb::GraphParam &opGraph, const DecoderLayerParam &param,
     std::map<std::string, uint32_t> tensorMap)
 {
     atb::Node rsNode;
     atb::infer::ReduceScatterParam rsParam;
     atb_speed::common::ParallelInfo parallelInfo = param.mapping.Get(base::ATTN_TP);
     rsParam.rank = parallelInfo.rank;
     rsParam.rankSize = parallelInfo.rankIds.size();
     rsParam.backend = parallelInfo.backend;
     rsParam.hcclComm = parallelInfo.hcclComm;
     rsParam.commDomain = parallelInfo.commDomain;
     CHECK_OPERATION_STATUS_RETURN(CreateOperation(rsParam, &rsNode.operation));
     rsNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_attention_out_padding"});
     rsNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_attention_out_scatter"});
     opGraph.nodes.push_back(rsNode);
     return atb::NO_ERROR;
 }
 
 atb::Status SetResidualPadding(atb::GraphParam &opGraph, std::map<std::string, uint32_t> tensorMap)
 {
     atb::Node gatherNode;
     atb::infer::GatherParam gatherParam;
     CHECK_OPERATION_STATUS_RETURN(CreateOperation(gatherParam, &gatherNode.operation));
 
     gatherNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"in_hidden_states",
                                                                              "in_attn_padding_idx"});
     gatherNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_hidden_states_padding"});
     opGraph.nodes.push_back(gatherNode);
     return atb::NO_ERROR;
 }
 
 atb::Status SetResidualSliceNode(atb::GraphParam &opGraph, const DecoderLayerParam &param,
     std::map<std::string, uint32_t> tensorMap)
 {
     atb::infer::SliceParam sliceParam;
     atb::Node sliceNode;
 
     sliceParam.offsets.resize(3); // 3: Slice offset dim
     sliceParam.offsets[0] = param.mapping.Get(base::ATTN_TP).rank;
     sliceParam.offsets[1] = 0;
     sliceParam.offsets[2] = 0; // 2: dim：2
 
     sliceParam.size.resize(3); // 3: Slice Size dim
     sliceParam.size[0] = 1;
     sliceParam.size[1] = -1;
     sliceParam.size[2] = -1; // 2: dim：2
     CreateOperation(sliceParam, &sliceNode.operation);
 
     sliceNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_hidden_states_padding"});
     sliceNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_hidden_states_scatter"});
 
     sliceNode.inTensorReshapeFuncs.resize(sliceNode.inTensorIds.size());
     sliceNode.inTensorReshapeFuncs[0] = [=] (const atb::Dims &oldShape, atb::Dims &newShape) {
         if (oldShape.dimNum == 2) { // 2: dimNum
             newShape.dimNum = 3; // 3: dimNum
             newShape.dims[0] = param.mapping.Get(base::ATTN_TP).rankIds.size();
             newShape.dims[1] = oldShape.dims[0] / param.mapping.Get(base::ATTN_TP).rankIds.size();
             newShape.dims[2] = oldShape.dims[1]; // 2: dim 2
         } else {
             newShape.dimNum = 3; // 3: dimNum
             newShape.dims[0] = param.mapping.Get(base::ATTN_TP).rankIds.size();
             newShape.dims[1] = oldShape.dims[0] * oldShape.dims[1] / param.mapping.Get(base::ATTN_TP).rankIds.size();
             newShape.dims[2] = oldShape.dims[2]; // 2: dim 2
         }
     };
     opGraph.nodes.push_back(sliceNode);
     return atb::NO_ERROR;
 }
 
 atb::Status SetSelfResidualAdd(atb::GraphParam &opGraph, const DecoderLayerParam &param,
     std::map<std::string, uint32_t> &tensorMap)
 {
     atb::Node selfResidualAddNode;
     atb::infer::ElewiseParam addParam;
     addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
     CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(addParam, &selfResidualAddNode.operation));
     if (param.attnReduceScatter) {
         selfResidualAddNode.inTensorIds = \
         atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_hidden_states_scatter",
                                                         "intermediate_attention_out_scatter"});
         selfResidualAddNode.inTensorReshapeFuncs.resize(selfResidualAddNode.inTensorIds.size());
         selfResidualAddNode.inTensorReshapeFuncs[0] = [=] (const atb::Dims &oldShape, atb::Dims &newShape) {
             newShape.dimNum = 2; // 2: dimNum
             newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
             newShape.dims[1] = oldShape.dims[2]; // 2: dim 2
         };
     } else {
         selfResidualAddNode.inTensorIds = \
         atb_speed::common::GetTensorIdxList(tensorMap, {"in_hidden_states", "intermediate_attention_out"});
     }
     selfResidualAddNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap,
         {"intermediate_attention_add_out"});
 
     opGraph.nodes.push_back(selfResidualAddNode);
     ATB_SPEED_LOG_DEBUG("SelfResidualAdd calculation success");
     return atb::NO_ERROR;
 }
 
 int64_t SetAllGather(atb::GraphParam &opGraph, const DecoderLayerParam &param,
                      std::map<std::string, uint32_t> &tensorMap)
 {
     atb::Node allGatherNode;
     atb::infer::AllGatherParam allGatherParam;
     if (param.attnReduceScatter) {
         atb_speed::common::ParallelInfo parallelInfo = param.mapping.Get(base::MLP_TP);
         allGatherParam.rank = parallelInfo.rank;
         allGatherParam.rankSize = parallelInfo.rankIds.size();
         allGatherParam.backend = parallelInfo.backend;
         allGatherParam.hcclComm = parallelInfo.hcclComm;
         allGatherParam.commDomain = parallelInfo.commDomain;
     } else {
         atb_speed::common::ParallelInfo parallelInfo = param.mapping.Get(base::ATTN_DP);
         allGatherParam.rank = parallelInfo.rank;
         allGatherParam.rankSize = parallelInfo.rankIds.size();
         allGatherParam.backend = parallelInfo.backend;
         allGatherParam.hcclComm = parallelInfo.hcclComm;
         allGatherParam.commDomain = parallelInfo.commDomain;
     }
     if (!FLAGS_enable_atb_comm_multiprocess) {
          allGatherParam.commMode = atb::infer::CommMode::COMM_MULTI_THREAD;
      }
     CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(allGatherParam, &allGatherNode.operation));
     allGatherNode.inTensorIds = atb_speed::common::GetTensorIdxList(
         tensorMap, {"intermediate_selfattention_norm_out_partial"});
     allGatherNode.outTensorIds = atb_speed::common::GetTensorIdxList(
         tensorMap, {"intermediate_dp_attn_out_all_with_padding"});
     opGraph.nodes.push_back(allGatherNode);
 
     ATB_SPEED_LOG_DEBUG("AllGather calculation success");
     return atb::NO_ERROR;
 }
 
 atb::Status SetAllGatherCCOverlap(atb::GraphParam &opGraph, const DecoderLayerParam &param)
 {
     if (param.enableSharedExpertOverlap) {
         if (!param.isPrefill || !param.enableGatingDp) {
             CHECK_OPERATION_STATUS_RETURN(atb_speed::common::CreateRecordWithoutNodeId(
                 opGraph, atb_speed::EventAction::POP, atb_speed::common::COMM_CONTROL));
             CHECK_OPERATION_STATUS_RETURN(atb_speed::common::CreateWaitWithoutNodeId(
                 opGraph, atb_speed::EventAction::POP, atb_speed::common::COMP_CONTROL));
         } else {
             CHECK_OPERATION_STATUS_RETURN(atb_speed::common::CreateRecordWithoutNodeId(
                 opGraph, atb_speed::EventAction::PUSH, atb_speed::common::COMM_CONTROL));
             CHECK_OPERATION_STATUS_RETURN(atb_speed::common::CreateWaitWithoutNodeId(
                 opGraph, atb_speed::EventAction::PUSH, atb_speed::common::COMP_CONTROL));
         }
     }
     ATB_SPEED_LOG_DEBUG("AllGather CCOverlap Event success");
     return atb::NO_ERROR;
 }
 
 int64_t SetAttnUnpadding(atb::GraphParam &opGraph, const DecoderLayerParam &param,
     std::map<std::string, uint32_t> &tensorMap)
 {
     atb::Node unpadNode;
     atb::infer::GatherParam unpadParam;
     CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(unpadParam, &unpadNode.operation));
     unpadNode.inTensorIds = atb_speed::common::GetTensorIdxList(
         tensorMap, {param.attnAllGather ? "intermediate_dp_attn_out_all_with_padding" :
             "intermediate_selfattention_norm_out_partial", "in_attn_unpadding_idx"});
     unpadNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap,
         {"intermediate_selfattention_norm_out"});
     unpadNode.inTensorReshapeFuncs.reserve(unpadNode.inTensorIds.size());
     unpadNode.inTensorReshapeFuncs.resize(unpadNode.inTensorIds.size());
     unpadNode.inTensorReshapeFuncs[0] = [=] (const atb::Dims &oldShape, atb::Dims &newShape) {
     newShape.dimNum = 2; // 2：新shape维度为2
         if (oldShape.dimNum == 3) { // 3：旧shape维度为3
             newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1]; // 0, 0, 1： 新shape前两维合轴
             newShape.dims[1] = oldShape.dims[2]; // 1, 2: 新shape最后一维不变
         } else {
             newShape.dims[0] = oldShape.dims[0];
             newShape.dims[1] = oldShape.dims[1]; // 1, 2: 新shape最后一维不变
         }
     };
     opGraph.nodes.push_back(unpadNode);
 
     ATB_SPEED_LOG_DEBUG("AllGather calculation success");
     return atb::NO_ERROR;
 }
 
 atb::Status SetNormQauntInTensors(
     std::vector<std::string> &selfNormInTensorNames,
     atb::infer::RmsNormParam &mlpRmsNormParam,
     atb::infer::RmsNormParam &mlpRmsNormQuantParam,
     const DecoderLayerParam &param,
     atb::Node &selfNormNode)
 {
     if (param.mlpNormQuantType == atb::infer::QUANT_INT8) { // w8a8
         CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(mlpRmsNormQuantParam, &selfNormNode.operation));
         if (param.isAntiOutlier) {
             selfNormInTensorNames.push_back("in_selfattention_out_new_norm_weight");
             selfNormInTensorNames.push_back("in_selfattention_out_new_norm_bias");
         } else {
             selfNormInTensorNames.push_back("in_selfattention_out_norm_weight");
             selfNormInTensorNames.push_back("in_selfattention_out_norm_bias");
         }
     } else if (param.normHasBias) { // FP
         CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(mlpRmsNormParam, &selfNormNode.operation));
         selfNormInTensorNames.push_back("in_selfattention_out_norm_weight");
         selfNormInTensorNames.push_back("in_selfattention_out_new_norm_bias");
     } else {
         CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(mlpRmsNormParam, &selfNormNode.operation));
         if (param.isAntiOutlier) {
             selfNormInTensorNames.push_back("in_selfattention_out_new_norm_weight");
         } else {
             selfNormInTensorNames.push_back("in_selfattention_out_norm_weight");
         }
     }
     return atb::NO_ERROR;
 }
 
 int64_t SetSelfNorm(atb::GraphParam &opGraph, const DecoderLayerParam &param,
                     std::map<std::string, uint32_t> &tensorMap)
 {
     atb::Node selfNormNode;
     atb::infer::RmsNormParam mlpRmsNormParam;
     atb::infer::RmsNormParam mlpRmsNormQuantParam;
     std::vector<std::string> selfNormInTensorNames;
     std::vector<std::string> selfNormOutTensorNames;
     if (!param.attnAllreduce && param.hasAttnComm) {
         selfNormOutTensorNames.push_back("intermediate_selfattention_norm_out_partial");
     } else {
         selfNormOutTensorNames.push_back("intermediate_selfattention_norm_out");
     }
 
     if (param.enableIntraLayerAddNorm) {
         mlpRmsNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_PRENORM;
         mlpRmsNormParam.preNormParam.epsilon = param.normEps;
         mlpRmsNormQuantParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_PRENORM;
         mlpRmsNormQuantParam.preNormParam.epsilon = param.normEps;
         mlpRmsNormQuantParam.preNormParam.quantType = atb::infer::QUANT_INT8;
         if (param.attnReduceScatter) { // 2: Dynamic EP
             selfNormInTensorNames.push_back("intermediate_hidden_states_scatter");
             selfNormOutTensorNames.push_back("intermediate_attention_add_out");
         } else {
             selfNormInTensorNames.push_back("in_hidden_states");
             selfNormOutTensorNames.push_back("intermediate_attention_add_out");
         }
     } else {
         mlpRmsNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
         mlpRmsNormParam.normParam.epsilon = param.normEps;
         mlpRmsNormQuantParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
         mlpRmsNormQuantParam.normParam.epsilon = param.normEps;
         mlpRmsNormQuantParam.normParam.quantType = atb::infer::QUANT_INT8;
         selfNormInTensorNames.push_back("intermediate_attention_add_out");
     }
 
     SetNormQauntInTensors(selfNormInTensorNames, mlpRmsNormParam, mlpRmsNormQuantParam, param, selfNormNode);
     selfNormNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, selfNormInTensorNames);
     selfNormNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, selfNormOutTensorNames);
     opGraph.nodes.push_back(selfNormNode);
     ATB_SPEED_LOG_DEBUG("SelfNorm calculation success");
     return atb::NO_ERROR;
 }
 
 int64_t SetMlpExpert(atb::GraphParam &opGraph, const DecoderLayerParam &param,
                      std::map<std::string, uint32_t> &tensorMap)
 {
     atb::Node mlpExpertNode;
     atb_speed::common::SharedExpertParam mlpExpertParam;
     mlpExpertParam.isBF16 = param.isBF16;
     mlpExpertParam.transposeGateup = param.mlpLinearTransposeType[MLP_GATEUP_LINEAR_INDEX];
     mlpExpertParam.transposeDown = param.mlpLinearTransposeType[MLP_DOWN_LINEAR_INDEX];
     mlpExpertParam.hasSharedExpertGate = false;
     mlpExpertParam.mlpLinearQuantType = param.mlpLinearQuantType;
     mlpExpertParam.mlpLinearTransposeType = param.mlpLinearTransposeType;
     mlpExpertParam.packQuantType = param.packQuantType.at(1);
     mlpExpertParam.quantGroupSize = param.quantGroupSize;
     mlpExpertParam.enableSwiGLUQuantForSharedExperts = param.enableSwiGLUQuantForSharedExperts;
     atb_speed::common::CreateSharedExpertOperation(mlpExpertParam, &mlpExpertNode.operation);
     std::vector<std::string> mlpExpertInTensorNames = {
         "intermediate_selfattention_norm_out",
         "in_mlp_gateup_weight_shared_expert", "in_mlp_gateup_bias_shared_expert",
         "in_mlp_gateup_descale_shared_expert", "in_mlp_gateup_offset_shared_expert",
         "in_mlp_gateup_scale_shared_expert", "in_mlp_gateup_compress_idx_shared_expert",
         "in_mlp_down_weight_shared_expert", "in_mlp_down_bias_shared_expert",
         "in_mlp_down_descale_shared_expert", "in_mlp_down_offset_shared_expert",
         "in_mlp_down_scale_shared_expert", "in_mlp_down_compress_idx_shared_expert",
         "in_shared_expert_gate_weight", "in_shared_expert_gate_bias", "in_shared_expert_gate_descale",
         "in_shared_expert_gate_offset", "in_shared_expert_gate_scale", "in_shared_expert_gate_compress_idx"
     };
     mlpExpertNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, mlpExpertInTensorNames);
     mlpExpertNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_moe_out_with_shared"});
     opGraph.nodes.push_back(mlpExpertNode);
     ATB_SPEED_LOG_DEBUG("mlp expert calculation success");
     return atb::NO_ERROR;
 }
 
 int64_t SetExpertRoutingMapSlice(
     atb::GraphParam &opGraph,
     const DecoderLayerParam &param, std::map<std::string,
     uint32_t> tensorMap)
 {
     atb::Node sliceNode;
     atb::infer::SliceParam sliceParam;
     sliceParam.offsets.resize(NUM2);
     sliceParam.offsets[0] = param.layerId - param.firstKDenseReplace;
     sliceParam.offsets[1] = 0;
     sliceParam.size.resize(NUM2);
     sliceParam.size[0] = 1;
     sliceParam.size[1] = -1;
     CreateOperation(sliceParam, &sliceNode.operation);
     sliceNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"in_expert_routing_map"});
     sliceNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_expert_routing_map"});
     opGraph.nodes.push_back(sliceNode);
     return atb::NO_ERROR;
 }
 
 atb::Status SetSparseMoeParam(atb_speed::common::SparseMoeParam &sparseMoeParam, const DecoderLayerParam &param)
 {
     sparseMoeParam.isBF16 = param.isBF16;
     sparseMoeParam.gateUpTransposeB = param.moeLinearTransposeType[MOE_GATEUP_LINEAR_INDEX];
     sparseMoeParam.downTransposeB = param.moeLinearTransposeType[MOE_DOWN_LINEAR_INDEX];
     sparseMoeParam.numOfExperts = param.numOfExperts;
     sparseMoeParam.numOfDeviceExperts = param.numOfDeviceExperts;
     sparseMoeParam.num = param.numOfSelectedExperts;
     sparseMoeParam.routingMethod = param.routingMethod;
     sparseMoeParam.numOfGroups = param.numOfGroups;
     sparseMoeParam.topkGroups = param.topkGroups;
     sparseMoeParam.expertParallelDegree = param.expertParallelDegree;
     sparseMoeParam.isDynamicEp = param.isDynamicEp;
     sparseMoeParam.deviceExpert = param.deviceExpert;
     sparseMoeParam.routedScalingFactor = param.routedScalingFactor;
     sparseMoeParam.processLogits = param.processLogits;
     sparseMoeParam.moeLinearQuantType = param.moeLinearQuantType;
     if (param.moePackQuantType == atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED) {
         sparseMoeParam.packQuantType = param.packQuantType.at(1);
     } else {
         sparseMoeParam.packQuantType = param.moePackQuantType;
     }
     sparseMoeParam.quantGroupSize = param.quantGroupSize;
     sparseMoeParam.enableFusedRouting = param.enableFusedRouting;
     sparseMoeParam.enableInitQuant = param.enableInitQuant;
     sparseMoeParam.enableSwigluQuant = param.enableSwigluQuant;
     sparseMoeParam.enableFusedTopk = param.enableFusedTopk;
     sparseMoeParam.enableExpertCumSumOutput = param.enableExpertCumSumOutput;
     
     sparseMoeParam.enableGMMSwigluQuant = param.enableGMMSwigluQuant;
     sparseMoeParam.enableCVOverlap = param.enableCVOverlap;
     sparseMoeParam.enableLoadBalance = param.enableLoadBalance;
     sparseMoeParam.enableEPWB = param.enableEPWB;
     sparseMoeParam.numOfRedundantExpert = param.numOfRedundantExpert;
     sparseMoeParam.maxDecodeDpTokenSize = param.maxDecodeDpTokenSize;
     sparseMoeParam.enableMoeDistribute = !param.isPrefill && param.enableAllToAllMC2 && param.isDynamicEp;
     sparseMoeParam.enableGatingDp = param.enableGatingDp && param.isPrefill;  // h3p gatingdp for moe
     sparseMoeParam.enableGatingShift = param.enableGatingDp && !param.isPrefill;  // h3p gatingshift for decode
     sparseMoeParam.enableGatingOverlap = sparseMoeParam.enableGatingDp &&
                                         param.enableSharedExpertOverlap;  // h3p Gating overlap
     return atb::NO_ERROR;
 }
 
 atb::Status SetSparseMoeCommParam(atb_speed::common::SparseMoeParam &sparseMoeParam, const DecoderLayerParam &param)
 {
     sparseMoeParam.hcclComm = param.mapping.Get(base::MOE_EP).hcclComm;
     sparseMoeParam.backend = param.mapping.Get(base::MOE_EP).backend;
     
     sparseMoeParam.hasMoeEp = param.mapping.Get(base::MOE_EP).IsEnabled();
     sparseMoeParam.moeEpRank = param.mapping.Get(base::MOE_EP).rank;
     sparseMoeParam.moeEpSize = param.mapping.Get(base::MOE_EP).rankIds.size();
     sparseMoeParam.moeEpDomain = param.mapping.Get(base::MOE_EP).commDomain;
     sparseMoeParam.moeEpRankTableFile = "";
     
     sparseMoeParam.hcclTpComm = param.mapping.Get(base::MLP_TP).hcclComm;
     sparseMoeParam.mlpTpBackend = param.mapping.Get(base::MLP_TP).backend;
     sparseMoeParam.hasMlpTp = param.mapping.Get(base::MLP_TP).IsEnabled();
     sparseMoeParam.mlpTpRank = param.mapping.Get(base::MLP_TP).rank;
     sparseMoeParam.mlpTpSize = param.mapping.Get(base::MLP_TP).rankIds.size();
     sparseMoeParam.mlpTpDomain = param.mapping.Get(base::MLP_TP).commDomain;
     sparseMoeParam.mlpTpRankTableFile = "";
     
     if (sparseMoeParam.enableMoeDistribute) {
         sparseMoeParam.moeEpDomain = param.dispatchAndCombinecommDomain;
         sparseMoeParam.hcclComm = param.dispatchAndCombineHcclComm;
     }
     return atb::NO_ERROR;
 }
 
 int64_t SetMoe(atb::GraphParam &opGraph, const DecoderLayerParam &param, std::map<std::string, uint32_t> tensorMap)
 {
     atb::Node moeNode;
     atb_speed::common::SparseMoeParam sparseMoeParam;
     SetSparseMoeParam(sparseMoeParam, param);
     SetSparseMoeCommParam(sparseMoeParam, param);
     CHECK_OPERATION_STATUS_RETURN(atb_speed::common::CreateSparseMoeOperation(sparseMoeParam, &moeNode.operation));
     std::vector<std::string> moeInTensorNames;
     moeInTensorNames = std::vector<std::string>{
         "intermediate_selfattention_norm_out", "in_block_sparse_moe_gate_weight",
         "in_block_sparse_moe_gate_bias", "in_block_sparse_moe_gate_descale", "in_block_sparse_moe_gate_offset",
         "in_block_sparse_moe_gate_scale", "in_block_sparse_moe_gate_compress_idx",
         "in_mlp_gateup_weight_expert", "in_mlp_gateup_bias_expert", "in_mlp_gateup_descale_expert",
         "in_mlp_gateup_offset_expert", "in_mlp_gateup_scale_expert", "in_mlp_gateup_compress_idx_expert",
         "in_mlp_down_weight_expert", "in_mlp_down_bias_expert", "in_mlp_down_descale_expert",
         "in_mlp_down_offset_expert", "in_mlp_down_scale_expert", "in_mlp_down_compress_idx_expert",
         "in_expert_array", "in_expert_group", "in_one_hot", "in_zero_hot",
     };
     if (param.hasP2DWeight && !param.isPrefill) {
         for (int i = 1; i < 7; i++) {  // 7: 需要更改的最后一个变量
             moeInTensorNames[i] += "_shuffled";
         }
     }
     if (param.enableLoadBalance) {
         moeInTensorNames.push_back("in_fake_topk");
     }
     if (param.mapping.Get(base::MOE_EP).IsEnabled()) {
         moeInTensorNames.push_back("in_start_expert_idx");
         moeInTensorNames.push_back("in_device_expert_count");
         moeInTensorNames.push_back("in_ffn_padding_idx");
         if (param.isDynamicEp) {
             moeInTensorNames.push_back("in_lty_idx");
             moeInTensorNames.push_back("in_moe_idx");
         }
     }
     if (param.enableEPWB) {
         SetExpertRoutingMapSlice(opGraph, param, tensorMap);
         moeInTensorNames.push_back("intermediate_expert_routing_map");
     }
     // h3p gatingdp prefill add intensor partial
     if (param.enableGatingDp && param.isPrefill) {
         moeInTensorNames.push_back("intermediate_selfattention_norm_out_partial");
         moeInTensorNames.push_back("in_attn_unpadding_idx");
     }
     moeNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, moeInTensorNames);
     moeNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_moe_out_with_shared"});
     if (param.enableExpertCumSumOutput) {
         moeNode.outTensorIds.push_back(atb_speed::common::GetTensorIdx(tensorMap, {"out_gmm_cumsum_list"}));
     }
     opGraph.nodes.push_back(moeNode);
     ATB_SPEED_LOG_DEBUG("Moe sparse calculation success");
     return atb::NO_ERROR;
 }
 
 atb::Status SetSharedExpertParam(atb_speed::common::SharedExpertParam &sharedExpertParam,
                                  const DecoderLayerParam &param)
 {
     sharedExpertParam.isBF16 = param.isBF16;
     sharedExpertParam.transposeGateup = param.mlpLinearTransposeType[MLP_GATEUP_LINEAR_INDEX];
     sharedExpertParam.transposeDown = param.mlpLinearTransposeType[MLP_DOWN_LINEAR_INDEX];
     sharedExpertParam.hasSharedExpertGate = param.hasSharedExpertGate;
     sharedExpertParam.mlpLinearQuantType = param.mlpLinearQuantType;
     sharedExpertParam.mlpLinearTransposeType = param.mlpLinearTransposeType;
     sharedExpertParam.quantGroupSize = param.quantGroupSize;
     sharedExpertParam.packQuantType = param.packQuantType.at(1);
     sharedExpertParam.enableCVOverlap = param.enableCVOverlap;
     sharedExpertParam.enableSwiGLUQuantForSharedExperts = param.enableSwiGLUQuantForSharedExperts;
     return atb::NO_ERROR;
 }
 
 int64_t SetSharedExpert(atb::GraphParam &opGraph, const DecoderLayerParam &param,
                         std::map<std::string, uint32_t> tensorMap)
 {
     atb::Node sharedExpertNode;
     atb_speed::common::SharedExpertParam sharedExpertParam;
     SetSharedExpertParam(sharedExpertParam, param);
     atb_speed::common::CreateSharedExpertOperation(sharedExpertParam, &sharedExpertNode.operation);
     if (param.hasP2DWeight && !param.isPrefill) {
         std::vector<std::string> sharedExpertInTensorNames = {
             "intermediate_selfattention_norm_out",
             "in_mlp_gateup_weight_shared_expert_tp", "in_mlp_gateup_bias_shared_expert_tp",
             "in_mlp_gateup_descale_shared_expert_tp", "in_mlp_gateup_offset_shared_expert_tp",
             "in_mlp_gateup_scale_shared_expert_tp", "in_mlp_gateup_compress_idx_shared_expert_tp",
             "in_mlp_down_weight_shared_expert_tp", "in_mlp_down_bias_shared_expert_tp",
             "in_mlp_down_descale_shared_expert_tp", "in_mlp_down_offset_shared_expert_tp",
             "in_mlp_down_scale_shared_expert_tp", "in_mlp_down_compress_idx_shared_expert_tp",
             "in_shared_expert_gate_weight_tp", "in_shared_expert_gate_bias_tp",
             "in_shared_expert_gate_descale_tp", "in_shared_expert_gate_offset_tp",
             "in_shared_expert_gate_scale_tp", "in_shared_expert_gate_compress_idx_tp"
         };
         sharedExpertNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, sharedExpertInTensorNames);
     } else {
         std::vector<std::string> sharedExpertInTensorNames = {
             "intermediate_selfattention_norm_out",
             "in_mlp_gateup_weight_shared_expert", "in_mlp_gateup_bias_shared_expert",
             "in_mlp_gateup_descale_shared_expert", "in_mlp_gateup_offset_shared_expert",
             "in_mlp_gateup_scale_shared_expert", "in_mlp_gateup_compress_idx_shared_expert",
             "in_mlp_down_weight_shared_expert", "in_mlp_down_bias_shared_expert",
             "in_mlp_down_descale_shared_expert", "in_mlp_down_offset_shared_expert",
             "in_mlp_down_scale_shared_expert", "in_mlp_down_compress_idx_shared_expert",
             "in_shared_expert_gate_weight", "in_shared_expert_gate_bias", "in_shared_expert_gate_descale",
             "in_shared_expert_gate_offset", "in_shared_expert_gate_scale", "in_shared_expert_gate_compress_idx"
         };
         // h3p shared expert dp first intensor partial
         if (param.enableSharedExpertDp) {
             sharedExpertInTensorNames[0] = "intermediate_selfattention_norm_out_partial";
         }
         sharedExpertNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, sharedExpertInTensorNames);
     }
     sharedExpertNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_shared_expert_out"});
     if (param.enableCVOverlap) {
         // set extreme stream for moe cv parallel, stream id is 2
         atb::SetExecuteStreamId(sharedExpertNode.operation, 2);
     }
     if (param.enableSharedExpertOverlap) {
         atb::SetExecuteStreamId(sharedExpertNode.operation, STREAM1);
     }
     opGraph.nodes.push_back(sharedExpertNode);
     ATB_SPEED_LOG_DEBUG("Shared expert calculation success");
     return atb::NO_ERROR;
 }
 
 int64_t AddExpertAdd(atb::GraphParam &opGraph, const DecoderLayerParam &param,
                      std::map<std::string, uint32_t> tensorMap)
 {
     atb::Node expertAddNode;
     atb::infer::ElewiseParam addParam;
     addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
     CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(addParam, &expertAddNode.operation));
     expertAddNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_moe_out_with_shared",
                                                                                 "intermediate_shared_expert_out"});
     expertAddNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_moe_out_with_shared"});
     // h3p shared expert dp add routing expert out after reduce scatter
     if (param.enableSharedExpertDp) {
         expertAddNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_mlp_out",
                                                                                     "intermediate_shared_expert_out"});
         expertAddNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_mlp_out"});
     }
     opGraph.nodes.push_back(expertAddNode);
     ATB_SPEED_LOG_DEBUG("create add operation");
     return atb::NO_ERROR;
 }
 
 int64_t SetAllReduce(atb::GraphParam &opGraph, const DecoderLayerParam &param,
                      std::map<std::string, uint32_t> tensorMap)
 {
     atb::Node moeAllReduceNode;
     atb::infer::AllReduceParam allReduceParam;
     if (param.isDenseLayer) {
         atb_speed::common::ParallelInfo parallelInfo = param.mapping.Get(base::ATTN_TP);
         allReduceParam.rank = parallelInfo.rank;
         allReduceParam.rankSize = parallelInfo.rankIds.size();
         allReduceParam.backend = parallelInfo.backend;
         allReduceParam.hcclComm = parallelInfo.hcclComm;
         allReduceParam.commDomain = parallelInfo.commDomain;
     } else {
         atb_speed::common::ParallelInfo parallelInfo = param.mapping.Get(base::MLP_TP);
         allReduceParam.rank = parallelInfo.rank;
         allReduceParam.rankSize = parallelInfo.rankIds.size();
         allReduceParam.backend = parallelInfo.backend;
         allReduceParam.hcclComm = parallelInfo.hcclComm;
         allReduceParam.commDomain = parallelInfo.commDomain;
     }
 
     CreateOperation(allReduceParam, &moeAllReduceNode.operation);
     if (moeAllReduceNode.operation == nullptr) {
         ATB_SPEED_LOG_ERROR("moeAllReduceNode op is nullptr: ");
     }
     moeAllReduceNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap,
         {"intermediate_moe_out_with_shared"});
     moeAllReduceNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_mlp_out"});
     opGraph.nodes.push_back(moeAllReduceNode);
     ATB_SPEED_LOG_DEBUG("create all reduce");
     return atb::NO_ERROR;
 }
 
 atb::Status SetMlpResidualAdd(atb::GraphParam &opGraph, const DecoderLayerParam &param,
                               std::map<std::string, uint32_t> tensorMap)
 {
     atb::Node mlpResidualAddNode;
     atb::infer::ElewiseParam addParam;
     addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
     CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(addParam, &mlpResidualAddNode.operation));
     std::vector<std::string> mlpResidualAddInTensorNames = {"intermediate_attention_add_out",
         param.ffnAllreduce || param.ffnReduceScatter ?
         "intermediate_mlp_out" :
         ((param.hasAttnComm) && (param.hasFfnComm) ?
             "intermediate_moe_out_with_shared_with_padding" : "intermediate_moe_out_with_shared")};
     std::vector<std::string> mlpResidualAddOutTensorNames = {param.ffnAllGather || param.ffnReduceScatter ?
         ((param.enableQkvdownDp && !param.isLastLayer) ? "out_decoder_layer" : "intermediate_mlp_out") :
         "out_decoder_layer"};
     mlpResidualAddNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, mlpResidualAddInTensorNames);
     mlpResidualAddNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, mlpResidualAddOutTensorNames);
     opGraph.nodes.push_back(mlpResidualAddNode);
     ATB_SPEED_LOG_DEBUG("create mlpResidualAdd");
     return atb::NO_ERROR;
 }
 
 int64_t SetFFNPadding(atb::GraphParam &opGraph, const DecoderLayerParam &param,
     std::map<std::string, uint32_t> tensorMap)
 {
     atb::Node padNode;
     atb::infer::GatherParam padParam;
     atb::CreateOperation(padParam, &padNode.operation);
     padNode.inTensorIds = atb_speed::common::GetTensorIdxList(
         tensorMap, {param.hasAttnComm ?
         "intermediate_moe_out_with_shared" : "intermediate_mlp_out",
         param.hasAttnComm ?
         "in_ffn_padding_idx" : "in_attn_padding_idx"});
     padNode.outTensorIds = atb_speed::common::GetTensorIdxList(
         tensorMap, {"intermediate_moe_out_with_shared_with_padding"});
     opGraph.nodes.push_back(padNode);
     ATB_SPEED_LOG_DEBUG("create padNode");
     return atb::NO_ERROR;
 }
 
 int64_t SetMlpReduceScatter(atb::GraphParam &opGraph, const DecoderLayerParam &param,
     std::map<std::string, uint32_t> tensorMap)
 {
     atb::Node reduceScatterNode;
     atb::infer::ReduceScatterParam reduceScatterParam;
     atb_speed::common::ParallelInfo parallelInfo = param.mapping.Get(base::MLP_TP);
     reduceScatterParam.rank = parallelInfo.rank;
     reduceScatterParam.rankSize = parallelInfo.rankIds.size();
     reduceScatterParam.backend = parallelInfo.backend;
     reduceScatterParam.hcclComm = parallelInfo.hcclComm;
     reduceScatterParam.commDomain = parallelInfo.commDomain;
 
     CreateOperation(reduceScatterParam, &reduceScatterNode.operation);
     if (reduceScatterNode.operation == nullptr) {
         ATB_SPEED_LOG_ERROR("moeAllReduceNode op is nullptr: ");
     }
     reduceScatterNode.inTensorIds = atb_speed::common::GetTensorIdxList(
         tensorMap, {"intermediate_moe_out_with_shared_with_padding"});
     reduceScatterNode.outTensorIds = atb_speed::common::GetTensorIdxList(
         tensorMap, {"intermediate_mlp_out"});
     opGraph.nodes.push_back(reduceScatterNode);
     ATB_SPEED_LOG_DEBUG("create all reduce");
     return atb::NO_ERROR;
 }
 
 int64_t SetMlpReduceScatterNanToNum(atb::GraphParam &opGraph,
     std::map<std::string, uint32_t> tensorMap)
 {
     atb::Node nanToNumNode;
     atb_speed::common::AclNNNanToNumParam NanToNumParam;
     NanToNumParam.posInfValue = FLOAT16_MAX;  // replaces positive infinity values in tensor elements
     NanToNumParam.negInfValue = FLOAT16_MIN;  // replaces negative infinity values in tensor elements
     nanToNumNode.operation = new atb_speed::common::InplaceNanToNumOperation("nanToNumNode", NanToNumParam);
     nanToNumNode.inTensorIds = {atb_speed::common::GetTensorIdx(tensorMap, "intermediate_mlp_out")};
     nanToNumNode.outTensorIds = {atb_speed::common::GetTensorIdx(tensorMap, "intermediate_mlp_out")};
     opGraph.nodes.push_back(nanToNumNode);
     ATB_SPEED_LOG_DEBUG("create nan to num");
     return atb::NO_ERROR;
 }
 
 int64_t SetMlpResidualAddNanToNum(atb::GraphParam &opGraph, const DecoderLayerParam &param,
     std::map<std::string, uint32_t> tensorMap)
 {
     atb::Node nanToNumNode;
     atb_speed::common::AclNNNanToNumParam NanToNumParam;
     NanToNumParam.posInfValue = FLOAT16_MAX;  // replaces positive infinity values in tensor elements
     NanToNumParam.negInfValue = FLOAT16_MIN;  // replaces negative infinity values in tensor elements
     nanToNumNode.operation = new atb_speed::common::InplaceNanToNumOperation("nanToNumNode1", NanToNumParam);
     std::vector<std::string> mlpResidualAddOutTensorNames = {param.ffnAllGather || param.ffnReduceScatter ? \
                           ((param.enableQkvdownDp && !param.isLastLayer) ? "out_decoder_layer" : \
                            "intermediate_mlp_out") : "out_decoder_layer"};
 
     nanToNumNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, mlpResidualAddOutTensorNames);
     nanToNumNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, mlpResidualAddOutTensorNames);
     opGraph.nodes.push_back(nanToNumNode);
     ATB_SPEED_LOG_DEBUG("create nan to num");
     return atb::NO_ERROR;
 }
 
 atb::Status SetTPAllGatherNode(atb::GraphParam &opGraph, const DecoderLayerParam &param,
     std::map<std::string, uint32_t> tensorMap)
 {
     atb::Node allGatherNode;
     atb::infer::AllGatherParam allGatherParam;
     if (!param.isLastLayer || (param.isLastLayer && param.enableDpOut && !param.lmHeadLocalTp)) {
         atb_speed::common::ParallelInfo parallelInfo = param.mapping.Get(base::ATTN_TP);
         allGatherParam.rank = parallelInfo.rank;
         allGatherParam.rankSize = parallelInfo.rankIds.size();
         allGatherParam.backend = parallelInfo.backend;
         allGatherParam.hcclComm = parallelInfo.hcclComm;
         allGatherParam.commDomain = parallelInfo.commDomain;
     } else if (param.isLastLayer && param.enableDpOut && param.lmHeadLocalTp) {
         atb_speed::common::ParallelInfo parallelInfo = param.mapping.Get(base::LM_HEAD_TP);
         allGatherParam.rank = parallelInfo.rank;
         allGatherParam.rankSize = parallelInfo.rankIds.size();
         allGatherParam.backend = parallelInfo.backend;
         allGatherParam.hcclComm = parallelInfo.hcclComm;
         allGatherParam.commDomain = parallelInfo.commDomain;
     } else {
         atb_speed::common::ParallelInfo parallelInfo = param.mapping.Get(base::MLP_TP);
         allGatherParam.rank = parallelInfo.rank;
         allGatherParam.rankSize = parallelInfo.rankIds.size();
         allGatherParam.backend = parallelInfo.backend;
         allGatherParam.hcclComm = parallelInfo.hcclComm;
         allGatherParam.commDomain = parallelInfo.commDomain;
     }
    if (!FLAGS_enable_atb_comm_multiprocess) {
        allGatherParam.commMode = atb::infer::CommMode::COMM_MULTI_THREAD;
    }
     CreateOperation(allGatherParam, &allGatherNode.operation);
 
     allGatherNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap,
         {param.hasAttnComm ?
         "intermediate_mlp_out" : "intermediate_moe_out_with_shared_with_padding"});
     allGatherNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_mlp_out_all"});
 
     opGraph.nodes.push_back(allGatherNode);
     return atb::NO_ERROR;
 }
 
 atb::Status SetFFNUnPadding(atb::GraphParam &opGraph, const DecoderLayerParam &param,
     std::map<std::string, uint32_t> tensorMap)
 {
     atb::Node gatherNode;
     atb::infer::GatherParam gatherParam;
     CHECK_OPERATION_STATUS_RETURN(CreateOperation(gatherParam, &gatherNode.operation));
 
     gatherNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, \
         {param.ffnAllGather ? "intermediate_mlp_out_all" : "intermediate_mlp_out",
         (param.isLastLayer && (!param.enableDpOut || (param.enableDpOut && param.lmHeadLocalTp))) ? \
             "in_lm_head_skip_padding_token_indices" : "in_ffn_unpadding_idx"});
         gatherNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"out_decoder_layer"});
     // intermediate_layer_out
     if (param.ffnAllGather) {
         gatherNode.inTensorReshapeFuncs.resize(gatherNode.inTensorIds.size());
         gatherNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
                 newShape.dimNum = 2; // 2: dimNum
                 newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
                 newShape.dims[1] = oldShape.dims[2]; // 2: dim 2
         };
     }
     opGraph.nodes.push_back(gatherNode);
     return atb::NO_ERROR;
 }
 
 atb::Status CalculateDataPartition(DecoderLayerParam &param)
 {
     // ATTN
     param.attnStreamNum = param.mapping.Get(base::ATTN_DP).rankIds.size();
     // FFN
     if (param.isDenseLayer) {
         if (param.isMlpFullTP) {
             param.ffnStreamNum = 1;
         } else {
             param.ffnStreamNum = param.mapping.Get(base::ATTN_DP).rankIds.size();
         }
     } else {
         if (param.isDynamicEp) {
             param.ffnStreamNum = param.mapping.Get(base::MOE_EP).rankIds.size() *
             param.mapping.Get(base::MOE_TP).rankIds.size();
         } else {
             param.ffnStreamNum = 1; // 暂不支持MoE DP
         }
     }
     // Lmhead
     param.lmheadStreamNum = 1; // Lmhead DP使用
     ATB_SPEED_LOG_DEBUG("CalculateDataPartition done"
         << ". Attention Stream Num is " << param.attnStreamNum
         << " . FFN Stream Num is " << param.ffnStreamNum
         << " . lmheadStreamNum Stream Num is " << param.lmheadStreamNum);
     return atb::NO_ERROR;
 }
 
 atb::Status CalculateCommType(DecoderLayerParam &param)
 {
     if (param.worldSize == 1) {
         return atb::NO_ERROR;
     }
     int outStreamNum = (param.isLastLayer && (!param.enableDpOut || (param.enableDpOut && param.lmHeadLocalTp))) ? \
         param.lmheadStreamNum : param.attnStreamNum;
 
     param.attnAllreduce = param.mapping.Get(base::ATTN_TP).IsEnabled() &&
                             param.ffnStreamNum == param.attnStreamNum ? true : false;
 
     param.attnReduceScatter = !param.attnAllreduce && param.mapping.Get(base::ATTN_TP).IsEnabled() ? true : false;
 
     param.attnAllGather = (param.attnReduceScatter && param.worldSize > param.ffnStreamNum) || \
         (param.attnStreamNum > param.ffnStreamNum) ?
         true : false;
     param.ffnAllreduce = param.attnAllreduce && param.ffnStreamNum == param.attnStreamNum ? true : false;
 
     param.ffnReduceScatter = !param.ffnAllreduce && param.attnAllGather ? true : false;
 
     int ffnOutStreamNum = param.ffnReduceScatter ? param.mapping.worldSize_ : param.ffnStreamNum;
     param.ffnAllGather = ffnOutStreamNum > outStreamNum ? true : false;
 
     param.hasAttnComm = param.attnReduceScatter || param.attnAllGather;
     param.hasFfnComm = param.ffnReduceScatter || param.ffnAllGather;
     ATB_SPEED_LOG_DEBUG("CalculateCommType done"
         << ". outStreamNum is " << outStreamNum
         << ". attnAllreduce is " << param.attnAllreduce << " . attnReduceScatter is " << param.attnReduceScatter
         << " . attnAllGather is " << param.attnAllGather
         << " . ffnAllreduce is " << param.ffnAllreduce << " . ffnReduceScatter is " << param.ffnReduceScatter
         << " . ffnAllGather is " << param.ffnAllGather);
     return atb::NO_ERROR;
 }
 
 atb::Status CreateNewStreamRecordWithoutNodeId(atb::GraphParam &opGraph, atb_speed::EventAction eventAction,
     const std::string &cvKey)
 {
     atb::Node recordNode;
     recordNode.inTensorIds = {};
     recordNode.outTensorIds = {};
     CHECK_OPERATION_STATUS_RETURN(atb_speed::EventManager::GetInstance().RecordEvent(
         recordNode.operation,
         eventAction,
         cvKey));
     atb::SetExecuteStreamId(recordNode.operation, STREAM1);
     opGraph.nodes.push_back(recordNode);
     ATB_SPEED_LOG_DEBUG("Record event success");
     return atb::NO_ERROR;
 }
 
 atb::Status CreateNewStreamWaitWithoutNodeId(atb::GraphParam &opGraph, atb_speed::EventAction eventAction,
     const std::string &cvKey)
 {
     atb::Node waitNode;
     waitNode.inTensorIds = {};
     waitNode.outTensorIds = {};
     CHECK_OPERATION_STATUS_RETURN(atb_speed::EventManager::GetInstance().WaitEvent(
         waitNode.operation,
         eventAction,
         cvKey));
     atb::SetExecuteStreamId(waitNode.operation, STREAM1);
     opGraph.nodes.push_back(waitNode);
     ATB_SPEED_LOG_DEBUG("Wait event success");
     return atb::NO_ERROR;
 }
 
 atb::Status SetFFN(std::map<std::string, uint32_t> &tensorMap,
     const DecoderLayerParam &param, atb::GraphParam &opGraph)
 {
     if (param.isDenseLayer) {
         CHECK_OPERATION_STATUS_RETURN(SetMlpExpert(opGraph, param, tensorMap));
     } else {
         if (param.hasSharedExpert && !param.enableSharedExpertOverlap) {
             CHECK_OPERATION_STATUS_RETURN(SetSharedExpert(opGraph, param, tensorMap));
         }
         CHECK_OPERATION_STATUS_RETURN(SetMoe(opGraph, param, tensorMap));
         if (param.hasSharedExpert && !param.enableSharedExpertDp) {
             CHECK_OPERATION_STATUS_RETURN(AddExpertAdd(opGraph, param, tensorMap));
         }
     };
     return atb::NO_ERROR;
 }
 
 atb::Status SetCast(atb::GraphParam &opGraph, const DecoderLayerParam &param,
     std::map<std::string, uint32_t> &tensorMap)
 {
     atb::Node castNode;
     atb::infer::ElewiseParam castParam;
     castParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_CAST;
     castParam.outTensorType = param.isBF16 ? ACL_BF16 : ACL_FLOAT16;
     CHECK_OPERATION_STATUS_RETURN(CreateOperation(castParam, &castNode.operation));
     castNode.inTensorIds = {atb_speed::common::GetTensorIdx(tensorMap, "intermediate_selfattention_norm_out_fp32")};
     castNode.outTensorIds = {
         atb_speed::common::GetTensorIdx(tensorMap, param.enableExtraOprojTp ?
             "intermediate_selfattention_norm_out" : "intermediate_selfattention_norm_out_partial")};
 
     opGraph.nodes.push_back(castNode);
     ATB_SPEED_LOG_DEBUG("Cast calculation success");
     return atb::NO_ERROR;
 }
 
 atb::Status SetGatherPreNorm(atb::GraphParam &opGraph, const DecoderLayerParam &param,
     std::map<std::string, uint32_t> &tensorMap)
 {
     atb::Node gatherNormNode;
     atb::infer::GatherPreRmsNormParam gatherRmsNormParam;
     gatherRmsNormParam.epsilon = param.normEps;
     CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(gatherRmsNormParam, &gatherNormNode.operation));
 
     std::vector<std::string> outTensorNames;
     std::vector<std::string> inTensorNames;
 
     outTensorNames.push_back("intermediate_selfattention_norm_out_fp32");
     outTensorNames.push_back("intermediate_attention_add_out");
 
     if (param.enableExtraOprojTp) {
         inTensorNames.push_back("in_hidden_states");
         inTensorNames.push_back("intermediate_attention_out");
     } else {
         if (param.attnReduceScatter) {
             inTensorNames.push_back("intermediate_attention_out_scatter");
         } else {
             inTensorNames.push_back("intermediate_attention_out_padding");
         }
         inTensorNames.push_back("in_hidden_states");
     }
     inTensorNames.push_back("in_attention_padding_idx_slice");
 
     if (param.normHasBias) { // FP
         inTensorNames.push_back("in_selfattention_out_norm_weight");
         inTensorNames.push_back("in_selfattention_out_new_norm_bias");
     } else {
         if (param.isAntiOutlier) {
             inTensorNames.push_back("in_selfattention_out_new_norm_weight");
         } else {
             inTensorNames.push_back("in_selfattention_out_norm_weight");
         }
     }
 
     gatherNormNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, inTensorNames);
     gatherNormNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, outTensorNames);
     opGraph.nodes.push_back(gatherNormNode);
     ATB_SPEED_LOG_DEBUG("SelfNorm calculation success");
 
     return atb::NO_ERROR;
 }
 
 atb::Status SetPreNorm(atb::GraphParam &opGraph, const DecoderLayerParam &param,
     std::map<std::string, uint32_t> &tensorMap)
 {
     atb::Node preNormNode;
     atb::infer::RmsNormParam preRmsNormParam;
     preRmsNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_PRENORM;
     preRmsNormParam.preNormParam.epsilon = param.normEps;
     CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(preRmsNormParam, &preNormNode.operation));
     
     std::vector<std::string> outTensorNames;
     std::vector<std::string> inTensorNames;
 
     outTensorNames.push_back("intermediate_selfattention_norm_out_partial");
     outTensorNames.push_back("intermediate_attention_add_out");
 
     if (param.attnReduceScatter) {
         inTensorNames.push_back("intermediate_attention_out_scatter");
     } else {
         inTensorNames.push_back("intermediate_attention_out_padding");
     }
     inTensorNames.push_back("in_hidden_states");
 
     if (param.normHasBias) { // FP
         inTensorNames.push_back("in_selfattention_out_norm_weight");
         inTensorNames.push_back("in_selfattention_out_new_norm_bias");
     } else {
         if (param.isAntiOutlier) {
             inTensorNames.push_back("in_selfattention_out_new_norm_weight");
         } else {
             inTensorNames.push_back("in_selfattention_out_norm_weight");
         }
     }
 
     preNormNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, inTensorNames);
     preNormNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, outTensorNames);
     opGraph.nodes.push_back(preNormNode);
     ATB_SPEED_LOG_DEBUG("SelfNorm calculation success");
 
     return atb::NO_ERROR;
 }
 
 atb::Status SetPostAttnProcess(std::map<std::string, uint32_t> &tensorMap,
     const DecoderLayerParam &param, atb::GraphParam &opGraph)
 {
     if (param.hasAttnComm) {
         CHECK_OPERATION_STATUS_RETURN(SetPadding(opGraph, tensorMap));
         if (!param.enableGatherPreNorm) {
             CHECK_OPERATION_STATUS_RETURN(SetResidualPadding(opGraph, tensorMap));
             CHECK_OPERATION_STATUS_RETURN(SetResidualSliceNode(opGraph, param, tensorMap));
         }
     }
     if (param.attnReduceScatter) {
         CHECK_OPERATION_STATUS_RETURN(SetAttnReduceScatter(opGraph, param, tensorMap));
     }
     if ((param.hasAttnComm && param.enableGatherPreNorm) || param.enableExtraOprojTp) {
         // h3p qkvdown dp move moe allgather+gather to mla, without first moe
         if (param.enableQkvdownDp && param.layerId > param.firstKDenseReplace) {
             CHECK_OPERATION_STATUS_RETURN(SetPreNorm(opGraph, param, tensorMap));
         } else {
             CHECK_OPERATION_STATUS_RETURN(SetGatherPreNorm(opGraph, param, tensorMap));
             CHECK_OPERATION_STATUS_RETURN(SetCast(opGraph, param, tensorMap));
         }
     } else {
         if (!param.enableIntraLayerAddNorm) {
             CHECK_OPERATION_STATUS_RETURN(SetSelfResidualAdd(opGraph, param, tensorMap));
         }
         CHECK_OPERATION_STATUS_RETURN(SetSelfNorm(opGraph, param, tensorMap));
     }
     if (param.enableSharedExpertOverlap) {
         CHECK_OPERATION_STATUS_RETURN(atb_speed::common::CreateRecordWithoutNodeId(
             opGraph, atb_speed::EventAction::PUSH, atb_speed::common::CC_START));
         CHECK_OPERATION_STATUS_RETURN(CreateNewStreamWaitWithoutNodeId(
             opGraph, atb_speed::EventAction::POP, atb_speed::common::CC_START));
     }
     if (param.enableSharedExpertOverlap && !param.isDenseLayer && param.hasSharedExpert) {
         CHECK_OPERATION_STATUS_RETURN(SetSharedExpert(opGraph, param, tensorMap));
         if (!param.isPrefill || !param.enableGatingDp) {
             CHECK_OPERATION_STATUS_RETURN(CreateNewStreamRecordWithoutNodeId(
                 opGraph, atb_speed::EventAction::PUSH, atb_speed::common::COMP_CONTROL));
             CHECK_OPERATION_STATUS_RETURN(CreateNewStreamWaitWithoutNodeId(
                 opGraph, atb_speed::EventAction::PUSH, atb_speed::common::COMM_CONTROL));
         }
     }
     if (param.attnAllGather) {
         CHECK_OPERATION_STATUS_RETURN(SetAllGather(opGraph, param, tensorMap));
         CHECK_OPERATION_STATUS_RETURN(SetAllGatherCCOverlap(opGraph, param));
     }
     if (param.hasAttnComm) {
         CHECK_OPERATION_STATUS_RETURN(SetAttnUnpadding(opGraph, param, tensorMap));
     }
     return atb::NO_ERROR;
 }
 
 atb::Status SetPostMoeProcess(std::map<std::string, uint32_t> &tensorMap,
     const DecoderLayerParam &param, atb::GraphParam &opGraph)
 {
     if (param.hasFfnComm && param.hasAttnComm) {
         CHECK_OPERATION_STATUS_RETURN(SetFFNPadding(opGraph, param, tensorMap));
     }
     if (param.ffnAllreduce) {
         CHECK_OPERATION_STATUS_RETURN(SetAllReduce(opGraph, param, tensorMap));
     } else if (param.ffnReduceScatter) {
         CHECK_OPERATION_STATUS_RETURN(SetMlpReduceScatter(opGraph, param, tensorMap));
         if (param.enableInfNan) {
             CHECK_OPERATION_STATUS_RETURN(SetMlpReduceScatterNanToNum(opGraph, tensorMap));
         }
         if (param.enableSharedExpertDp) {
             CHECK_OPERATION_STATUS_RETURN(AddExpertAdd(opGraph, param, tensorMap));
         }
     }
     CHECK_OPERATION_STATUS_RETURN(SetMlpResidualAdd(opGraph, param, tensorMap));
     if (param.enableInfNan) {
         CHECK_OPERATION_STATUS_RETURN(SetMlpResidualAddNanToNum(opGraph, param, tensorMap));
     }
     if (param.ffnAllGather) {
         if (!param.hasAttnComm) {
             CHECK_OPERATION_STATUS_RETURN(SetFFNPadding(opGraph, param, tensorMap));
         }
         // h3p qkvdown dp move moe allgather to mla, without last moe
         if (!param.enableQkvdownDp || param.isLastLayer) {
             CHECK_OPERATION_STATUS_RETURN(SetTPAllGatherNode(opGraph, param, tensorMap));
         }
     }
     if (param.hasFfnComm) {
         // h3p qkvdown dp move moe gather to mla, without last moe
         if (!param.enableQkvdownDp || param.isLastLayer) {
             CHECK_OPERATION_STATUS_RETURN(SetFFNUnPadding(opGraph, param, tensorMap));
         }
     }
     return atb::NO_ERROR;
 }
 
 atb::Status DecoderLayer(DecoderLayerParam &param, atb::Operation **operation)
 {
     atb::GraphParam opGraph;
     opGraph.name = param.isPrefill ? "Prefill_layer" : "Decoder_layer";
     CalculateDataPartition(param);
     CalculateCommType(param);
     param.enableQkvdownDp = param.enableQkvdownDp && param.ffnAllGather;
     std::map<std::string, uint32_t> tensorMap = ConstructTensorMap(
         param, opGraph.inTensorNum, opGraph.outTensorNum, opGraph.internalTensorNum);
     ATB_SPEED_LOG_DEBUG("layer graph inTensorNum: " << opGraph.inTensorNum);
     ATB_SPEED_LOG_DEBUG("layer graph outTensorNum: " << opGraph.outTensorNum);
     ATB_SPEED_LOG_DEBUG("layer graph internalTensorNum: " << opGraph.internalTensorNum);
     CHECK_OPERATION_STATUS_RETURN(SetAttention(opGraph, param, tensorMap));
     CHECK_OPERATION_STATUS_RETURN(SetPostAttnProcess(tensorMap, param, opGraph));
     CHECK_OPERATION_STATUS_RETURN(SetFFN(tensorMap, param, opGraph));
     CHECK_OPERATION_STATUS_RETURN(SetPostMoeProcess(tensorMap, param, opGraph));
     opGraph.inferShapeFunc = [=] (const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                  atb::SVector<atb::TensorDesc> &outTensorDescs) {
         if (param.mapping.Get(base::ATTN_DP).IsEnabled() && param.isLastLayer && !param.enableDpOut) {
             outTensorDescs.at(0) = inTensorDescs.at(atb_speed::common::GetTensorIdx(tensorMap, "in_final_state"));
         } else if (param.mapping.Get(base::ATTN_DP).IsEnabled() && param.isLastLayer && \
             param.enableDpOut && param.lmHeadLocalTp) {
             outTensorDescs.at(0) = inTensorDescs.at(atb_speed::common::GetTensorIdx(tensorMap, "in_final_state"));
             outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(atb_speed::common::GetTensorIdx(tensorMap, \
                 "in_lm_head_skip_padding_token_indices")).shape.dims[0];
         } else {
             outTensorDescs.at(0) = inTensorDescs.at(atb_speed::common::GetTensorIdx(tensorMap, "in_hidden_states"));
         }
 
         if (param.enableQkvdownDp && param.layerId == param.firstKDenseReplace && \
             param.mapping.Get(base::MLP_TP).rankIds.size() != 0) {
             outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(
                 atb_speed::common::GetTensorIdx(tensorMap, "in_ffn_padding_idx")
             ).shape.dims[0] / param.mapping.Get(base::MLP_TP).rankIds.size();
         }
 
         if (!param.isDenseLayer && param.enableExpertCumSumOutput) {
             outTensorDescs.at(1) = atb::TensorDesc{};
             outTensorDescs.at(1).format = ACL_FORMAT_ND;
             outTensorDescs.at(1).shape.dimNum = 1;
             outTensorDescs.at(1).dtype = ACL_INT64;
             if (!param.isPrefill && param.enableAllToAllMC2 && param.isDynamicEp) {
                 outTensorDescs.at(1).shape.dims[0] = param.numOfDeviceExperts;
             } else if (param.enableFusedRouting) {
                 if (param.mapping.Get(base::MOE_EP).IsEnabled()) {
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
     CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(opGraph, operation));
     return atb::NO_ERROR;
 }
 
 DecoderLayer::DecoderLayer() {}
 
 DecoderLayer::~DecoderLayer() {}
 
 } // namespace deepseekV2
 } // namespace atb_speed
 
 