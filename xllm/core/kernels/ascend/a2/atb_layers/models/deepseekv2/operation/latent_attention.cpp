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

 #include "atb_speed/base/model.h"
 #include "operations/fusion/utils.h"
 #include "operations/aclnn/ops/repeat_operation.h"
 #include "models/deepseekv2/operation/sp_decode.h"
 #include "models/deepseekv2/operation/latent_attention.h"
 #include <gflags/gflags.h>

DECLARE_bool(enable_atb_comm_multiprocess);

 namespace atb_speed {
 namespace deepseekV2 {
 using namespace atb_speed::common;
 template <typename NormParamType>
 bool EnableFA3Quant(const LatentAttentionParam<NormParamType> &param)
 {
     return param.pageAttentionParam.quantType == \
         atb::infer::PagedAttentionParam::QuantType::TYPE_QUANT_QKV_ONLINE;
 }
 
 std::map<std::string, std::vector<std::string>> GetLatentAttnInTensorCandidates()
 {
     std::map<std::string, std::vector<std::string>>  latentAttnInTensorCandidates = {
         {"default", {
             "in_input", "in_norm_weight", "in_norm_bias", "in_norm_new_weight", "in_norm_new_weight_bias",
             "in_q_proj_a_weight", "in_q_proj_a_bias", "in_q_proj_a_descale", "in_q_proj_a_offset", "in_q_proj_a_scale",
             "in_q_proj_a_compress_idx",
             "in_q_proj_a_layernorm_weight", "in_q_proj_a_layernorm_bias",
             "in_q_proj_b_weight", "in_q_proj_b_bias", "in_q_proj_b_descale", "in_q_proj_b_offset", "in_q_proj_b_scale",
             "in_q_proj_b_compress_idx",
             "in_kv_proj_with_mqa_weight", "in_kv_proj_with_mqa_bias", "in_kv_proj_with_mqa_descale",
             "in_kv_proj_with_mqa_offset", "in_kv_proj_with_mqa_scale", "in_kv_proj_with_mqa_compress_idx",
             "in_kv_proj_a_layernorm_weight", "in_kv_proj_a_layernorm_bias",
             "in_k_proj_b_for_q_weight", "in_k_proj_b_for_q_bias", "in_k_proj_b_for_q_descale",
             "in_k_proj_b_for_q_offset", "in_k_proj_b_for_q_scale", "in_k_proj_b_for_q_compress_idx",
             "in_v_proj_b_for_o_weight", "in_v_proj_b_for_o_bias", "in_v_proj_b_for_o_descale",
             "in_v_proj_b_for_o_offset", "in_v_proj_b_for_o_scale", "in_v_proj_b_for_o_compress_idx",
             "in_attn_out_weight", "in_attn_out_bias", "in_attn_out_descale", "in_attn_out_offset",
             "in_attn_out_scale", "in_attn_out_compress_idx",
             "in_cos_embed", "in_sin_embed", "in_seq_len", "in_k_cache", "in_k_rope_cache",
             "in_attention_mask", "in_q_len",
             "in_token_offset", "in_layer_id", "in_block_tables",
             "in_slots_in_pa_or_logn_in_fa", "in_attn_padding_idx"}
         },
         {"fa3_quant", {
             "in_q_quant_scale", "in_k_quant_scale", "in_qk_descale",
             "kv_offset", "fa3_v_quant_scale"}
         },
         {"attn_inner_sp_prefill", {"in_k_sp_gather_indices"}},
         {"attn_inner_sp_decode", {"in_seq_len_sp"}},
         {"qkvdown_dp", {"in_ffn_unpadding_idx"}
         },
     };
     return latentAttnInTensorCandidates;
 }
 
 std::map<std::string, std::vector<std::string>> GetLatentAttnIntermediateTensorCandidates()
 {
     std::map<std::string, std::vector<std::string>> latentAttnIntermediateTensorCandidates = {
         {"default",
             {
                 "in_input_norm", "latent_q", "nope_q", "rope_q", "rope_k", "rope_q_o", "rope_k_o",
                 "intermediate_kv", "intermediate_self_attention"
             }
         },
         {"prefill",
             {
                 "intermediate_q", "rope_k_o_repeat", "intermediate_k_nope", "intermediate_k_mha", "intermediate_v_mha",
                 "temp_v_proj_b"
             }
         },
         {"decode", {"reproj_nope_q", "reproj_o"}},
         {"q_lora", {"latent_qkv", "latent_q_norm", "q_lora_out"}},
         {"no_q_lora", {"latent_kv"}},
         {"mla_preprocess",
             {
                 "intermediate_q_nope", "intermediate_self_attention", "reproj_o", "intermediate_q_rope",
             }
         },
         {"kv_quant_scale", {"intermediate_kv_int8"}},
         {"attn_inner_sp_prefill", {"intermediate_kv_sp", "rope_k_o_sp"}},
         {"attn_inner_sp_decode", {"intermediate_q", "intermediate_q_sp_nope", "intermediate_q_sp_rope",
             "intermediate_go", "intermediate_lse", "intermediate_go_lse_concat",
             "intermediate_go_lse_concat_allgather_slice",
             "intermediate_go_allgather_slice", "intermediate_lse_allgather_slice",
             "intermediate_go_fp32", "intermediate_lse_fp32", "intermediate_fa_update_out_fp32",
             "intermediate_q_allgather_sp", "intermediate_q_allgather_sp_t"}
         },
         {"extra_o_proj_tp",
             {"intermediate_self_attention_padding"}
         },
         {"extra_o_proj_tp_quant",
             {"intermediate_self_attention_padding_quant"}
         },
         {"qkvdown_dp", {"intermediate_qkv_all", "intermediate_qkv_unpadding_all"}},
     };
     return latentAttnIntermediateTensorCandidates;
 }
 
 template <typename NormParamType>
 std::map<std::string, uint32_t> ConstructTensorMap(const LatentAttentionParam<NormParamType> &param,
     uint32_t &inTensorNum, uint32_t &outTensorNum, uint32_t &internalTensorNum)
 {
     std::vector<std::string> inTensorList = {};
     std::vector<std::string> outTensorList = {"out"};
     std::vector<std::string> intermediateTensorList = {};
     auto latentAttnInTensorCandidates = GetLatentAttnInTensorCandidates();
     auto latentAttnIntermediateTensorCandidates = GetLatentAttnIntermediateTensorCandidates();
 
     // 添加默认的Tensor
     AddTensorToList(latentAttnInTensorCandidates, "default", inTensorList);
     // 添加FA3特性的Tensor
     if (EnableFA3Quant(param)) {
         AddTensorToList(latentAttnInTensorCandidates, "fa3_quant", inTensorList);
         if (param.isPrefill) {
             AddTensorToList(latentAttnIntermediateTensorCandidates, "kv_quant_scale", intermediateTensorList);
         }
     }
     if (param.enableQkvdownDp) {
         AddTensorToList(latentAttnInTensorCandidates, "qkvdown_dp", inTensorList);
         AddTensorToList(latentAttnIntermediateTensorCandidates, "qkvdown_dp", intermediateTensorList);
     }
     if (param.enableMlaPreprocess && !param.isPrefill) {
         AddTensorToList(latentAttnIntermediateTensorCandidates, "mla_preprocess", intermediateTensorList);
     } else {
         AddTensorToList(latentAttnIntermediateTensorCandidates, "default", intermediateTensorList);
         if (param.qLoraRank != 0) {
             AddTensorToList(latentAttnIntermediateTensorCandidates, "q_lora", intermediateTensorList);
         } else {
             AddTensorToList(latentAttnIntermediateTensorCandidates, "no_q_lora", intermediateTensorList);
         }
         if (param.isPrefill) {
             AddTensorToList(latentAttnIntermediateTensorCandidates, "prefill", intermediateTensorList);
         } else {
             AddTensorToList(latentAttnIntermediateTensorCandidates, "decode", intermediateTensorList);
         }
     }
     if (param.hasAttnInnerSp) {
         if (param.isPrefill) {
             AddTensorToList(latentAttnInTensorCandidates, "attn_inner_sp_prefill", inTensorList);
             AddTensorToList(latentAttnIntermediateTensorCandidates, "attn_inner_sp_prefill", intermediateTensorList);
         } else {
             AddTensorToList(latentAttnInTensorCandidates, "attn_inner_sp_decode", inTensorList);
             AddTensorToList(latentAttnIntermediateTensorCandidates, "attn_inner_sp_decode", intermediateTensorList);
         }
     }
     if (param.enableExtraOprojTp) {
         AddTensorToList(latentAttnIntermediateTensorCandidates, "extra_o_proj_tp", intermediateTensorList);
         if (UseExtraQuant(param, O_LINEAR_INDEX)) {
             AddTensorToList(latentAttnIntermediateTensorCandidates, "extra_o_proj_tp_quant", intermediateTensorList);
         }
     }
     inTensorNum = inTensorList.size();
     outTensorNum = outTensorList.size();
     internalTensorNum = intermediateTensorList.size();
 
     return GetTensorMap(inTensorList, outTensorList, intermediateTensorList);
 }
 
 atb::Status AddKVQuantNode(atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
 {
     atb::Node kvQuantNode;
     atb::infer::ElewiseParam kvQuantParam;
     kvQuantParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_QUANT_PER_CHANNEL;
     CREATE_OPERATION(kvQuantParam, &kvQuantNode.operation);
     kvQuantNode.inTensorIds = {
         GetTensorIdx(tensorMap, "intermediate_kv"), GetTensorIdx(tensorMap, "in_k_quant_scale"),
         GetTensorIdx(tensorMap, "kv_offset")
     };
     kvQuantNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_kv_int8")};
     kvQuantNode.inTensorReshapeFuncs.resize(kvQuantNode.inTensorIds.size());
     opGraph.nodes.push_back(kvQuantNode);
     return atb::NO_ERROR;
 }
 
 void SqueezeHeadNumHeadDim(const atb::Dims &oldShape, atb::Dims &newShape)
 {
     if (oldShape.dimNum == 4) {  // 4: FA
         newShape.dimNum = 3;  // 3: 新的shape维度为3
         newShape.dims[0] = oldShape.dims[0];  // 0, 0: 新shape的第0维不变
         newShape.dims[1] = oldShape.dims[1];  // 1, 1: 新shape的第1维不变
         newShape.dims[2] =  oldShape.dims[2] * oldShape.dims[3];  // 2, 2, 3: 后两维合轴
     } else {
         newShape.dimNum = 2;  // 2: 新的shape维度为2
         newShape.dims[0] = oldShape.dims[0];  // 0, 0: 新shape的第0维不变
         newShape.dims[1] =  oldShape.dims[1] * oldShape.dims[2];  // 1, 1, 2: 后两维合轴
     }
 }
 
 
 template <typename NormParamType>
 bool UseExtraQuant(const LatentAttentionParam<NormParamType> &param, uint64_t linearIndex)
 {
     LinearQuantType quantType = GetLinearQuantType(
         param.denseQuantType == atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED \
             ? param.packQuantType : param.denseQuantType,
         param.attnLinearQuantType[linearIndex], true);
     if (quantType == LinearQuantType::LINEAR_W8A8_DEQUANT || \
         quantType == LinearQuantType::LINEAR_W8A8_SC_DEQUANT) {
         return true;
     } else {
         return false;
     }
 }
 
 template <typename NormParamType>
 atb::Status AddMlaPreprocessNode(const LatentAttentionParam<NormParamType> &param, atb::GraphParam &opGraph,
     std::map<std::string, uint32_t> &tensorMap)
 {
     atb::Node mlaPreprocessNode;
     atb::infer::MlaPreprocessParam mlaPreprocessParam;
     mlaPreprocessParam.wdqDim = param.qLoraRank;
     mlaPreprocessParam.qRopeDim = param.qkRopeHeadDim;
     mlaPreprocessParam.kRopeDim = param.qkRopeHeadDim;
     if (EnableFA3Quant(param)) {
         mlaPreprocessParam.cacheMode = atb::infer::MlaPreprocessParam::CacheMode::INT8_NZCACHE;
     } else if (param.isNzCache) {
         mlaPreprocessParam.cacheMode = atb::infer::MlaPreprocessParam::CacheMode::NZCACHE;
     } else {
         mlaPreprocessParam.cacheMode = atb::infer::MlaPreprocessParam::CacheMode::KROPE_CTKV;
     }
     CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(mlaPreprocessParam, &mlaPreprocessNode.operation));
     mlaPreprocessNode.inTensorIds = {
         GetTensorIdx(tensorMap, "in_input"), GetTensorIdx(tensorMap, "in_norm_weight"),
         GetTensorIdx(tensorMap, "in_norm_bias"), GetTensorIdx(tensorMap, "in_q_proj_a_scale"),
         GetTensorIdx(tensorMap, "in_q_proj_a_offset"), GetTensorIdx(tensorMap, "in_q_proj_a_weight"),
         GetTensorIdx(tensorMap, "in_q_proj_a_descale"), GetTensorIdx(tensorMap, "in_q_proj_a_bias"),
         GetTensorIdx(tensorMap, "in_q_proj_a_layernorm_weight"), GetTensorIdx(tensorMap, "in_q_proj_a_layernorm_bias"),
         GetTensorIdx(tensorMap, "in_q_proj_b_scale"), GetTensorIdx(tensorMap, "in_q_proj_b_offset"),
         GetTensorIdx(tensorMap, "in_q_proj_b_weight"), GetTensorIdx(tensorMap, "in_q_proj_b_descale"),
         GetTensorIdx(tensorMap, "in_q_proj_b_bias"), GetTensorIdx(tensorMap, "in_kv_proj_a_layernorm_weight"),
         GetTensorIdx(tensorMap, "in_cos_embed"), GetTensorIdx(tensorMap, "in_sin_embed"),
         GetTensorIdx(tensorMap, "in_k_proj_b_for_q_weight"), GetTensorIdx(tensorMap, "in_k_cache"),
         GetTensorIdx(tensorMap, "in_k_rope_cache"), GetTensorIdx(tensorMap, "in_slots_in_pa_or_logn_in_fa"),
     };
     if (EnableFA3Quant(param)) {
         mlaPreprocessNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_k_quant_scale"));
         mlaPreprocessNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_q_quant_scale"));
     } else {
         mlaPreprocessNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_kv_proj_with_mqa_scale"));
         mlaPreprocessNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_q_proj_b_scale"));
     }
     mlaPreprocessNode.outTensorIds = {
         GetTensorIdx(tensorMap, "intermediate_q_nope"),
         GetTensorIdx(tensorMap, "in_k_cache"),
         GetTensorIdx(tensorMap, "intermediate_q_rope"),
         GetTensorIdx(tensorMap, "in_k_rope_cache")
     };
     opGraph.nodes.push_back(mlaPreprocessNode);
     ATB_SPEED_LOG_DEBUG("MlaPreprocessNode calculation success");
     return atb::NO_ERROR;
 }
 
 template <typename NormParamType>
 atb::Status AddLAttnPreNormNode(const LatentAttentionParam<NormParamType> &param,
     atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
 {
     atb::Node normNode;
 
     if (UseExtraQuant(param, Q_PROJ_A_LINEAR_INDEX)) {  // W8A8
         CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(param.normQuantParamType, &normNode.operation));
         normNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_input"));
         normNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_norm_weight"));
         normNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_norm_bias"));
         normNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_q_proj_a_scale"));
         normNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_q_proj_a_offset"));
     } else {
         CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(param.normParamType, &normNode.operation));
         normNode.inTensorIds = {GetTensorIdx(tensorMap, "in_input"), GetTensorIdx(tensorMap, "in_norm_weight")};
     }
     normNode.outTensorIds = {GetTensorIdx(tensorMap, "in_input_norm")};
     opGraph.nodes.push_back(normNode);
     ATB_SPEED_LOG_DEBUG("Attention PreNorm calculation success");
     return atb::NO_ERROR;
 }
 
 template <typename NormParamType>
 atb::Status AddLAttnQKVProjNode(const LatentAttentionParam<NormParamType> &param,
                                 atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
 {
     atb::Node qkvAProjNode;
     atb_speed::common::FusionLinearParam kvAProjNodeParam;
     kvAProjNodeParam.isBF16 = param.isBF16;
     kvAProjNodeParam.hasBias = param.selfAttnHasBias;
     kvAProjNodeParam.quantType = GetLinearQuantType(
         param.denseQuantType == atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED ? \
             param.packQuantType : param.denseQuantType,
         param.attnLinearQuantType[Q_PROJ_A_LINEAR_INDEX], true);
     kvAProjNodeParam.quantGroupSize = param.quantGroupSize;
     qkvAProjNode.inTensorIds = {
         GetTensorIdx(tensorMap, "in_input_norm"),
         GetTensorIdx(tensorMap, "in_q_proj_a_weight"),
         GetTensorIdx(tensorMap, "in_q_proj_a_scale"),
         GetTensorIdx(tensorMap, "in_q_proj_a_offset"),
         GetTensorIdx(tensorMap, "in_q_proj_a_descale"),
         GetTensorIdx(tensorMap, "in_q_proj_a_bias"),
         GetTensorIdx(tensorMap, "in_q_proj_a_compress_idx"),
     };
     qkvAProjNode.outTensorIds = {GetTensorIdx(tensorMap, "latent_qkv")};
     CHECK_OPERATION_STATUS_RETURN(FusionLinear(kvAProjNodeParam, &qkvAProjNode.operation));
     opGraph.nodes.push_back(qkvAProjNode);
     ATB_SPEED_LOG_DEBUG("MLA proj_qkv_a calculation success");
     return atb::NO_ERROR;
 }
 
 template<typename NormParamType>
 atb::Status AddSplitQKNode(const LatentAttentionParam<NormParamType> &param, atb::GraphParam &opGraph,
                            std::map<std::string, uint32_t> &tensorMap)
 {
     atb::Node splitKNode;
     atb::infer::SplitParam splitKParam = {
         (param.isFA ? 2 : 1), 3, {param.kvLoraRank, param.qkRopeHeadDim, param.qLoraRank}};
     CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(splitKParam, &splitKNode.operation));
     splitKNode.inTensorIds = {GetTensorIdx(tensorMap, param.enableQkvdownDp ?
         "intermediate_qkv_unpadding_all" : "latent_qkv")};
     splitKNode.outTensorIds = {GetTensorIdxList(tensorMap, {"intermediate_kv", "rope_k", "latent_q"})};
     opGraph.nodes.push_back(splitKNode);
     ATB_SPEED_LOG_DEBUG("MLA spilt_qk calculation success");
     return atb::NO_ERROR;
 }
 
 template <typename NormParamType>
 atb::Status AddLAttnQProjANode(const LatentAttentionParam<NormParamType> &param,
     atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
 {
     atb::Node qAProjNode;
     atb_speed::common::FusionLinearParam qAProjNodeParam;
     qAProjNodeParam.isBF16 = param.isBF16;
     qAProjNodeParam.hasBias = param.selfAttnHasBias;
     qAProjNodeParam.quantType = GetLinearQuantType(
         param.denseQuantType == atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED \
             ? param.packQuantType : param.denseQuantType,
         param.attnLinearQuantType[Q_PROJ_A_LINEAR_INDEX], true);
     qAProjNodeParam.quantGroupSize = param.quantGroupSize;
     qAProjNodeParam.transposeType = param.attnLinearTransposeType[Q_PROJ_A_LINEAR_INDEX];
     qAProjNode.inTensorIds = {
         GetTensorIdx(tensorMap, "in_input_norm"),
         GetTensorIdx(tensorMap, "in_q_proj_a_weight"),
         GetTensorIdx(tensorMap, "in_q_proj_a_scale"),
         GetTensorIdx(tensorMap, "in_q_proj_a_offset"),
         GetTensorIdx(tensorMap, "in_q_proj_a_descale"),
         GetTensorIdx(tensorMap, "in_q_proj_a_bias"),
         GetTensorIdx(tensorMap, "in_q_proj_a_compress_idx"),
     };
     qAProjNode.outTensorIds = {GetTensorIdx(tensorMap, "latent_q")};
     CHECK_OPERATION_STATUS_RETURN(FusionLinear(qAProjNodeParam, &qAProjNode.operation));
     opGraph.nodes.push_back(qAProjNode);
     ATB_SPEED_LOG_DEBUG("MLA proj_q_a calculation success");
     return atb::NO_ERROR;
 }
 
 
 template <typename NormParamType>
 atb::Status AddLAttnQProjBNode(const LatentAttentionParam<NormParamType> &param,
     atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
 {
     atb::Node qNormNode;
     if (UseExtraQuant(param, Q_PROJ_B_LINEAR_INDEX)) {  // W8A8
         CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(param.normQuantParamType, &qNormNode.operation));
         qNormNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "latent_q"));
         qNormNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_q_proj_a_layernorm_weight"));
         qNormNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_q_proj_a_layernorm_bias"));
         qNormNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_q_proj_b_scale"));
         qNormNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_q_proj_b_offset"));
     } else {
         CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(param.normParamType, &qNormNode.operation));
         qNormNode.inTensorIds = {GetTensorIdx(tensorMap, "latent_q"),
                                 GetTensorIdx(tensorMap, "in_q_proj_a_layernorm_weight")};
     }
     qNormNode.outTensorIds = {GetTensorIdx(tensorMap, "latent_q_norm")};
     opGraph.nodes.push_back(qNormNode);
 
     atb::Node qBProjNode;
     atb_speed::common::FusionLinearParam qBProjNodeParam;
     qBProjNodeParam.isBF16 = param.isBF16;
     qBProjNodeParam.hasBias = param.selfAttnHasBias;
     qBProjNodeParam.quantType = GetLinearQuantType(
         param.denseQuantType == atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED \
             ? param.packQuantType : param.denseQuantType,
         param.attnLinearQuantType[Q_PROJ_B_LINEAR_INDEX], true);
     qBProjNodeParam.quantGroupSize = param.quantGroupSize;
     qBProjNodeParam.transposeType = param.attnLinearTransposeType[Q_PROJ_B_LINEAR_INDEX];
     CHECK_OPERATION_STATUS_RETURN(FusionLinear(qBProjNodeParam, &qBProjNode.operation));
     qBProjNode.inTensorIds = {
         GetTensorIdx(tensorMap, "latent_q_norm"),
         GetTensorIdx(tensorMap, "in_q_proj_b_weight"),
         GetTensorIdx(tensorMap, "in_q_proj_b_scale"),
         GetTensorIdx(tensorMap, "in_q_proj_b_offset"),
         GetTensorIdx(tensorMap, "in_q_proj_b_descale"),
         GetTensorIdx(tensorMap, "in_q_proj_b_bias"),
         GetTensorIdx(tensorMap, "in_q_proj_b_compress_idx"),
     };
     qBProjNode.outTensorIds = {GetTensorIdx(tensorMap, "q_lora_out")};
     opGraph.nodes.push_back(qBProjNode);
     ATB_SPEED_LOG_DEBUG("MLA proj_q_b calculation success");
     return atb::NO_ERROR;
 }
 
 
 template <typename NormParamType>
 atb::Status AddSplitQNode(const LatentAttentionParam<NormParamType> &param, atb::GraphParam &opGraph,
     std::map<std::string, uint32_t> &tensorMap)
 {
     atb::Node splitQNode;
     atb::infer::SplitParam splitQParam = {(param.isFA ? 3 : 2), 2, {param.qkNopeHeadDim, param.qkRopeHeadDim}};
     CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(splitQParam, &splitQNode.operation));
     if (param.qLoraRank == 0) {    // 如果是lite
         splitQNode.inTensorIds = {GetTensorIdx(tensorMap, "latent_q")};
     } else {
         splitQNode.inTensorIds = {GetTensorIdx(tensorMap, "q_lora_out")};
     }
     splitQNode.inTensorReshapeFuncs.resize(splitQNode.inTensorIds.size());
     splitQNode.outTensorIds = {GetTensorIdxList(tensorMap, {"nope_q", "rope_q"})};
     if (param.isFA) {
         splitQNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
                 newShape.dimNum = 4; // 4: dimNum
                 newShape.dims[0] = oldShape.dims[0];
                 newShape.dims[1] = oldShape.dims[1];
                 newShape.dims[2] = param.selfAttentionParam.headNum; // 2: dim id
                 newShape.dims[3] = param.qkNopeHeadDim + param.qkRopeHeadDim; // 3: dim id
             };
     } else {
         splitQNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
             newShape.dimNum = 3; // 3: dimNum
             newShape.dims[0] = oldShape.dims[0];
             newShape.dims[1] = param.selfAttentionParam.headNum;
             newShape.dims[2] = param.qkNopeHeadDim + param.qkRopeHeadDim; // 2: dim id
         };
     }
     opGraph.nodes.push_back(splitQNode);
     ATB_SPEED_LOG_DEBUG("MLA split q calculation success");
     return atb::NO_ERROR;
 }
 
 template <typename NormParamType>
 atb::Status AddReprojQNode(const LatentAttentionParam<NormParamType> &param, atb::GraphParam &opGraph,
     std::map<std::string, uint32_t> &tensorMap)
 {
     atb::Node qReprojNode;
     atb_speed::common::FusionLinearParam qReprojNodeParam;
     qReprojNodeParam.isBF16 = param.isBF16;
     qReprojNodeParam.hasBias = param.selfAttnHasBias;
     qReprojNodeParam.quantType = GetLinearQuantType(
         param.denseQuantType == atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED \
             ? param.packQuantType : param.denseQuantType,
         param.attnLinearQuantType[KV_PROJ_B_FOR_Q_LINEAR_INDEX], false);
     qReprojNodeParam.quantGroupSize = param.quantGroupSize;
     qReprojNodeParam.transposeType = false;
     qReprojNodeParam.enEin = true;
     CHECK_OPERATION_STATUS_RETURN(FusionLinear(qReprojNodeParam, &qReprojNode.operation));
     qReprojNode.inTensorIds = {
         GetTensorIdx(tensorMap, "nope_q"),
         GetTensorIdx(tensorMap, "in_k_proj_b_for_q_weight"),
         GetTensorIdx(tensorMap, "in_k_proj_b_for_q_scale"),
         GetTensorIdx(tensorMap, "in_k_proj_b_for_q_offset"),
         GetTensorIdx(tensorMap, "in_k_proj_b_for_q_descale"),
         GetTensorIdx(tensorMap, "in_k_proj_b_for_q_bias"),
         GetTensorIdx(tensorMap, "in_k_proj_b_for_q_compress_idx"),
     };
     qReprojNode.outTensorIds = {GetTensorIdx(tensorMap, "reproj_nope_q")};
     opGraph.nodes.push_back(qReprojNode);
     ATB_SPEED_LOG_DEBUG("MLA reproj q calculation success");
     return atb::NO_ERROR;
 }
 
 
 template <typename NormParamType>
 atb::Status AddLAttnKVAProjNode(const LatentAttentionParam<NormParamType> &param,
     atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
 {
     atb::Node kvAProjNode;
     atb_speed::common::FusionLinearParam kvAProjNodeParam;
     kvAProjNodeParam.isBF16 = param.isBF16;
     kvAProjNodeParam.hasBias = param.selfAttnHasBias;
     kvAProjNodeParam.quantType = GetLinearQuantType(
         param.denseQuantType == atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED \
             ? param.packQuantType : param.denseQuantType,
         param.attnLinearQuantType[KV_PROJ_A_LINEAR_INDEX], true);
     kvAProjNodeParam.quantGroupSize = param.quantGroupSize;
     kvAProjNodeParam.transposeType = param.attnLinearTransposeType[KV_PROJ_A_LINEAR_INDEX];
     kvAProjNode.inTensorIds = {
         GetTensorIdx(tensorMap, "in_input_norm"),
         GetTensorIdx(tensorMap, "in_kv_proj_with_mqa_weight"),
         GetTensorIdx(tensorMap, "in_kv_proj_with_mqa_scale"),
         GetTensorIdx(tensorMap, "in_kv_proj_with_mqa_offset"),
         GetTensorIdx(tensorMap, "in_kv_proj_with_mqa_descale"),
         GetTensorIdx(tensorMap, "in_kv_proj_with_mqa_bias"),
         GetTensorIdx(tensorMap, "in_kv_proj_with_mqa_compress_idx"),
     };
     kvAProjNode.outTensorIds = {GetTensorIdx(tensorMap, "latent_kv")};
     CHECK_OPERATION_STATUS_RETURN(FusionLinear(kvAProjNodeParam, &kvAProjNode.operation));
     opGraph.nodes.push_back(kvAProjNode);
     ATB_SPEED_LOG_DEBUG("MLA proj_kv_a calculation success");
     return atb::NO_ERROR;
 }
 
 
 template <typename NormParamType>
 atb::Status AddSplitKNode(const LatentAttentionParam<NormParamType> &param, atb::GraphParam &opGraph,
     std::map<std::string, uint32_t> &tensorMap)
 {
     atb::Node splitKNode;
     atb::infer::SplitParam splitKParam = {(param.isFA ? 2 : 1), 2, {param.kvLoraRank, param.qkRopeHeadDim}};
     CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(splitKParam, &splitKNode.operation));
     splitKNode.inTensorIds = {GetTensorIdx(tensorMap, "latent_kv")};
     splitKNode.outTensorIds = {GetTensorIdxList(tensorMap, {"intermediate_kv", "rope_k"})};
     opGraph.nodes.push_back(splitKNode);
     ATB_SPEED_LOG_DEBUG("MLA spilt_k calculation success");
     return atb::NO_ERROR;
 }
 
 template <typename NormParamType>
 atb::Status AddLAttnKVNormNode(const LatentAttentionParam<NormParamType> &param,
     atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
 {
     atb::Node kvNormNode;
     kvNormNode.inTensorIds = {
         GetTensorIdx(tensorMap, "intermediate_kv"), GetTensorIdx(tensorMap, "in_kv_proj_a_layernorm_weight")
     };
     kvNormNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_kv")};
     if (!param.isFA) {
         kvNormNode.inTensorReshapeFuncs.resize(kvNormNode.inTensorIds.size());
         kvNormNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
             newShape.dimNum = 3; // 3: dimNum
             newShape.dims[0] = oldShape.dims[0];
             newShape.dims[1] = 1;
             newShape.dims[2] = param.kvLoraRank; // 2: dim id
         };
     }
     CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(param.normParamType, &kvNormNode.operation));
     opGraph.nodes.push_back(kvNormNode);
     ATB_SPEED_LOG_DEBUG("MLA kv norm calculation success");
     return atb::NO_ERROR;
 }
 
 
 template <typename NormParamType>
 atb::Status AddLAttnRopeNode(const LatentAttentionParam<NormParamType> &param,
     atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
 {
     atb::Node ropeNode;
     atb::infer::RopeParam ropeParam;
     ropeParam.rotaryCoeff = param.ropeParam.rotaryCoeff;
     CreateOperation(ropeParam, &ropeNode.operation);
     ropeNode.inTensorIds = {
         GetTensorIdx(tensorMap, "rope_q"), GetTensorIdx(tensorMap, "rope_k"),
         GetTensorIdx(tensorMap, "in_cos_embed"), GetTensorIdx(tensorMap, "in_sin_embed"),
         GetTensorIdx(tensorMap, "in_seq_len")
     };
     ropeNode.outTensorIds = {
         GetTensorIdx(tensorMap, "rope_q_o"), GetTensorIdx(tensorMap, "rope_k_o"),
     };
     ropeNode.inTensorReshapeFuncs.resize(ropeNode.inTensorIds.size());
     ropeNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
         SqueezeHeadNumHeadDim(oldShape, newShape);
     };
     opGraph.nodes.push_back(ropeNode);
     ATB_SPEED_LOG_DEBUG("MLA rope calculation success");
     return atb::NO_ERROR;
 }
 
 
 template <typename NormParamType>
 atb::Status AddLAttnQCatNode(const LatentAttentionParam<NormParamType> &param,
     atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
 {
     atb::Node qCatNode;
     atb::infer::ConcatParam qCatParam;
     qCatParam.concatDim = -1;
     if (param.isPrefill) {
         qCatNode.inTensorIds = {
             GetTensorIdx(tensorMap, "nope_q"), GetTensorIdx(tensorMap, "rope_q_o")};
     } else {
         qCatNode.inTensorIds = {
             GetTensorIdx(tensorMap, param.enableMlaPreprocess ? "intermediate_q_nope" : "reproj_nope_q"),
             GetTensorIdx(tensorMap, param.enableMlaPreprocess ? "intermediate_q_rope" : "rope_q_o")
         };
     }
     qCatNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_q")};
     qCatNode.inTensorReshapeFuncs.resize(qCatNode.inTensorIds.size());
     if (param.isFA) {
         qCatNode.inTensorReshapeFuncs[1] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
             newShape.dimNum = 4; // 4: dimNum
             newShape.dims[0] = oldShape.dims[0];
             newShape.dims[1] = oldShape.dims[1];
             newShape.dims[2] = param.selfAttentionParam.headNum; // 2: dim id
             newShape.dims[3] = param.qkRopeHeadDim; // 3: dim id
         };
     } else {
         qCatNode.inTensorReshapeFuncs[1] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
             newShape.dimNum = 3; // 3: dimNum
             newShape.dims[0] = oldShape.dims[0];
             newShape.dims[1] = param.selfAttentionParam.headNum;
             newShape.dims[2] = param.qkRopeHeadDim; // 2: dim id
         };
     }
     CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(qCatParam, &qCatNode.operation));
     opGraph.nodes.push_back(qCatNode);
     ATB_SPEED_LOG_DEBUG("MLA qCatNode calculation success");
     return atb::NO_ERROR;
 }
 
 
 template <typename NormParamType>
 atb::Status AddLAttnKCatPrefillNode(const LatentAttentionParam<NormParamType> &param,
     atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
 {
     atb::Node keyRepeatNode;
     atb_speed::common::AclNNRepeatParam kvRepeatParam;
     kvRepeatParam.repeatsArray = {1, param.selfAttentionParam.headNum, 1};
     keyRepeatNode.inTensorIds = {GetTensorIdx(tensorMap, "rope_k_o")};
     keyRepeatNode.outTensorIds = {GetTensorIdx(tensorMap, "rope_k_o_repeat")};
     keyRepeatNode.inTensorReshapeFuncs.resize(keyRepeatNode.inTensorIds.size());
     keyRepeatNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
         newShape.dimNum = 3; // 3: dim id
         newShape.dims[0] = oldShape.dims[0];
         newShape.dims[1] = 1;
         newShape.dims[2] = oldShape.dims[1]; // 2:dim id
     };
     keyRepeatNode.operation = new atb_speed::common::RepeatOperation("RepeatNode", kvRepeatParam);
     opGraph.nodes.push_back(keyRepeatNode);
 
     atb::Node kCatNode;
     atb::infer::ConcatParam kCatParam;
     kCatParam.concatDim = 2; // 2: dim id
     kCatNode.inTensorIds = {
         GetTensorIdx(tensorMap, "intermediate_k_nope"), GetTensorIdx(tensorMap, "rope_k_o_repeat")};
     kCatNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_k_mha")};
     kCatNode.inTensorReshapeFuncs.resize(kCatNode.inTensorIds.size());
     kCatNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
         newShape.dimNum = 3; // 3: dim id
         newShape.dims[0] = oldShape.dims[0];
         newShape.dims[1] = param.selfAttentionParam.headNum;
         newShape.dims[2] = param.qkNopeHeadDim; // 2:dim id
     };
     CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(kCatParam, &kCatNode.operation));
     opGraph.nodes.push_back(kCatNode);
     ATB_SPEED_LOG_DEBUG("MLA kCatNode prefill calculation success");
     return atb::NO_ERROR;
 }
 
 
 atb::Status AddKGatherTp2SpNode(atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
 {
     atb::Node nopeGatherNode;
     atb::infer::GatherParam nopeGatherParam;
     CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(nopeGatherParam, &nopeGatherNode.operation));
 
     nopeGatherNode.inTensorIds = GetTensorIdxList(tensorMap, {"intermediate_kv", "in_k_sp_gather_indices"});
     nopeGatherNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_kv_sp")};
     opGraph.nodes.push_back(nopeGatherNode);
 
     atb::Node ropeGatherNode;
     atb::infer::GatherParam ropeGatherParam;
     CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(ropeGatherParam, &ropeGatherNode.operation));
 
     ropeGatherNode.inTensorIds = GetTensorIdxList(tensorMap, {"rope_k_o", "in_k_sp_gather_indices"});
     ropeGatherNode.outTensorIds = {GetTensorIdx(tensorMap, "rope_k_o_sp")};
     opGraph.nodes.push_back(ropeGatherNode);
     
     return atb::NO_ERROR;
 }
 
 
 template <typename NormParamType>
 atb::Status AddReshapeAndCacheNode(const LatentAttentionParam<NormParamType> &param,
     atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
 {
     if (param.hasAttnInnerSp && param.isPrefill) {
         CHECK_OPERATION_STATUS_RETURN(AddKGatherTp2SpNode(opGraph, tensorMap));
     }
     atb::Node reshapeAndCacheNode;
     CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(param.reshapeCacheParm, &reshapeAndCacheNode.operation));
     reshapeAndCacheNode.inTensorIds = {
         EnableFA3Quant(param) ? GetTensorIdx(tensorMap, "intermediate_kv_int8") :
         GetTensorIdx(tensorMap, "intermediate_kv"), GetTensorIdx(tensorMap, "rope_k_o"),
         GetTensorIdx(tensorMap, "in_k_cache"), GetTensorIdx(tensorMap, "in_k_rope_cache"),
         GetTensorIdx(tensorMap, "in_slots_in_pa_or_logn_in_fa"),
     };
     if (param.hasAttnInnerSp && param.isPrefill) {
         reshapeAndCacheNode.inTensorIds[0] = GetTensorIdx(tensorMap, "intermediate_kv_sp");
         reshapeAndCacheNode.inTensorIds[1] = GetTensorIdx(tensorMap, "rope_k_o_sp");
     }
     reshapeAndCacheNode.outTensorIds = {
         GetTensorIdx(tensorMap, "in_k_cache"),
         GetTensorIdx(tensorMap, "in_k_rope_cache"),
     };
     reshapeAndCacheNode.inTensorReshapeFuncs.resize(reshapeAndCacheNode.inTensorIds.size());
     reshapeAndCacheNode.inTensorReshapeFuncs[1] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
         newShape.dimNum = 3; // 3: dim num
         newShape.dims[0] = oldShape.dims[0];
         newShape.dims[1] = 1;
         newShape.dims[2] = oldShape.dims[1]; // 2: dim id
     };
     opGraph.nodes.push_back(reshapeAndCacheNode);
     return atb::NO_ERROR;
 }
 
 
 template <typename NormParamType>
 atb::Status AddReprojVNode(const LatentAttentionParam<NormParamType> &param, atb::GraphParam &opGraph,
     std::map<std::string, uint32_t> &tensorMap)
 {
     atb::Node vReprojNode;
     atb_speed::common::FusionLinearParam vReprojNodeParam;
     vReprojNodeParam.isBF16 = param.isBF16;
     vReprojNodeParam.hasBias = param.selfAttnHasBias;
     vReprojNodeParam.quantType = GetLinearQuantType(
         param.denseQuantType == atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED \
             ? param.packQuantType : param.denseQuantType,
         param.attnLinearQuantType[KV_PROJ_B_FOR_V_LINEAR_INDEX], false);
     vReprojNodeParam.quantGroupSize = param.quantGroupSize;
     vReprojNodeParam.transposeType = false;
     vReprojNodeParam.enEin = true;
     CHECK_OPERATION_STATUS_RETURN(FusionLinear(vReprojNodeParam, &vReprojNode.operation));
     vReprojNode.inTensorIds = {
         GetTensorIdx(tensorMap, "reproj_o"), GetTensorIdx(tensorMap, "in_v_proj_b_for_o_weight"),
         GetTensorIdx(tensorMap, "in_v_proj_b_for_o_scale"), GetTensorIdx(tensorMap, "in_v_proj_b_for_o_offset"),
         GetTensorIdx(tensorMap, "in_v_proj_b_for_o_descale"), GetTensorIdx(tensorMap, "in_v_proj_b_for_o_bias"),
         GetTensorIdx(tensorMap, "in_v_proj_b_for_o_compress_idx"),
     };
     vReprojNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_self_attention")};
     opGraph.nodes.push_back(vReprojNode);
     ATB_SPEED_LOG_DEBUG("MLA reproj v calculation success");
     return atb::NO_ERROR;
 }
 
 
 template <typename NormParamType>
 atb::Status AddLAttnKProjBNode(const LatentAttentionParam<NormParamType> &param, atb::GraphParam &opGraph,
     std::map<std::string, uint32_t> &tensorMap)
 {
     atb::Node kProjBNode;
     atb_speed::common::FusionLinearParam kProjBNodeParam;
     kProjBNodeParam.isBF16 = param.isBF16;
     kProjBNodeParam.hasBias = param.selfAttnHasBias;
     kProjBNodeParam.quantType = GetLinearQuantType(
         param.denseQuantType == atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED
             ? param.packQuantType
             : param.denseQuantType,
         param.attnLinearQuantType[KV_PROJ_B_FOR_Q_LINEAR_INDEX], false);
     kProjBNodeParam.quantGroupSize = param.quantGroupSize;
     kProjBNodeParam.transposeType = param.attnLinearTransposeType[KV_PROJ_B_FOR_Q_LINEAR_INDEX];
     CHECK_OPERATION_STATUS_RETURN(FusionLinear(kProjBNodeParam, &kProjBNode.operation));
     kProjBNode.inTensorIds = {
         GetTensorIdx(tensorMap, "intermediate_kv"),
         GetTensorIdx(tensorMap, "in_k_proj_b_for_q_weight"),
         GetTensorIdx(tensorMap, "in_k_proj_b_for_q_scale"),
         GetTensorIdx(tensorMap, "in_k_proj_b_for_q_offset"),
         GetTensorIdx(tensorMap, "in_k_proj_b_for_q_descale"),
         GetTensorIdx(tensorMap, "in_k_proj_b_for_q_bias"),
         GetTensorIdx(tensorMap, "in_k_proj_b_for_q_compress_idx"),
     };
     kProjBNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_k_nope")};
 
     kProjBNode.inTensorReshapeFuncs.resize(kProjBNode.inTensorIds.size());
     kProjBNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
         newShape.dimNum = 2; // 2: dim id
         newShape.dims[0] = oldShape.dims[0];
         newShape.dims[1] = oldShape.dims[2]; // 2: dim id
     };
     kProjBNode.inTensorReshapeFuncs[1] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
         newShape.dimNum = 2; // 2: dim id
         newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
         newShape.dims[1] = oldShape.dims[2]; // 2: dim id
     };
 
     opGraph.nodes.push_back(kProjBNode);
     ATB_SPEED_LOG_DEBUG("MLA proj_k_b calculation success");
     return atb::NO_ERROR;
 }
 
 
 template <typename NormParamType>
 atb::Status AddLAttnVProjBNode(const LatentAttentionParam<NormParamType> &param, atb::GraphParam &opGraph,
     std::map<std::string, uint32_t> &tensorMap)
 {
     atb::Node transposeVProjBWeightNode;
     atb::infer::TransposeParam transposeVProjBWeightParam;
     transposeVProjBWeightNode.inTensorIds = {GetTensorIdx(tensorMap, "in_v_proj_b_for_o_weight")};
     transposeVProjBWeightNode.outTensorIds = {GetTensorIdx(tensorMap, "temp_v_proj_b")};
     if (param.isFA) {
         transposeVProjBWeightParam.perm = {0, 1, 3, 2};
     } else {
         transposeVProjBWeightParam.perm = {0, 2, 1};
     }
     CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(transposeVProjBWeightParam,
         &transposeVProjBWeightNode.operation));
     opGraph.nodes.push_back(transposeVProjBWeightNode);
 
     atb::Node vProjBNode;
     atb_speed::common::FusionLinearParam vProjBNodeParam;
     vProjBNodeParam.isBF16 = param.isBF16;
     vProjBNodeParam.hasBias = param.selfAttnHasBias;
     vProjBNodeParam.quantType = GetLinearQuantType(
         param.denseQuantType == atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED
             ? param.packQuantType
             : param.denseQuantType,
         param.attnLinearQuantType[KV_PROJ_B_FOR_V_LINEAR_INDEX], false);
     vProjBNodeParam.quantGroupSize = param.quantGroupSize;
     vProjBNodeParam.transposeType = param.attnLinearTransposeType[KV_PROJ_B_FOR_V_LINEAR_INDEX];
     CHECK_OPERATION_STATUS_RETURN(FusionLinear(vProjBNodeParam, &vProjBNode.operation));
     vProjBNode.inTensorIds = {
         GetTensorIdx(tensorMap, "intermediate_kv"),
         GetTensorIdx(tensorMap, "temp_v_proj_b"),
         GetTensorIdx(tensorMap, "in_v_proj_b_for_o_scale"),
         GetTensorIdx(tensorMap, "in_v_proj_b_for_o_offset"),
         GetTensorIdx(tensorMap, "in_v_proj_b_for_o_descale"),
         GetTensorIdx(tensorMap, "in_v_proj_b_for_o_bias"),
         GetTensorIdx(tensorMap, "in_v_proj_b_for_o_compress_idx"),
     };
     vProjBNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_v_mha")};
     vProjBNode.inTensorReshapeFuncs.resize(vProjBNode.inTensorIds.size());
     vProjBNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
         newShape.dimNum = 2; // 2: dim num
         newShape.dims[0] = oldShape.dims[0];
         newShape.dims[1] = oldShape.dims[2]; // 2: dim id
     };
     vProjBNode.inTensorReshapeFuncs[1] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
         newShape.dimNum = 2; // 2: dim num
         newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
         newShape.dims[1] = oldShape.dims[2]; // 2: dim id
     };
 
     opGraph.nodes.push_back(vProjBNode);
     ATB_SPEED_LOG_DEBUG("MLA proj_v_b calculation success");
     return atb::NO_ERROR;
 }
 
 template <typename NormParamType>
 atb::Status SetSelfOutLinearParallelParam(const LatentAttentionParam<NormParamType> &param,
     atb_speed::common::LinearParallelParam &selfOutLinearParam)
 {
     selfOutLinearParam.parallelType = atb_speed::common::ROW_PARALLEL;
     selfOutLinearParam.fusionLinearParam.isBF16 = param.isBF16;
     selfOutLinearParam.fusionLinearParam.hasBias = param.selfAttnHasBias && !selfOutLinearParam.biasAfterSync;
     selfOutLinearParam.fusionLinearParam.quantType = GetLinearQuantType(
         param.denseQuantType == atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED \
             ? param.packQuantType : param.denseQuantType,
         param.attnLinearQuantType[O_LINEAR_INDEX], false);
     if (selfOutLinearParam.fusionLinearParam.quantType == LinearQuantType::LINEAR_W8A8_QUANT &&
         param.enableExtraOprojTp) {
         selfOutLinearParam.fusionLinearParam.quantType = LinearQuantType::LINEAR_W8A8_DEQUANT;
     }
     selfOutLinearParam.fusionLinearParam.quantGroupSize = param.quantGroupSize;
     selfOutLinearParam.fusionLinearParam.transposeType = param.attnLinearTransposeType[O_LINEAR_INDEX];
     selfOutLinearParam.tensorParallelInfo = param.selfOutLinearTensorParallelInfo;
     selfOutLinearParam.supportLcoc = param.enableLcoc;
 
     selfOutLinearParam.innerTensorParallelInfo = param.selfOutLinearInnerTensorParallelInfo;
     selfOutLinearParam.innerTensorParallelInfoLCCL = param.selfOutLinearInnerTensorParallelInfoLCCL;
     selfOutLinearParam.fusionLinearParam.enablePrefetch = param.attnOprojPrefetch;
 
     if (param.selfOutLinearInnerTensorParallelInfo.rankIds.size() == 0) {
         std::stringstream ss;
         ss << "Cannot be devided by zero. Param attnOprojTpSize is zero!" << std::endl;
         throw std::runtime_error(ss.str());
     }
     selfOutLinearParam.innerTpShape = \
         param.qkNopeHeadDim * param.selfAttentionParam.headNum / \
         param.selfOutLinearInnerTensorParallelInfo.rankIds.size();
 
     selfOutLinearParam.fusionLinearParam.isPrefill = param.isPrefill;
     return atb::NO_ERROR;
 }
 
 template <typename NormParamType>
 atb::Status AddSelfOutLinearParallelNode(const LatentAttentionParam<NormParamType> &param,
     atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
 {
     atb::Node selfOutLinearParallelNode;
     atb_speed::common::LinearParallelParam selfOutLinearParam;
     SetSelfOutLinearParallelParam(param, selfOutLinearParam);
     CHECK_OPERATION_STATUS_RETURN(LinearParallel(selfOutLinearParam, &selfOutLinearParallelNode.operation));
     selfOutLinearParallelNode.inTensorIds = {
         GetTensorIdx(tensorMap, !param.enableExtraOprojTp ?
         "intermediate_self_attention" : \
         selfOutLinearParam.fusionLinearParam.quantType == LinearQuantType::LINEAR_W8A8_DEQUANT ? \
         "intermediate_self_attention_padding_quant" : "intermediate_self_attention_padding"),
         GetTensorIdx(tensorMap, "in_attn_out_weight"),
         GetTensorIdx(tensorMap, "in_attn_out_scale"),
         GetTensorIdx(tensorMap, "in_attn_out_offset"),
         GetTensorIdx(tensorMap, "in_attn_out_descale"),
         GetTensorIdx(tensorMap, "in_attn_out_bias"),
         GetTensorIdx(tensorMap, "in_attn_out_compress_idx"),
     };
     selfOutLinearParallelNode.inTensorReshapeFuncs.resize(selfOutLinearParallelNode.inTensorIds.size());
     if (!param.isFA) {
         selfOutLinearParallelNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
             SqueezeHeadNumHeadDim(oldShape, newShape);
         };
     }
     selfOutLinearParallelNode.outTensorIds = {GetTensorIdx(tensorMap, "out")};
     opGraph.nodes.push_back(selfOutLinearParallelNode);
     ATB_SPEED_LOG_DEBUG("MLA o_proj calculation success");
     return atb::NO_ERROR;
 }
 
 atb::Status AddQuantOprojNode(atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
 {
     // quant
     atb::Node inputQuantNode;
     atb::infer::ElewiseParam inputQuantParam;
     inputQuantParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_QUANT_PER_CHANNEL;
     CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(inputQuantParam, &inputQuantNode.operation));
     inputQuantNode.inTensorIds = GetTensorIdxList(tensorMap, {"intermediate_self_attention_padding",
         "in_attn_out_scale", "in_attn_out_offset"});
     inputQuantNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_self_attention_padding_quant")};
     opGraph.nodes.push_back(inputQuantNode);
     ATB_SPEED_LOG_DEBUG("MLA Quant O calculation success");
     return atb::NO_ERROR;
 }
 
 atb::Status AddOprojAllToAllPaddingNode(atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
 {
     atb::Node gatherNode;
     atb::infer::GatherParam gatherParam;
     CHECK_OPERATION_STATUS_RETURN(CreateOperation(gatherParam, &gatherNode.operation));
 
     gatherNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_self_attention",
                                                                              "in_attn_padding_idx"});
     gatherNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_self_attention_padding"});
     opGraph.nodes.push_back(gatherNode);
     ATB_SPEED_LOG_DEBUG("MLA Wo AllToAll Padding calculation success");
     return atb::NO_ERROR;
 }
 
 template <typename NormParamType>
 atb::Status SetTPAllGatherNode(const LatentAttentionParam<NormParamType> &param, atb::GraphParam &opGraph,
     std::map<std::string, uint32_t> tensorMap)
 {
     atb::Node allGatherNode;
     atb::infer::AllGatherParam allGatherParam;
     allGatherParam.rank = param.attnTpRank;
     allGatherParam.rankSize = param.attnTpSize;
     allGatherParam.backend = param.attnTpBackend;
     allGatherParam.commDomain = param.attnTpDomain;
     allGatherParam.rankTableFile = param.attnTpRankTableFile;
     allGatherParam.hcclComm = param.hcclComm;
     if (!FLAGS_enable_atb_comm_multiprocess) {
          allGatherParam.commMode = atb::infer::CommMode::COMM_MULTI_THREAD;
     }
     CreateOperation(allGatherParam, &allGatherNode.operation);
 
     allGatherNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"latent_qkv"});
     allGatherNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_qkv_all"});
 
     opGraph.nodes.push_back(allGatherNode);
     return atb::NO_ERROR;
 }
 
 template <typename NormParamType>
 atb::Status SetFFNUnPadding(const LatentAttentionParam<NormParamType> &param, atb::GraphParam &opGraph,
     std::map<std::string, uint32_t> tensorMap)
 {
     atb::Node gatherNode;
     atb::infer::GatherParam gatherParam;
     CHECK_OPERATION_STATUS_RETURN(CreateOperation(gatherParam, &gatherNode.operation));
 
     gatherNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, \
         {param.ffnAllGather ? "intermediate_qkv_all" : "latent_qkv",
         "in_ffn_unpadding_idx"});
     gatherNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_qkv_unpadding_all"});
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
 
 template <typename NormParamType>
 atb::Status Attention(const LatentAttentionParam<NormParamType> &param, atb::Operation **operation)
 {
     std::shared_ptr<int64_t> batchSizePtr = std::make_shared<int64_t>(0);
     atb::GraphParam opGraph;
     opGraph.name = "Attention";
     std::map<std::string, uint32_t> tensorMap = ConstructTensorMap(param,
         opGraph.inTensorNum, opGraph.outTensorNum, opGraph.internalTensorNum);
     ATB_SPEED_LOG_DEBUG("opGraph.inTensorNum " << opGraph.inTensorNum);
     ATB_SPEED_LOG_DEBUG("opGraph.outTensorNum " << opGraph.outTensorNum);
     ATB_SPEED_LOG_DEBUG("opGraph.internalTensorNum " << opGraph.internalTensorNum);
     // Preprocess
     CHECK_OPERATION_STATUS_RETURN(Preprocess(param, opGraph, tensorMap));
     // PA or MLA
     if (param.isPrefill) {
         CHECK_OPERATION_STATUS_RETURN(AddPaEncoderNode(param, opGraph, tensorMap));
     } else {
         if (param.hasAttnInnerSp) {
             CHECK_OPERATION_STATUS_RETURN(AddLAttnQCatNode(param, opGraph, tensorMap));
             CHECK_OPERATION_STATUS_RETURN(AddQAllGatherTp2Sp(param, opGraph, tensorMap));
             CHECK_OPERATION_STATUS_RETURN(AddQSplitNode(param, opGraph, tensorMap));
             CHECK_OPERATION_STATUS_RETURN(AddMlaDecoderNode(param, opGraph, tensorMap));
             CHECK_OPERATION_STATUS_RETURN(AddDecodeUpdate(param, opGraph, tensorMap));
         } else {
             CHECK_OPERATION_STATUS_RETURN(AddMlaDecoderNode(param, opGraph, tensorMap));
         }
         CHECK_OPERATION_STATUS_RETURN(AddReprojVNode(param, opGraph, tensorMap));
     }
     if (param.enableExtraOprojTp) {
         CHECK_OPERATION_STATUS_RETURN(AddOprojAllToAllPaddingNode(opGraph, tensorMap));
     }
     if (UseExtraQuant(param, O_LINEAR_INDEX) && param.enableExtraOprojTp) {
         CHECK_OPERATION_STATUS_RETURN(AddQuantOprojNode(opGraph, tensorMap));
     }
     CHECK_OPERATION_STATUS_RETURN(AddSelfOutLinearParallelNode(param, opGraph, tensorMap));
     opGraph.inferShapeFunc = [=]
                 (const atb::SVector<atb::TensorDesc> &inTensorDescs, atb::SVector<atb::TensorDesc> &outTensorDescs) {
         outTensorDescs.at(0) = inTensorDescs.at(0);
         if (param.enableQkvdownDp) {
             outTensorDescs.at(0).shape.dims[0] *= param.attnTpSize;
         }
         return atb::NO_ERROR;
     };
     CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(opGraph, operation));
     return atb::NO_ERROR;
 }
 
 template <typename NormParamType>
 atb::Status Preprocess(const LatentAttentionParam<NormParamType> &param,
     atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
 {
     if (param.qLoraRank > 0 && param.enableMlaPreprocess && !param.isPrefill) {
         CHECK_OPERATION_STATUS_RETURN(AddMlaPreprocessNode(param, opGraph, tensorMap));
     } else {
         CHECK_OPERATION_STATUS_RETURN(AddLAttnPreNormNode(param, opGraph, tensorMap));
         if (param.qLoraRank > 0) {
             CHECK_OPERATION_STATUS_RETURN(AddLAttnQKVProjNode(param, opGraph, tensorMap));
             if (param.enableQkvdownDp) {
                 CHECK_OPERATION_STATUS_RETURN(SetTPAllGatherNode(param, opGraph, tensorMap));
                 CHECK_OPERATION_STATUS_RETURN(SetFFNUnPadding(param, opGraph, tensorMap));
             }
             CHECK_OPERATION_STATUS_RETURN(AddSplitQKNode(param, opGraph, tensorMap));
             CHECK_OPERATION_STATUS_RETURN(AddLAttnQProjBNode(param, opGraph, tensorMap));
             CHECK_OPERATION_STATUS_RETURN(AddSplitQNode(param, opGraph, tensorMap));
             CHECK_OPERATION_STATUS_RETURN(AddLAttnKVNormNode(param, opGraph, tensorMap));
         } else {
             CHECK_OPERATION_STATUS_RETURN(AddLAttnQProjANode(param, opGraph, tensorMap));
             CHECK_OPERATION_STATUS_RETURN(AddSplitQNode(param, opGraph, tensorMap));
             CHECK_OPERATION_STATUS_RETURN(AddLAttnKVAProjNode(param, opGraph, tensorMap));
             CHECK_OPERATION_STATUS_RETURN(AddSplitKNode(param, opGraph, tensorMap));
             CHECK_OPERATION_STATUS_RETURN(AddLAttnKVNormNode(param, opGraph, tensorMap));
         }
         if (param.rotaryType != RotaryType::NO_ROTARY) {
             CHECK_OPERATION_STATUS_RETURN(AddLAttnRopeNode(param, opGraph, tensorMap));
         }
         if (param.isPrefill) {
             CHECK_OPERATION_STATUS_RETURN(AddLAttnKProjBNode(param, opGraph, tensorMap));
             CHECK_OPERATION_STATUS_RETURN(AddLAttnKCatPrefillNode(param, opGraph, tensorMap));
             CHECK_OPERATION_STATUS_RETURN(AddLAttnVProjBNode(param, opGraph, tensorMap));
             CHECK_OPERATION_STATUS_RETURN(AddLAttnQCatNode(param, opGraph, tensorMap));
             if (EnableFA3Quant(param)) {
                 CHECK_OPERATION_STATUS_RETURN(AddKVQuantNode(opGraph, tensorMap));
             }
         } else {
             CHECK_OPERATION_STATUS_RETURN(AddReprojQNode(param, opGraph, tensorMap));
         }
         CHECK_OPERATION_STATUS_RETURN(AddReshapeAndCacheNode(param, opGraph, tensorMap));
     }
     return atb::NO_ERROR;
 }
 
 template <typename NormParamType>
 atb::Status AddQAllGatherTp2Sp(const LatentAttentionParam<NormParamType> &param,
     atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
 {
     atb::Node allGatherNode;
     atb::infer::AllGatherParam allGatherParam;
     allGatherParam.rank = param.attnSpRank;
     allGatherParam.rankSize = param.attnSpSize;
     allGatherParam.backend = param.attnSpBackend;
     allGatherParam.rankTableFile = param.attnSpRankTableFile;
     allGatherParam.commDomain = param.attnSpDomain;
     allGatherParam.hcclComm = param.attnSpHcclComm;
    if (!FLAGS_enable_atb_comm_multiprocess) {
          allGatherParam.commMode = atb::infer::CommMode::COMM_MULTI_THREAD;
    }
     CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(allGatherParam, &allGatherNode.operation));
 
     allGatherNode.inTensorIds = GetTensorIdxList(tensorMap, {"intermediate_q"});
     allGatherNode.outTensorIds = GetTensorIdxList(tensorMap, {"intermediate_q_allgather_sp_t"});
     opGraph.nodes.push_back(allGatherNode);
 
     atb::Node transposeNode;
     atb::infer::TransposeParam transposeParam;
     transposeParam.perm = {1, 0, 2, 3}; // sp, B, N/tp, D -> B, sp, N/tp, D
     CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(transposeParam, &transposeNode.operation));
 
     transposeNode.inTensorIds = GetTensorIdxList(tensorMap, {"intermediate_q_allgather_sp_t"});
     transposeNode.outTensorIds = GetTensorIdxList(tensorMap, {"intermediate_q_allgather_sp"});
     opGraph.nodes.push_back(transposeNode);
     
     return atb::NO_ERROR;
 }
 
 template <typename NormParamType>
 atb::Status AddPaEncoderNode(
     const LatentAttentionParam<NormParamType> &param, atb::GraphParam &opGraph,
     std::map<std::string, uint32_t> &tensorMap)
 {
     atb::Node selfAttentionNode;
     CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(param.selfAttentionParam, &selfAttentionNode.operation));
     selfAttentionNode.inTensorIds = {
         GetTensorIdx(tensorMap, "intermediate_q"),
         GetTensorIdx(tensorMap, "intermediate_k_mha"),
         GetTensorIdx(tensorMap, "intermediate_v_mha"),
     };
     if (param.selfAttentionParam.maskType != atb::infer::SelfAttentionParam::MASK_TYPE_UNDEFINED) {
         selfAttentionNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_attention_mask"));
     }
     selfAttentionNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_seq_len"));
     selfAttentionNode.inTensorReshapeFuncs.resize(selfAttentionNode.inTensorIds.size());
     selfAttentionNode.inTensorReshapeFuncs[2] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
         newShape.dimNum = 3; // 3: dim num
         newShape.dims[0] = oldShape.dims[0];
         newShape.dims[1] = param.selfAttentionParam.headNum;
         newShape.dims[2] = param.qkNopeHeadDim; // 2: dim id
     };
     selfAttentionNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_self_attention")};
     opGraph.nodes.push_back(selfAttentionNode);
     ATB_SPEED_LOG_DEBUG("PA encoder calculation success");
     return atb::NO_ERROR;
 }
 
 template <typename NormParamType>
 atb::Status AddQSplitNode(const LatentAttentionParam<NormParamType> &param, atb::GraphParam &opGraph,
     std::map<std::string, uint32_t> &tensorMap)
 {
     // [B, N, dc+dr] --> [B, N, dc]  [B, N, dr]
     atb::Node splitNode;
     atb::infer::SplitParam splitParam;
     splitParam.splitDim = 2; // 2: position os dc+dr
     splitParam.splitSizes = {param.kvLoraRank, param.qkRopeHeadDim};
     CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(splitParam, &splitNode.operation));
     splitNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_q_allgather_sp")};
     splitNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_q_sp_nope"),
                               GetTensorIdx(tensorMap, "intermediate_q_sp_rope")};
     splitNode.inTensorReshapeFuncs.resize(splitNode.inTensorIds.size());
     splitNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
         newShape.dimNum = 3; // 3: [B, tp, N/tp, D] -> [B, N, D]
         newShape.dims[0] = oldShape.dims[0];
         newShape.dims[1] = oldShape.dims[1] * oldShape.dims[2]; // 2: dim id
         newShape.dims[2] = oldShape.dims[3]; // 2, 3: dim id
     };
     opGraph.nodes.push_back(splitNode);
     return atb::NO_ERROR;
 }
 
 template <typename NormParamType>
 atb::Status AddMlaDecoderNode(
     const LatentAttentionParam<NormParamType> &param, atb::GraphParam &opGraph,
     std::map<std::string, uint32_t> &tensorMap)
 {
     atb::Node selfAttentionNode;
     atb::infer::MultiLatentAttentionParam multiLatentAttentionParam;
     multiLatentAttentionParam.headNum = param.pageAttentionParam.headNum;
     multiLatentAttentionParam.qkScale = param.pageAttentionParam.qkScale;
     multiLatentAttentionParam.kvHeadNum = param.pageAttentionParam.kvHeadNum;
     multiLatentAttentionParam.maskType = atb::infer::MultiLatentAttentionParam::MaskType::UNDEFINED;
     multiLatentAttentionParam.calcType = param.hasAttnInnerSp ?
         atb::infer::MultiLatentAttentionParam::CalcType::CALC_TYPE_RING :
         atb::infer::MultiLatentAttentionParam::CalcType::CALC_TYPE_UNDEFINED;
     if (param.pageAttentionParam.maskType == atb::infer::PagedAttentionParam::MaskType::MASK_TYPE_SPEC) {
         multiLatentAttentionParam.maskType = atb::infer::MultiLatentAttentionParam::MaskType::MASK_TYPE_SPEC;
     } else if (param.pageAttentionParam.maskType == atb::infer::PagedAttentionParam::MaskType::MASK_TYPE_NORM) {
         multiLatentAttentionParam.maskType = atb::infer::MultiLatentAttentionParam::MaskType::MASK_TYPE_MASK_FREE;
     }
     if (param.pageAttentionParam.calcType == atb::infer::PagedAttentionParam::CalcType::CALC_TYPE_SPEC) {
         multiLatentAttentionParam.calcType = atb::infer::MultiLatentAttentionParam::CalcType::CALC_TYPE_SPEC;
     }
     if (EnableFA3Quant(param)) {
         multiLatentAttentionParam.cacheMode = atb::infer::MultiLatentAttentionParam::CacheMode::INT8_NZCACHE;
     } else if (param.isNzCache) {
         multiLatentAttentionParam.cacheMode = atb::infer::MultiLatentAttentionParam::CacheMode::NZCACHE;
     } else {
         multiLatentAttentionParam.cacheMode = atb::infer::MultiLatentAttentionParam::CacheMode::KROPE_CTKV;
     }
 
     CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(multiLatentAttentionParam, &selfAttentionNode.operation));
     selfAttentionNode.inTensorIds = {
         GetTensorIdx(tensorMap, param.enableMlaPreprocess ? "intermediate_q_nope" : "reproj_nope_q"),
         GetTensorIdx(tensorMap, param.enableMlaPreprocess ? "intermediate_q_rope" : "rope_q_o"),
         GetTensorIdx(tensorMap, "in_k_cache"),
         GetTensorIdx(tensorMap, "in_k_rope_cache"),
         GetTensorIdx(tensorMap, "in_block_tables"),
         GetTensorIdx(tensorMap, "in_seq_len")
     };
     // reshape
     selfAttentionNode.inTensorReshapeFuncs.resize(selfAttentionNode.inTensorIds.size());
     selfAttentionNode.inTensorReshapeFuncs[1] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
         if (oldShape.dimNum == 2) { // 2: dim num
             newShape.dimNum = 3; // 3: dim num
             newShape.dims[0] = oldShape.dims[0];
             newShape.dims[1] = param.selfAttentionParam.headNum;
             newShape.dims[2] = oldShape.dims[1] / newShape.dims[1]; // 2: dim num
         } else {
             newShape = oldShape;
         }
     };
     if (param.pageAttentionParam.maskType != atb::infer::PagedAttentionParam::MaskType::UNDEFINED) {
         selfAttentionNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_attention_mask"));
     }
     if (param.pageAttentionParam.calcType == atb::infer::PagedAttentionParam::CalcType::CALC_TYPE_SPEC) {
         selfAttentionNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_q_len"));
     }
     if (EnableFA3Quant(param)) {
         selfAttentionNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_qk_descale"));
         selfAttentionNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "fa3_v_quant_scale"));
     }
     if (!param.enableMlaPreprocess) {
         selfAttentionNode.inTensorReshapeFuncs.resize(selfAttentionNode.inTensorIds.size());
         selfAttentionNode.inTensorReshapeFuncs[1] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
             newShape.dimNum = 3; // 3: dimNum
             newShape.dims[0] = oldShape.dims[0];
             newShape.dims[1] = param.selfAttentionParam.headNum;
             newShape.dims[2] = param.qkRopeHeadDim; // 2: dim id
         };
     }
     selfAttentionNode.outTensorIds = {GetTensorIdx(tensorMap, "reproj_o")};
 
     if (param.hasAttnInnerSp) {
         selfAttentionNode.inTensorIds.at(0) = GetTensorIdx(tensorMap, "intermediate_q_sp_nope");
         selfAttentionNode.inTensorIds.at(1) = GetTensorIdx(tensorMap, "intermediate_q_sp_rope");
         selfAttentionNode.inTensorIds.at(5) = GetTensorIdx(tensorMap, "in_seq_len_sp"); // 5:position of in_seq_len_sp
         selfAttentionNode.outTensorIds = GetTensorIdxList(tensorMap, {"intermediate_go", "intermediate_lse"});
     }
     
     opGraph.nodes.push_back(selfAttentionNode);
     ATB_SPEED_LOG_DEBUG("MLA decoder calculation success");
     return atb::NO_ERROR;
 }
 
 template atb::Status Preprocess(
     const LatentAttentionParam<atb::infer::RmsNormParam> &param, atb::GraphParam &opGraph,
     std::map<std::string, uint32_t> &tensorMap);
 template atb::Status AddPaEncoderNode(
     const LatentAttentionParam<atb::infer::RmsNormParam> &param, atb::GraphParam &opGraph,
     std::map<std::string, uint32_t> &tensorMap);
 template atb::Status AddMlaDecoderNode(
     const LatentAttentionParam<atb::infer::RmsNormParam> &param, atb::GraphParam &opGraph,
     std::map<std::string, uint32_t> &tensorMap);
 
 
 template std::map<std::string, uint32_t> ConstructTensorMap(
     const LatentAttentionParam<atb::infer::RmsNormParam> &param,
     uint32_t &inTensorNum, uint32_t &outTensorNum, uint32_t &internalTensorNum);
 template atb::Status AddLAttnPreNormNode(const LatentAttentionParam<atb::infer::RmsNormParam> &param,
     atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap);
 template atb::Status AddLAttnQProjANode(const LatentAttentionParam<atb::infer::RmsNormParam> &param,
     atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap);
 template atb::Status AddLAttnQProjBNode(const LatentAttentionParam<atb::infer::RmsNormParam> &param,
     atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap);
 template atb::Status AddSplitQNode(const LatentAttentionParam<atb::infer::RmsNormParam> &param,
     atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap);
 template atb::Status AddReprojQNode(const LatentAttentionParam<atb::infer::RmsNormParam> &param,
     atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap);
 template atb::Status AddLAttnQCatNode(const LatentAttentionParam<atb::infer::RmsNormParam> &param,
     atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap);
 template atb::Status AddLAttnKVAProjNode(const LatentAttentionParam<atb::infer::RmsNormParam> &param,
     atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap);
 template atb::Status AddSplitKNode(const LatentAttentionParam<atb::infer::RmsNormParam> &param,
     atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap);
 template atb::Status AddLAttnKVNormNode(const LatentAttentionParam<atb::infer::RmsNormParam> &param,
     atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap);
 template atb::Status AddLAttnRopeNode(const LatentAttentionParam<atb::infer::RmsNormParam> &param,
     atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap);
 template atb::Status AddReshapeAndCacheNode(const LatentAttentionParam<atb::infer::RmsNormParam> &param,
     atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap);
 template atb::Status AddLAttnKCatPrefillNode(const LatentAttentionParam<atb::infer::RmsNormParam> &param,
     atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap);
 template atb::Status AddLAttnKProjBNode(const LatentAttentionParam<atb::infer::RmsNormParam> &param,
     atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap);
 template atb::Status AddLAttnVProjBNode(const LatentAttentionParam<atb::infer::RmsNormParam> &param,
     atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap);
 template atb::Status AddReprojVNode(const LatentAttentionParam<atb::infer::RmsNormParam> &param,
     atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap);
 template atb::Status SetSelfOutLinearParallelParam(const LatentAttentionParam<atb::infer::RmsNormParam> &param,
     atb_speed::common::LinearParallelParam &selfOutLinearParam);
 template atb::Status AddSelfOutLinearParallelNode(const LatentAttentionParam<atb::infer::RmsNormParam> &param,
     atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap);
 template atb::Status Attention(
     const LatentAttentionParam<atb::infer::RmsNormParam> &param, atb::Operation **operation);
 template atb::Status AddQAllGatherTp2Sp(const LatentAttentionParam<atb::infer::RmsNormParam> &param,
     atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap);
 template atb::Status AddQSplitNode(const LatentAttentionParam<atb::infer::RmsNormParam> &param,
     atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap);
 } // namespace deepseekV2
 } // namespace atb_speed