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
#include "operations/fusion/norm/norm_linear.h"
#include "operations/fusion/attention/attention_edge.h"

#include "atb_speed/log.h"

namespace atb_speed {
    namespace common {

        static const uint64_t ROTARY_COEFF = 2;

        std::map<std::string, std::vector<std::string>> GetAttnInTensorCandidatesEdge()
        {
            std::map<std::string, std::vector<std::string>> attnInTensorCandidates = {
                {"default", {
                    "in_hidden_states", "in_input_norm_weight", "in_qkv_weight",
                    "in_attention_out_weight", "in_mlp_weight_0", "in_mlp_down_weight",
                    "in_post_attention_norm_weight", "in_attention_mask", "in_position_id",
                    "in_cos_emb", "in_sin_emb", "in_seq_len", "in_place_holder", "in_past_key",
                    "in_past_value"}
                },
                {"hasbias", {"in_qkv_bias"}
                },
            };
            return attnInTensorCandidates;
        }

        std::map<std::string, std::vector<std::string>> GetQuantAttnInTensorCandidatesEdge()
        {
            std::map<std::string, std::vector<std::string>> attnInTensorCandidates = {
                {"default", {
                    "in_hidden_states", "in_input_norm_weight", "in_qkv_weight", "in_qkv_weight_input_scale",
                    "in_qkv_weight_input_offset", "in_qkv_weight_deq_scale", "in_qkv_weight_quant_bias",
                    "in_attention_out_weight", "in_attention_out_weight_input_scale",
                    "in_attention_out_weight_input_offset",
                    "in_attention_out_weight_deq_scale", "in_attention_out_weight_quant_bias", "in_attention_mask",
                    "in_position_id", "in_cos_emb", "in_sin_emb", "in_seq_len", "in_place_holder", "in_past_key",
                    "in_past_value"}
                },
            };
            return attnInTensorCandidates;
        }

        std::map<std::string, std::vector<std::string>> GetAttnIntermediateTensorCandidatesEdge()
        {
            std::map<std::string, std::vector<std::string>> attnIntermediateTensorCandidates = {
                {"default", {
                    "intermediate_qkv_mixed_linear_out", "internal_q_scaled_out",
                    "internal_bmm_q_k_out", "internal_attention_scores",
                    "internal_attention_probs", "internal_k_split", "internal_v_split",
                    "internal_k_rope", "internal_bmm_v_out", "intermediate_input_norm_out",
                    "internal_q_split", "internal_q_rope", "internal_q_rope_transpose",
                    "internal_bmm_v_out_transpose"}
                },
                {"decode", {
                    "internal_k_rope_transpose", "out_present_value_transpose"}
                },
                {"gqa", {
                    "internal_key", "internal_value"}
                },
                {"quant", {
                    "intermediate_qkv_linear_input_quant", "internal_bmm_v_out_quant"}
                },
            };
            return attnIntermediateTensorCandidates;
        }

        std::map<std::string, std::vector<std::string>> GetAttnOutTensorCandidatesEdge()
        {
            std::map<std::string, std::vector<std::string>> attnOutTensorCandidates = {
                {"default", {
                    "out_attention", "out_present_key", "out_present_value"}
                },
            };
            return attnOutTensorCandidates;
        }

        std::map<std::string, uint32_t> ConstructTensorMap(const AttentionParam &param,
                                                           uint32_t &inTensorNum,
                                                           uint32_t &outTensorNum,
                                                           uint32_t &internalTensorNum)
        {
            bool isPrefill = param.isPrefill;
            bool isGQA = param.isGQA;
            bool isQuant = param.isQuant;
            bool hasBias = param.isHasQKVBias;

            std::map<std::string, std::vector<std::string>> attnInTensorCandidates  = {};
            if (isQuant) {
                attnInTensorCandidates = GetQuantAttnInTensorCandidatesEdge();
            } else {
                attnInTensorCandidates = GetAttnInTensorCandidatesEdge();
            }

            auto attnIntermediateTensorCandidates = GetAttnIntermediateTensorCandidatesEdge();
            auto attnOutTensorCandidates = GetAttnOutTensorCandidatesEdge();

            std::vector<std::string> inTensorList = {};
            std::vector<std::string> intermediateTensorList = {};
            std::vector<std::string> outTensorList = {};

            AddTensorToList(attnInTensorCandidates, "default", inTensorList);
            AddTensorToList(attnIntermediateTensorCandidates, "default", intermediateTensorList);
            AddTensorToList(attnOutTensorCandidates, "default", outTensorList);

            if (hasBias && !isQuant) {
                AddTensorToList(attnInTensorCandidates, "hasbias", inTensorList);
            }
            if (!isPrefill) {
                AddTensorToList(attnIntermediateTensorCandidates, "decode", intermediateTensorList);
            }
            if (isGQA) {
                AddTensorToList(attnIntermediateTensorCandidates, "gqa", intermediateTensorList);
            }
            if (isQuant) {
                AddTensorToList(attnIntermediateTensorCandidates, "quant", intermediateTensorList);
            }

            inTensorNum = inTensorList.size();
            outTensorNum = outTensorList.size();
            internalTensorNum = intermediateTensorList.size();

            return GetTensorMap(inTensorList, outTensorList, intermediateTensorList);
        }

        atb::Status RmsNormNode(const AttentionParam &param, atb::GraphParam &opGraph,
                                std::map<std::string, uint32_t> &tensorMap)
        {
            atb::Node inputNormNode;
            atb::infer::RmsNormParam rmsNormParam;
            rmsNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
            rmsNormParam.normParam.epsilon = param.normEps;
            CreateOperation(rmsNormParam, &inputNormNode.operation);
            inputNormNode.inTensorIds = {
                    GetTensorIdx(tensorMap, "in_hidden_states"), GetTensorIdx(tensorMap, "in_input_norm_weight")
            };
            inputNormNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_input_norm_out")};
            opGraph.nodes.push_back(inputNormNode);
            return atb::NO_ERROR;
        }

        atb::Status QuantQkvInput(atb::GraphParam &opGraph,
                                  std::map<std::string, uint32_t> &tensorMap)
        {
            atb::Node quantNode;
            atb::infer::ElewiseParam quantParam;
            quantParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_QUANT_PER_CHANNEL;
            CREATE_OPERATION(quantParam, &quantNode.operation);
            quantNode.inTensorIds = { GetTensorIdx(tensorMap, "intermediate_input_norm_out"),
                                      GetTensorIdx(tensorMap, "in_qkv_weight_input_scale"),
                                      GetTensorIdx(tensorMap, "in_qkv_weight_input_offset")
            };
            quantNode.outTensorIds = { GetTensorIdx(tensorMap, "intermediate_qkv_linear_input_quant") };
            opGraph.nodes.push_back(quantNode);
            return atb::NO_ERROR;
        }

        atb::Status QkvLinearNode(const AttentionParam &param, atb::GraphParam &opGraph,
                                  std::map<std::string, uint32_t> &tensorMap)
        {
            atb::Node qkvLinearNode;
            atb::infer::LinearParam linearParam;
            linearParam.hasBias = false;
            linearParam.transposeA = false;
            linearParam.transposeB = true;

            if (param.isQuant) {
                QuantQkvInput(opGraph, tensorMap);
                linearParam.hasBias = true;
                linearParam.outDataType = ACL_FLOAT16;
                CreateOperation(linearParam, &qkvLinearNode.operation);
                qkvLinearNode.inTensorIds = {
                        GetTensorIdx(tensorMap, "intermediate_qkv_linear_input_quant"),
                        GetTensorIdx(tensorMap, "in_qkv_weight"),
                        GetTensorIdx(tensorMap, "in_qkv_weight_quant_bias"),
                        GetTensorIdx(tensorMap, "in_qkv_weight_deq_scale")
                };
                qkvLinearNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_qkv_mixed_linear_out")};
                opGraph.nodes.push_back(qkvLinearNode);
            } else {
                CreateOperation(linearParam, &qkvLinearNode.operation);
                qkvLinearNode.inTensorIds = {
                        GetTensorIdx(tensorMap, "intermediate_input_norm_out"), GetTensorIdx(tensorMap, "in_qkv_weight")
                };
                qkvLinearNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_qkv_mixed_linear_out")};
                opGraph.nodes.push_back(qkvLinearNode);
            }

            return atb::NO_ERROR;
        }

        atb::Status QkvBiasNode(atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
        {
            atb::Node qkvAddBiasNode;
            atb::infer::ElewiseParam addParam;
            addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
            CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(addParam, &qkvAddBiasNode.operation));
            qkvAddBiasNode.inTensorIds =
            atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_qkv_mixed_linear_out", "in_qkv_bias"});
            qkvAddBiasNode.outTensorIds =
            atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_qkv_mixed_linear_out"});
            opGraph.nodes.push_back(qkvAddBiasNode);
            return atb::NO_ERROR;
        }

        atb::Status QkvSplitMHANode(atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
        {
            static const int ATTENTION_GROUPS = 3;
            atb::Node splitQkvNode;
            atb::infer::SplitParam splitParam = { -1, ATTENTION_GROUPS };
            CREATE_OPERATION(splitParam, &splitQkvNode.operation);
            splitQkvNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_qkv_mixed_linear_out")};
            splitQkvNode.outTensorIds = {
                    GetTensorIdx(tensorMap, "internal_q_split"), GetTensorIdx(tensorMap, "internal_k_split"),
                    GetTensorIdx(tensorMap, "internal_v_split"),
            };
            opGraph.nodes.push_back(splitQkvNode);

            return atb::NO_ERROR;
        }

        atb::Status QkvSplitGQANode(const AttentionParam &param, atb::GraphParam &opGraph,
                                    std::map<std::string, uint32_t> &tensorMap)
        {
            static const int HEAD_SIZE = param.hiddenSize;
            static const int ATTENTION_GROUPS = param.numAttentionHeads / param.numKeyValueHeads;
            atb::Node sliceQNode;
            atb::infer::SliceParam sliceQNodeParam;
            sliceQNodeParam.offsets = {0, 0, 0};
            sliceQNodeParam.size = {-1, -1, HEAD_SIZE};
            CREATE_OPERATION(sliceQNodeParam, &sliceQNode.operation);
            sliceQNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_qkv_mixed_linear_out")};
            sliceQNode.outTensorIds = {GetTensorIdx(tensorMap, "internal_q_split")};
            opGraph.nodes.push_back(sliceQNode);

            atb::Node sliceKVNode;
            atb::infer::SliceParam sliceKVNodeParam;
            sliceKVNodeParam.offsets = {0, 0, HEAD_SIZE};
            sliceKVNodeParam.size = {-1, -1, (HEAD_SIZE / ATTENTION_GROUPS) * 2 };
            CREATE_OPERATION(sliceKVNodeParam, &sliceKVNode.operation);
            sliceKVNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_qkv_mixed_linear_out")};
            sliceKVNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_qkv_mixed_linear_out")};
            opGraph.nodes.push_back(sliceKVNode);

            atb::Node splitKvNode;
            atb::infer::SplitParam splitParam2 = { -1, 2 };
            CREATE_OPERATION(splitParam2, &splitKvNode.operation);
            splitKvNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_qkv_mixed_linear_out")};
            splitKvNode.outTensorIds =  {
                    GetTensorIdx(tensorMap, "internal_k_split"),
                    GetTensorIdx(tensorMap, "internal_v_split"),
            };
            opGraph.nodes.push_back(splitKvNode);
            return atb::NO_ERROR;
        }

        atb::Status RepeatKVNode(const AttentionParam &param, atb::GraphParam &opGraph,
                                 std::map<std::string, uint32_t> &tensorMap)
        {
            static const int ATTENTION_GROUPS = param.numAttentionHeads / param.numKeyValueHeads;
            atb::Node expandKNode;
            atb::infer::RepeatParam expandKParam;
            expandKParam.multiples = {1, 1, ATTENTION_GROUPS, 1};
            CreateOperation(expandKParam, &expandKNode.operation);
            expandKNode.inTensorIds = {GetTensorIdx(tensorMap, "out_present_key")};
            expandKNode.outTensorIds = {GetTensorIdx(tensorMap, "internal_key")};
            opGraph.nodes.push_back(expandKNode);

            atb::Node expandVNode;
            atb::infer::RepeatParam expandVParam;
            expandVParam.multiples = {1, 1, ATTENTION_GROUPS, 1};
            CreateOperation(expandVParam, &expandVNode.operation);
            expandVNode.inTensorIds = {GetTensorIdx(tensorMap, "out_present_value")};
            expandVNode.outTensorIds = {GetTensorIdx(tensorMap, "internal_value")};
            opGraph.nodes.push_back(expandVNode);

            return atb::NO_ERROR;
        }

        atb::Status PermuteNode(const AttentionParam &param, atb::GraphParam &opGraph,
                                std::map<std::string, uint32_t> &tensorMap)
        {
            static const uint64_t NUM_KEY_VALUES_HEADS = param.numKeyValueHeads;

            atb::Node permuteQNode;
            atb::infer::TransposeParam permuteSeqHnHsParam;
            permuteSeqHnHsParam.perm = {0, 2, 1, 3};
            CreateOperation(permuteSeqHnHsParam, &permuteQNode.operation);
            permuteQNode.inTensorIds = {GetTensorIdx(tensorMap, "internal_q_rope")};
            permuteQNode.outTensorIds = {GetTensorIdx(tensorMap, "internal_q_rope_transpose")};
            opGraph.nodes.push_back(permuteQNode);

            atb::Node permuteKNode;
            CreateOperation(permuteSeqHnHsParam, &permuteKNode.operation);
            permuteKNode.inTensorIds = {GetTensorIdx(tensorMap, "internal_k_rope")};

            if (param.isPrefill) {
                permuteKNode.outTensorIds = {GetTensorIdx(tensorMap, "out_present_key")};
            } else {
                permuteKNode.outTensorIds = {GetTensorIdx(tensorMap, "internal_k_rope_transpose")};
            }
            opGraph.nodes.push_back(permuteKNode);

            atb::Node permuteVNode;
            CREATE_OPERATION(permuteSeqHnHsParam, &permuteVNode.operation);
            permuteVNode.inTensorIds = {GetTensorIdx(tensorMap, "internal_v_split")};
            if (param.isPrefill) {
                permuteVNode.outTensorIds = {GetTensorIdx(tensorMap, "out_present_value")};
            } else {
                permuteVNode.outTensorIds = {GetTensorIdx(tensorMap, "out_present_value_transpose")};
            }

            permuteVNode.inTensorReshapeFuncs.resize(permuteVNode.inTensorIds.size());
            permuteVNode.inTensorReshapeFuncs[0] = [&](const atb::Dims &oldShape, atb::Dims &newShape) {
                newShape.dimNum = 4; // 扩展维度为4维
                newShape.dims[0] = oldShape.dims[0]; // bs
                newShape.dims[1] = oldShape.dims[1];
                newShape.dims[2] = NUM_KEY_VALUES_HEADS; // 第2维度按头数切分
                newShape.dims[3] = oldShape.dims[2] / NUM_KEY_VALUES_HEADS; // 第3维度将旧维度除以头数
            };
            opGraph.nodes.push_back(permuteVNode);
            return atb::NO_ERROR;
        }

        atb::Status ConcatNode(atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
        {
            atb::Node concatKeyNode;
            atb::infer::ConcatParam concatKeyParam = {2};
            CreateOperation(concatKeyParam, &concatKeyNode.operation);
            concatKeyNode.inTensorIds = {
                    GetTensorIdx(tensorMap, "in_past_key"), GetTensorIdx(tensorMap, "internal_k_rope_transpose")
            };
            concatKeyNode.outTensorIds = {GetTensorIdx(tensorMap, "out_present_key")};
            opGraph.nodes.push_back(concatKeyNode);

            atb::Node concatValueNode;
            atb::infer::ConcatParam concatValueParam = {2};
            CreateOperation(concatValueParam, &concatValueNode.operation);
            concatValueNode.inTensorIds = {
                    GetTensorIdx(tensorMap, "in_past_value"), GetTensorIdx(tensorMap, "out_present_value_transpose")
            };
            concatValueNode.outTensorIds = {GetTensorIdx(tensorMap, "out_present_value")};
            opGraph.nodes.push_back(concatValueNode);
            return atb::NO_ERROR;
        }

        atb::Status RopeNode(const AttentionParam &param, atb::GraphParam &opGraph,
                             std::map<std::string, uint32_t> &tensorMap)
        {
            static const uint64_t NUM_ATTENTION_HEADS = param.numAttentionHeads;
            static const uint64_t NUM_KEY_VALUE_HEADS = param.numKeyValueHeads;

            atb::Node ropeNode;
            atb::infer::RopeParam ropeParam;
            ropeParam.rotaryCoeff = ROTARY_COEFF;
            CREATE_OPERATION(ropeParam, &ropeNode.operation);
            ropeNode.inTensorIds = {
                    GetTensorIdx(tensorMap, "internal_q_split"), GetTensorIdx(tensorMap, "internal_k_split"),
                    GetTensorIdx(tensorMap, "in_cos_emb"), GetTensorIdx(tensorMap, "in_sin_emb"),
                    GetTensorIdx(tensorMap, "in_seq_len"),
            };
            ropeNode.outTensorIds = {
                    GetTensorIdx(tensorMap, "internal_q_rope"),
                    GetTensorIdx(tensorMap, "internal_k_rope"),
            };
            ropeNode.inTensorReshapeFuncs.resize(ropeNode.inTensorIds.size());
            ropeNode.inTensorReshapeFuncs[0] = [&](const atb::Dims &oldShape, atb::Dims &newShape) {
                newShape.dimNum = 4; // 扩展维度为4维
                newShape.dims[0] = oldShape.dims[0]; // bs
                newShape.dims[1] = oldShape.dims[1]; // seqlen
                newShape.dims[2] = NUM_ATTENTION_HEADS; // 将第2维度按头数切分
                // 为第3维度赋值
                newShape.dims[3] = oldShape.dims[2] / NUM_ATTENTION_HEADS; // 将旧维度2的大小除以注意力头数
            };

            ropeNode.inTensorReshapeFuncs[1] = [&](const atb::Dims &oldShape, atb::Dims &newShape) {
                newShape.dimNum = 4; // 扩展维度为4维
                newShape.dims[0] = oldShape.dims[0]; // 扩展维度
                newShape.dims[1] = oldShape.dims[1];
                newShape.dims[2] = NUM_KEY_VALUE_HEADS; // 将第2维度按头数切分
                // 为第3维度赋值
                newShape.dims[3] = oldShape.dims[2] / NUM_KEY_VALUE_HEADS; // 将旧维度2的大小除以注意力头数
            };
            opGraph.nodes.push_back(ropeNode);

            PermuteNode(param, opGraph, tensorMap);
            if (!param.isPrefill) {
                ConcatNode(opGraph, tensorMap);
            }
            return atb::NO_ERROR;
        }

        atb::Status ReshapeBmmQKNode(const AttentionParam &param, atb::Node &bmmQKNode)
        {
            static const bool IS_GQA = param.isGQA;
            static const int ATTENTION_GROUPS = param.numAttentionHeads / param.numKeyValueHeads;

            bmmQKNode.inTensorReshapeFuncs.resize(bmmQKNode.inTensorIds.size());
            bmmQKNode.inTensorReshapeFuncs[0] = [&](const atb::Dims &oldShape, atb::Dims &newShape) {
                newShape.dimNum = 3; // 扩展维度为3维

                newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
                newShape.dims[1] = oldShape.dims[2]; // 使旧维度的第2维度赋值给新维度的第1维
                newShape.dims[2] = oldShape.dims[3]; // 使旧维度的第3维度赋值给新维度的第2维
                // }
            };
            bmmQKNode.inTensorReshapeFuncs[1] = [&](const atb::Dims &oldShape, atb::Dims &newShape) {
                newShape.dimNum = 3; // 定义新维度为3
                if (IS_GQA) {
                    newShape.dims[0] = oldShape.dims[1] * ATTENTION_GROUPS;
                    newShape.dims[1] = oldShape.dims[2] / ATTENTION_GROUPS; // dims[2] / ATTENTION_GROUPS
                    newShape.dims[2] = oldShape.dims[3] ; // 使旧维度的第3维度赋值给新维度的第2维
                } else {
                    newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
                    newShape.dims[1] = oldShape.dims[2]; // 使旧维度的第2维度赋值给新维度的第1维
                    newShape.dims[2] = oldShape.dims[3]; // 使旧维度的第3维度赋值给新维度的第2维
                }
            };
            return atb::NO_ERROR;
        }

        atb::Status ReshapeAddMaskNode(const AttentionParam &param, atb::Node &addMaskNode)
        {
            static const uint64_t NUM_ATTENTION_HEADS = param.numAttentionHeads;

            addMaskNode.inTensorReshapeFuncs.resize(addMaskNode.inTensorIds.size());
            addMaskNode.inTensorReshapeFuncs[0] = [&](const atb::Dims &oldShape, atb::Dims &newShape) {
                newShape.dimNum = 4;                    // 扩展维度为4维
                newShape.dims[0] = oldShape.dims[0] / NUM_ATTENTION_HEADS; // 扩展维度
                newShape.dims[1] = NUM_ATTENTION_HEADS; // 按头数切分
                newShape.dims[2] = oldShape.dims[1]; // 使旧维度的第2维度赋值给新维度的第1维
                newShape.dims[3] = oldShape.dims[2]; // 使旧维度的第3维度赋值给新维度的第2维
            };
            addMaskNode.inTensorReshapeFuncs[1] = [&](const atb::Dims &oldShape, atb::Dims &newShape) {
                newShape.dimNum = 4;  // 扩展维度为4维
                newShape.dims[0] = 1; // 扩展维度
                newShape.dims[1] = 1; // 扩展维度
                newShape.dims[2] = oldShape.dims[0]; // 使旧维度的第2维度赋值给新维度的第0维
                newShape.dims[3] = oldShape.dims[1]; // 使旧维度的第3维度赋值给新维度的第1维
            };
            return atb::NO_ERROR;
        }

        atb::Status AttentionScoreNode(const AttentionParam &param, atb::GraphParam &opGraph,
                                       std::map<std::string, uint32_t> &tensorMap)
        {
            static const uint64_t HEAD_DIM = param.hiddenSize / param.numAttentionHeads;
            atb::Node bmmQKNode;
            atb::infer::LinearParam matmulParam;
            matmulParam.hasBias = false;
            matmulParam.transposeA = false;
            matmulParam.transposeB = true;
            CreateOperation(matmulParam, &bmmQKNode.operation);

            if (!param.isGQA) {
                bmmQKNode.inTensorIds = {
                        GetTensorIdx(tensorMap, "internal_q_rope_transpose"),
                        GetTensorIdx(tensorMap, "out_present_key"),
                };
            } else {
                bmmQKNode.inTensorIds = {
                        GetTensorIdx(tensorMap, "internal_q_rope_transpose"),
                        GetTensorIdx(tensorMap, "internal_key"),
                };
            }
            bmmQKNode.outTensorIds = {GetTensorIdx(tensorMap, "internal_bmm_q_k_out")};
            ReshapeBmmQKNode(param, bmmQKNode);
            opGraph.nodes.push_back(bmmQKNode);

            atb::Node mulsQNode;
            float scalingAttr = 1.0 / sqrt(HEAD_DIM);
            atb::infer::ElewiseParam scalingElewiseMulsParam;
            scalingElewiseMulsParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MULS;
            scalingElewiseMulsParam.mulsParam.varAttr = scalingAttr;
            CreateOperation(scalingElewiseMulsParam, &mulsQNode.operation);
            mulsQNode.inTensorIds = {GetTensorIdx(tensorMap, "internal_bmm_q_k_out")};
            mulsQNode.outTensorIds = {GetTensorIdx(tensorMap, "internal_q_scaled_out")};
            opGraph.nodes.push_back(mulsQNode);
            return atb::NO_ERROR;
        }

        atb::Status AttentionMaskNode(const AttentionParam &param, atb::GraphParam &opGraph,
                                      std::map<std::string, uint32_t> &tensorMap)
        {
            atb::Node addMaskNode;
            atb::infer::ElewiseParam addParam;
            addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
            CreateOperation(addParam, &addMaskNode.operation);
            addMaskNode.inTensorIds = {
                    GetTensorIdx(tensorMap, "internal_q_scaled_out"),
                    GetTensorIdx(tensorMap, "in_attention_mask"),
            };
            addMaskNode.outTensorIds = {GetTensorIdx(tensorMap, "internal_attention_scores")};
            ReshapeAddMaskNode(param, addMaskNode);
            opGraph.nodes.push_back(addMaskNode);
            return atb::NO_ERROR;
        }

        atb::Status SoftMaxNode(atb::GraphParam &opGraph,
                                std::map<std::string, uint32_t> &tensorMap)
        {
            atb::Node softMaxNode;
            atb::infer::SoftmaxParam softmaxParam = {{-1}};
            CreateOperation(softmaxParam, &softMaxNode.operation);
            softMaxNode.inTensorIds = {GetTensorIdx(tensorMap, "internal_attention_scores")};
            softMaxNode.outTensorIds = {GetTensorIdx(tensorMap, "internal_attention_probs")};
            opGraph.nodes.push_back(softMaxNode);
            return atb::NO_ERROR;
        }

        atb::Status ValueLinearNode(const AttentionParam &param, atb::GraphParam &opGraph,
                                    std::map<std::string, uint32_t> &tensorMap)
        {
            static const int64_t NUM_ATTENTION_HEADS = param.numAttentionHeads;
            static const int ATTENTION_GROUPS = param.numAttentionHeads / param.numKeyValueHeads;
            atb::Node bmmVNode;
            atb::infer::LinearParam matmulParam2;
            matmulParam2.hasBias = false;
            matmulParam2.transposeA = false;
            matmulParam2.transposeB = false;
            CreateOperation(matmulParam2, &bmmVNode.operation);
            if (param.isGQA) {
                bmmVNode.inTensorIds = {
                        GetTensorIdx(tensorMap, "internal_attention_probs"),
                        GetTensorIdx(tensorMap, "internal_value"),
                };
            } else {
                bmmVNode.inTensorIds = {
                        GetTensorIdx(tensorMap, "internal_attention_probs"),
                        GetTensorIdx(tensorMap, "out_present_value"),
                };
            }
            bmmVNode.outTensorIds = {GetTensorIdx(tensorMap, "internal_bmm_v_out")};
            bmmVNode.inTensorReshapeFuncs.resize(bmmVNode.inTensorIds.size());
            bmmVNode.inTensorReshapeFuncs[0] = [&](const atb::Dims &oldShape, atb::Dims &newShape) {
                newShape.dimNum = 3; // 扩展维度为3维
                newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
                newShape.dims[1] = oldShape.dims[2]; // 使旧维度的第2维度赋值给新维度的第1维
                newShape.dims[2] = oldShape.dims[3]; // 使旧维度的第3维度赋值给新维度的第2维
            };
            bmmVNode.inTensorReshapeFuncs[1] = [&](const atb::Dims &oldShape, atb::Dims &newShape) {
                newShape.dimNum = 3; // 扩展维度为3维
                newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
                newShape.dims[1] = oldShape.dims[2]; // 使旧维度的第2维度赋值给新维度的第1维
                newShape.dims[2] = oldShape.dims[3]; // 使旧维度的第3维度赋值给新维度的第2维
                if (NUM_ATTENTION_HEADS != newShape.dims[0]) {
                    newShape.dims[0] = newShape.dims[0] * ATTENTION_GROUPS;
                    newShape.dims[1] = newShape.dims[1] / ATTENTION_GROUPS;
                }
            };
            opGraph.nodes.push_back(bmmVNode);
            return atb::NO_ERROR;
        }

        atb::Status QuantValueInput(atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
        {
            atb::Node quantValueNode;
            atb::infer::ElewiseParam quantParam;
            quantParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_QUANT_PER_CHANNEL;
            CREATE_OPERATION(quantParam, &quantValueNode.operation);

            quantValueNode.inTensorIds = {
                    GetTensorIdx(tensorMap, "internal_bmm_v_out_transpose"),
                    GetTensorIdx(tensorMap, "in_attention_out_weight_input_scale"),
                    GetTensorIdx(tensorMap, "in_attention_out_weight_input_offset")
            };

            quantValueNode.outTensorIds = { GetTensorIdx(tensorMap, "internal_bmm_v_out_quant") };
            opGraph.nodes.push_back(quantValueNode);
            return atb::NO_ERROR;
        }
        atb::Status OutLinearPermuteNode(const AttentionParam &param, atb::GraphParam &opGraph,
                                         std::map<std::string, uint32_t> &tensorMap)
        {
            static const uint64_t NUM_ATTENTION_HEADS = param.numAttentionHeads;
            atb::Node permuteAttentionNode;
            atb::infer::TransposeParam permuteParam;
            permuteParam.perm = {0, 2, 1, 3};
            CREATE_OPERATION(permuteParam, &permuteAttentionNode.operation);
            permuteAttentionNode.inTensorIds = {GetTensorIdx(tensorMap, "internal_bmm_v_out")};
            permuteAttentionNode.outTensorIds = {GetTensorIdx(tensorMap, "internal_bmm_v_out_transpose")};
            permuteAttentionNode.inTensorReshapeFuncs.resize(permuteAttentionNode.inTensorIds.size());
            permuteAttentionNode.inTensorReshapeFuncs[0] = [&](const atb::Dims &oldShape, atb::Dims &newShape) {
                newShape.dimNum = 4; // 扩展维度为4维
                newShape.dims[0] = oldShape.dims[0] / NUM_ATTENTION_HEADS; // 扩展维度
                newShape.dims[1] = NUM_ATTENTION_HEADS; // 按头数切分
                newShape.dims[2] = oldShape.dims[1]; // 使旧维度的第2维度赋值给新维度的第1维
                newShape.dims[3] = oldShape.dims[2]; // 使旧维度的第3维度赋值给新维度的第2维
            };
            opGraph.nodes.push_back(permuteAttentionNode);
            return atb::NO_ERROR;
        }


        atb::Status OutLinearNode(const AttentionParam &param, atb::GraphParam &opGraph,
                                  std::map<std::string, uint32_t> &tensorMap)
        {
            OutLinearPermuteNode(param, opGraph, tensorMap);

            atb::Node outLinearNode;
            atb::infer::LinearParam outLinearParam;
            outLinearParam.hasBias = false;
            outLinearParam.transposeA = false;
            outLinearParam.transposeB = true;

            if (param.isQuant) {
                QuantValueInput(opGraph, tensorMap);
                outLinearParam.hasBias = true;
                outLinearParam.outDataType = ACL_FLOAT16;
                CreateOperation(outLinearParam, &outLinearNode.operation);
                outLinearNode.inTensorIds = {
                        GetTensorIdx(tensorMap, "internal_bmm_v_out_quant"),
                        GetTensorIdx(tensorMap, "in_attention_out_weight"),
                        GetTensorIdx(tensorMap, "in_attention_out_weight_quant_bias"),
                        GetTensorIdx(tensorMap, "in_attention_out_weight_deq_scale")
                };
            } else {
                CreateOperation(outLinearParam, &outLinearNode.operation);
                outLinearNode.inTensorIds = {
                        GetTensorIdx(tensorMap, "internal_bmm_v_out_transpose"),
                        GetTensorIdx(tensorMap, "in_attention_out_weight"),
                };
            }
            outLinearNode.outTensorIds = {GetTensorIdx(tensorMap, "out_attention")};
            outLinearNode.inTensorReshapeFuncs.resize(outLinearNode.inTensorIds.size());

            outLinearNode.inTensorReshapeFuncs[0] = [&](const atb::Dims &oldShape, atb::Dims &newShape) {
                newShape.dimNum = 2; // 合并维度为2维
                newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1]; // 将第0维与第1维相乘
                newShape.dims[1] = oldShape.dims[2] * oldShape.dims[3]; // 将第2维与第3维相乘
            };
            opGraph.nodes.push_back(outLinearNode);
            return atb::NO_ERROR;
        }

        atb::Status AttentionNode(const AttentionParam &param, atb::GraphParam &opGraph,
                                  std::map<std::string, uint32_t> &tensorMap)
        {
            RmsNormNode(param, opGraph, tensorMap);
            QkvLinearNode(param, opGraph, tensorMap);
            if (param.isHasQKVBias && !param.isQuant) {
                QkvBiasNode(opGraph, tensorMap);
            }
            if (param.isGQA) {
                QkvSplitGQANode(param, opGraph, tensorMap);
            } else {
                QkvSplitMHANode(opGraph, tensorMap);
            }
            RopeNode(param, opGraph, tensorMap);
            if (param.isGQA) {
                RepeatKVNode(param, opGraph, tensorMap);
            }
            AttentionScoreNode(param, opGraph, tensorMap);
            AttentionMaskNode(param, opGraph, tensorMap);
            SoftMaxNode(opGraph, tensorMap);
            ValueLinearNode(param, opGraph, tensorMap);
            OutLinearNode(param, opGraph, tensorMap);
            return atb::NO_ERROR;
        }

        atb::Status AttentionEdge(const AttentionParam &param, atb::Operation **operation)
        {
            static const uint64_t KEY_VALUES_HEADS_NUM = param.numKeyValueHeads;
            static const uint64_t HIDDEN_SIZE = param.hiddenSize;

            atb::GraphParam opGraph;
            opGraph.name = "Attention";
            std::map<std::string, uint32_t> tensorMap = ConstructTensorMap(param,
                                                                           opGraph.inTensorNum,
                                                                           opGraph.outTensorNum,
                                                                           opGraph.internalTensorNum);
            opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                         atb::SVector<atb::TensorDesc> &outTensorDescs) {
                outTensorDescs.at(0) = inTensorDescs.at(0);
                if (!param.isPrefill) {
                    // 将in_past_key赋值到outTensorDescs[1]
                    outTensorDescs.at(1) = inTensorDescs.at(GetTensorIdx(tensorMap, "in_past_key"));
                    outTensorDescs.at(1).shape.dims[2] += 1; // dims[2] + 1
                    // 将in_past_value赋值到outTensorDescs[2]
                    outTensorDescs.at(2) = inTensorDescs.at(GetTensorIdx(tensorMap, "in_past_value"));
                    outTensorDescs.at(2).shape.dims[2] += 1; // dims[2] + 1
                } else {
                    outTensorDescs.at(1) = inTensorDescs.at(0);
                    outTensorDescs.at(1).shape.dimNum = 4; // 扩展维度为4维
                    outTensorDescs.at(1).shape.dims[0] = inTensorDescs.at(0).shape.dims[0]; // bs
                    outTensorDescs.at(1).shape.dims[1] = KEY_VALUES_HEADS_NUM;
                    outTensorDescs.at(1).shape.dims[2] = inTensorDescs.at(0).shape.dims[1]; // 2: 第2维为seqlen
                    // 将HIDDEN_SIZE / KEY_VALUES_HEADS_NUM赋值到此处的第3维
                    outTensorDescs.at(1).shape.dims[3] = HIDDEN_SIZE / KEY_VALUES_HEADS_NUM;

                    outTensorDescs.at(2) = inTensorDescs.at(0); // 将inTensorDescs[0]赋值到outTensorDescs[2]
                    outTensorDescs.at(2).shape.dimNum = 4; // 扩展维度为4维
                    outTensorDescs.at(2).shape.dims[0] = inTensorDescs.at(0).shape.dims[0]; // 2: 0维为bs
                    outTensorDescs.at(2).shape.dims[1] = KEY_VALUES_HEADS_NUM; // 为第2个outTensorDescs第1维定义
                    outTensorDescs.at(2).shape.dims[2] = inTensorDescs.at(0).shape.dims[1]; // 2: 第2维为seqlen
                    // 为outTensor[2]第3维度赋值
                    outTensorDescs.at(2).shape.dims[3] = HIDDEN_SIZE / KEY_VALUES_HEADS_NUM;
                }
                return atb::NO_ERROR;
            };
            CHECK_OPERATION_STATUS_RETURN(AttentionNode(param, opGraph, tensorMap));
            CREATE_OPERATION(opGraph, operation);
            return atb::NO_ERROR;
        }
    } // namespace common
} // namespace atb_speed