/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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
#include "parallel_lmhead_v2.h"

#include <atb/atb_infer.h>

#include "parallel_layer.h"

namespace atb_speed {
namespace common {

template <class T>
atb::Status CreateLmHeadLinearNode(const ParallelLmHeadAllToAllParam &param,
                                   atb::GraphParam &opGraph, T &config, size_t &nodeId)
{
    atb::Node &lmHeadLinearNode = opGraph.nodes.at(nodeId++);
    atb::infer::LinearParam lmHeadLinearParam;
    lmHeadLinearParam.transposeB = param.transposeB;
    lmHeadLinearParam.hasBias = false;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(lmHeadLinearParam, &lmHeadLinearNode.operation));
    lmHeadLinearNode.inTensorIds = {config.IN_HIDDENSTATES_ID, config.IN_WEIGHT_ID};
    lmHeadLinearNode.outTensorIds = {config.INTERMEDIATE_LMLINEAR_OUT_ID};
    
    return atb::NO_ERROR;
}

template <class T>
atb::Status CreateTransPose1Node(const ParallelLmHeadAllToAllParam &param,
                                 atb::GraphParam &opGraph, T &config, size_t &nodeId)
{
    atb::Node &transPose1Node = opGraph.nodes.at(nodeId++);
    atb::infer::TransposeParam transParam1;
    transParam1.perm = { 0, 2, 1 };
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(transParam1, &transPose1Node.operation));
    transPose1Node.inTensorIds = { config.INTERMEDIATE_LMLINEAR_OUT_ID };
    transPose1Node.outTensorIds = { config.INTERMEDIATE_TRANS1_OUT_ID };
    transPose1Node.inTensorReshapeFuncs.resize(transPose1Node.inTensorIds.size());
    transPose1Node.inTensorReshapeFuncs.at(0) = [param](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 3; // 3: rank, token, vocab_size
        newShape.dims[0] = param.rankSize;
        newShape.dims[1] = oldShape.dims[0] / param.rankSize;
        newShape.dims[2] = oldShape.dims[1]; // 2: vocab_size
    };
    return atb::NO_ERROR;
}

template <class T>
atb::Status CreateAllToAllNode(const ParallelLmHeadAllToAllParam &param,
                               atb::GraphParam &opGraph, T &config, size_t &nodeId)
{
    atb::Node &allToAllNode = opGraph.nodes.at(nodeId++);
    atb::infer::AllToAllParam allToAllParam;
    allToAllParam.rank = param.rank;
    allToAllParam.rankSize = param.rankSize;
    allToAllParam.backend = param.backend;
    allToAllParam.hcclComm = param.hcclComm;
    allToAllParam.commDomain = param.commDomain;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(allToAllParam, &allToAllNode.operation));
    allToAllNode.inTensorIds = {config.INTERMEDIATE_TRANS1_OUT_ID};
    allToAllNode.outTensorIds = {config.INTERMEDIATE_ALLTOALLTP_OUT_ID};
    allToAllNode.inTensorReshapeFuncs.resize(allToAllNode.inTensorIds.size());
    allToAllNode.inTensorReshapeFuncs.at(0) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2; // 2: rank* vocab_size, token
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
        newShape.dims[1] = oldShape.dims[2]; // 2: token
    };
    return atb::NO_ERROR;
}

template <class T>
atb::Status CreateTransPose2Node(const ParallelLmHeadAllToAllParam &param,
                                 atb::GraphParam &opGraph, T &config, size_t &nodeId)
{
    atb::Node &transPose2Node = opGraph.nodes.at(nodeId++);
    atb::infer::TransposeParam trans2Param;
    trans2Param.perm = { 1, 0 };
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(trans2Param, &transPose2Node.operation));
    transPose2Node.inTensorIds = { config.INTERMEDIATE_ALLTOALLTP_OUT_ID };
    transPose2Node.outTensorIds = { config.OUT_LOGITS_ID };

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
        atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        auto dimLast = inTensorDescs.at(0).shape.dimNum - 1;
        outTensorDescs.at(0).shape.dims[dimLast] = inTensorDescs.at(1).shape.dims[0] * param.rankSize;
        outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0] / param.rankSize;
        return atb::NO_ERROR;
    };
    return atb::NO_ERROR;
}

template <class T>
atb::Status CreateParallelLmHeadAllToAllBase(const ParallelLmHeadAllToAllParam &param,
                                             atb::Operation **operation, T config)
{
    atb::GraphParam opGraph;
    opGraph.inTensorNum = config.inTensorNum;
    opGraph.outTensorNum = config.outTensorNum;
    opGraph.internalTensorNum = config.interTensorNum;
    opGraph.nodes.resize(config.nodeCount);
    opGraph.name = "Parallel_LmHead";

    size_t nodeId = 0;
    CHECK_OPERATION_STATUS_RETURN(CreateLmHeadLinearNode(param, opGraph, config, nodeId));
    CHECK_OPERATION_STATUS_RETURN(CreateTransPose1Node(param, opGraph, config, nodeId));
    CHECK_OPERATION_STATUS_RETURN(CreateAllToAllNode(param, opGraph, config, nodeId));
    CHECK_OPERATION_STATUS_RETURN(CreateTransPose2Node(param, opGraph, config, nodeId));

    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
}

atb::Status ParallelLmHeadAllToAll(const ParallelLmHeadAllToAllParam &param, atb::Operation **operation)
{
    // 7, 1, 3, 4: in, out, inter, node
    return CreateParallelLmHeadAllToAllBase(param, operation, ParallelLmHeadAllToAllConfig(7, 1, 3, 4));
}
} // namespace common
} // namespace atb_speed