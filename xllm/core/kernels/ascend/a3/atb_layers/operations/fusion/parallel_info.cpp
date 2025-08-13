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
#include "atb_speed/utils/singleton.h"
#include "atb_speed/base/external_comm_manager.h"
#include "parallel_info.h"
#include <iostream>

namespace atb_speed {
namespace common {

void ParallelInfo::InitCommDomain(std::string commBackend)
{
    if (this->commDomain != "") {
        return;
    }

    this->backend = commBackend;

    // Assign commDomain by rankIds and rank
    this->commDomain = GetSingleton<ExternalCommManager>().GetCommDomain(
        this->groupId, this->rankIds, this->rank, this->backend, this->bufferSize);

    // Get hcclComm (only created when hccl backend is used and inference across multi nodes)
    this->hcclComm = GetSingleton<ExternalCommManager>().GetCommPtr(this->commDomain);
}

bool ParallelInfo::IsEnabled() const
{
    return this->rankIds.size() > 1;
}

std::string ParallelInfo::ToString() const
{
    std::stringstream ss;
    // ss << "ParallelInfo: rank: " << this->rank
    //     << ", rankIds: " << this->rankIds
    //     << ", groupId: " << this->groupId
    //     << ", backend: " << this->backend
    //     << ", commDomain: " << this->commDomain
    //     << ", hcclComm: " << this->hcclComm
    //     << ", bufferSize: " << this->bufferSize;
    return ss.str();
}

} // namespace common
} // namesapce atb_speed