/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

#include <cmath>
#include <cerrno>
#include "securec.h"
#include "atb_speed/base/external_comm_manager.h"
#include <iostream>

namespace atb_speed {

CommInfoCache::~CommInfoCache()
{
    if (this->hcclComm_ != nullptr) {
        auto ret = HcclCommDestroy(this->hcclComm_);
        if (ret != HCCL_SUCCESS) {
            std::cout<<"[ERROR] External Comm Manager: destroy hccl communication group failed, error: " << ret<<std::endl;
        }
    }
    this->hcclComm_ = nullptr;
}

std::string CommInfoCache::ToString() const
{
    std::stringstream ss;
    // ss << "Cache Addr[" << this << "] cacheId_: " << cacheId_
    // << ", subCommRankId_: " << subCommRankId_
    // << ", rankIds_: " << rankIds_
    // << ", bufferSize_: " << bufferSize_
    // << ", backend_: " << backend_
    // << ", hcclComm_: " << hcclComm_;
    return ss.str();
}

bool AreVectorsEqual(const std::vector<uint32_t> &rankIdsA, const std::vector<uint32_t> &rankIdsB)
{
    if (rankIdsA.size() != rankIdsB.size()) {
        return false;
    }
    for (size_t i = 0; i < rankIdsA.size(); i++) {
        if (rankIdsA.at(i) != rankIdsB.at(i)) {
            return false;
        }
    }
    return true;
}

void ExternalCommManager::Init(std::string rankTableFile, uint32_t worldSize, uint32_t subCommRankId)
{
    
    this->worldSize_ = worldSize;

    if (this->globalComm_ != nullptr) {
        return;
    }

    this->rankTableFile_ = rankTableFile;
    char commName[128] = {};  // 128: max commName length
    if (this->rankTableFile_ == "") {
        this->globalComm_ = atb::Comm::CreateHcclComm(subCommRankId, 0, worldSize, commName);
    } else {
        this->globalComm_ = atb::Comm::CreateHcclCommByRankTableFile(subCommRankId, worldSize,
            rankTableFile.data(), commName);
    }
    if (this->globalComm_ == nullptr) {
        throw std::runtime_error("External Comm Manager: Create the hccl communication group failed. " \
            "export ASDOPS_LOG_LEVEL=ERROR, export ASDOPS_LOG_TO_STDOUT=1 to see more details. " \
            "Default log path is $HOME/atb/log. ");
    }

    std::vector<uint32_t> rankIds = {};
    for (uint32_t id = 0; id < worldSize; id++) {
        rankIds.push_back(id);
    }
    std::shared_ptr<CommInfoCache> commInfoCache = std::make_shared<CommInfoCache>();
    commInfoCache->cacheId_ = this->commMap_.size();
    commInfoCache->subCommRankId_ = subCommRankId;
    commInfoCache->rankIds_ = rankIds;
    commInfoCache->backend_ = "hccl";
    commInfoCache->hcclComm_ = this->globalComm_;
    char hcclCommName[128] = {};
    HcclGetCommName(this->globalComm_, hcclCommName);
    this->commMap_[std::string(hcclCommName)] = commInfoCache;
}

void ExternalCommManager::SetLcclCommDomainRange(int32_t lowerBound, int32_t upperBound)
{
    this->lcclCommDomainLowerBound_ = lowerBound;
    this->lcclCommDomainUpperBound_ = upperBound;
}

std::string ExternalCommManager::GetCommDomain(uint32_t groupId, const std::vector<uint32_t> &rankIds,
    uint32_t subCommRankId, std::string backend, uint32_t bufferSize, bool enableReuse)
{
    if (enableReuse) {
        std::map<std::string, std::shared_ptr<CommInfoCache>>::iterator it;
        for (it = this->commMap_.begin(); it != this->commMap_.end(); it++) {
            if (AreVectorsEqual(it->second->rankIds_, rankIds) && \
                it->second->backend_ == backend && it->second->bufferSize_ == bufferSize) {
            
                return it->first;
            }
        }
    }

    std::shared_ptr<CommInfoCache> commInfoCache = std::make_shared<CommInfoCache>();
    commInfoCache->cacheId_ = this->commMap_.size();
    commInfoCache->subCommRankId_ = subCommRankId;
    commInfoCache->rankIds_ = rankIds;
    commInfoCache->backend_ = backend;
    commInfoCache->bufferSize_ = bufferSize;
    std::string commDomain = "";
    if (backend == "lccl" && rankIds.size() > 1) {
        commDomain = GetLcclCommDomain(commInfoCache, groupId);
    } else if (backend == "hccl" && rankIds.size() > 1) {
        commDomain = GetHcclCommDomain(commInfoCache);
    }
    this->commMap_[commDomain] = commInfoCache;
    return commDomain;
}

std::string ExternalCommManager::GetLcclCommDomain(std::shared_ptr<CommInfoCache> &commInfoCache, uint32_t groupId)
{
    uint32_t commDomainInt = this->lcclCommDomainLowerBound_ + this->commDomainCounter_ + groupId;
    if (commDomainInt >= this->lcclCommDomainUpperBound_) {
        std::stringstream ss;
        ss << "External Comm Manager: Lccl commDomain exceeds the upper bound. "
            << "Available commDomain range is [" << this->lcclCommDomainLowerBound_
            << ", " << this->lcclCommDomainUpperBound_ << "]. "
            << "The range of the communication domain is determinded by `num_lccl_comm_shards` "
            << "and `lccl_comm_shard_id`. Please review initializaion parameters "
            << "of the `GeneratorTorch` object.";
        throw std::runtime_error(ss.str());
    }
    std::string commDomain = std::to_string(commDomainInt);
    this->commDomainCounter_ = this->commDomainCounter_ + ceil(this->worldSize_ / commInfoCache->rankIds_.size());
    return commDomain;
}

std::string ExternalCommManager::GetHcclCommDomain(std::shared_ptr<CommInfoCache> &commInfoCache)
{
    HcclComm hcclComm;
    HcclCommConfig config;
    HcclCommConfigInit(&config);
    config.hcclBufferSize = commInfoCache->bufferSize_;
    std::vector<uint32_t> tempRankIds = {};
    for (auto item : commInfoCache->rankIds_) { tempRankIds.push_back(item); }
    auto ret = HcclCreateSubCommConfig(&this->globalComm_, tempRankIds.size(), tempRankIds.data(),
        commInfoCache->cacheId_, commInfoCache->subCommRankId_, &config, &hcclComm);
    if (hcclComm == nullptr) {
        // ATB_SPEED_LOG_ERROR("External Comm Manager: Call `HcclCreateSubCommConfig` API from CANN "
        //     << "to create the hccl communication group failed. "
        //     << "Error code: " << ret << ". "
        //     << "Check the default log path at $HOME/ascecnd/log for more details. ");
    }
    commInfoCache->hcclComm_ = hcclComm;
    char hcclCommName[128] = {};
    HcclGetCommName(hcclComm, hcclCommName);
    std::string commDomain = std::string(hcclCommName);
    return commDomain;
}

HcclComm ExternalCommManager::GetCommPtr(std::string commDomain)
{
    auto it = this->commMap_.find(commDomain);
    if (it == this->commMap_.end()) {
        std::stringstream ss;
        ss << "External Comm Manager: Comm domain[" << commDomain << "] not found in cache.";
        throw std::out_of_range(ss.str());
    }
    return it->second->hcclComm_;
}

std::string ExternalCommManager::PrintCommInfo()
{
    std::stringstream ss;
    ss << "External Comm Manager: Comm Info Cache Summary: Count " << this->commMap_.size();
    std::map<std::string, std::shared_ptr<CommInfoCache>>::const_iterator it;
    for (it = this->commMap_.begin(); it != this->commMap_.end(); it++) {
        ss << " Comm domain[" << it->first << "] " << it->second->ToString();
    }
    return ss.str();
}

}  // namespace atb_speed