/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "worker_server.h"

#include <brpc/channel.h>
#include <brpc/controller.h>
#include <folly/Unit.h>
#include <folly/futures/Future.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <sys/wait.h>
#include <torch/torch.h>
#include <unistd.h>

#include <fstream>
#include <memory>
#include <optional>
#include <utility>

#include "common/global_flags.h"
#include "common/metrics.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/mapping_npu.h"
#include "framework/model/model_input_params.h"
#include "framework/parallel_state.h"
#include "framework/state_dict/state_dict.h"
#include "runtime/worker.h"
#include "server/xllm_server_registry.h"
#include "util/net.h"
#include "util/threadpool.h"
#include "util/timer.h"
#if defined(USE_NPU)
#include "hccl/hccl.h"
#include "xllm_kernels/core/include/atb_speed/base/external_comm_manager.h"
#include "xllm_kernels/core/include/atb_speed/utils/singleton.h"
#include "xllm_kernels/models/base/param/mapping.h"
#endif

extern char** environ;

namespace xllm {

void WorkerServer::create_server(const runtime::Options& options,
                                 std::atomic<bool>& done,
                                 const std::string& master_node_addr,
                                 const torch::Device& d,
                                 int world_size,
                                 int global_rank,
                                 int32_t dp_size,
                                 int local_rank,
                                 int32_t ep_size,
                                 int32_t notify_fd) {
  const char* val = std::getenv("ASCEND_RT_VISIBLE_DEVICES");
  if (val == nullptr) {
    LOG(FATAL)
        << "========================> ASCEND_RT_VISIBLE_DEVICES is nullptr";
  }
  std::string strVal(val);
  LOG(ERROR) << "=========================> ASCEND_RT_VISIBLE_DEVICES is "
             << strVal << ", d.index = " << d.index()
             << ", global_rank = " << global_rank;

  //  int ret2 = aclrtSetDevice(global_rank);
  //  if (ret2 != 0) {
  //    LOG(FATAL) << "ACL set device id: " << d.index()
  //               << " failed, ret:" << ret2;
  //  }

  Device device(d);
  // torch_npu::init_npu(device);
  device.set_device();
  //  device.init_device_context();

  LOG(ERROR) << "==========================> WorkerServer::create_server: "
                "device.index = "
             << device.index();
  //  int ret = aclrtSetDevice(device.index());
  //  if (ret != 0) {
  //    LOG(FATAL) << "ACL set device id: " << device.index()
  //               << " failed, ret:" << ret;
  //  }

  auto worker_global_rank = global_rank;
  // TODO: FIXME Later
  // std::unique_ptr<WorkerImpl> worker_impl =
  // std::make_unique<LLMWorkerImpl>(...);
  // std::unique_ptr<WorkerServiceImpl> worker_service_impl =
  //    std::make_unique<WorkerServiceImpl>(worker_impl);
  auto worker_service = std::make_shared<WorkerService>(options, device);

  auto addr = net::get_local_ip_addr();
  auto worker_server =
      ServerRegistry::get_instance().register_server("DistributeWorkerServer");
  if (!worker_server->start(worker_service, addr + ":0")) {
    LOG(ERROR) << "failed to start distribute worker server on address: "
               << addr;
    return;
  }

  auto worker_server_addr =
      addr + ":" + std::to_string(worker_server->listen_port());
  LOG(INFO) << "Worker " << worker_global_rank
            << ": server address: " << worker_server_addr;

  // Sync with master node
  proto::AddressInfo addr_info;
  addr_info.set_address(worker_server_addr);
  addr_info.set_global_rank(worker_global_rank);
  proto::CommUniqueIdList uids;
  sync_master_node(master_node_addr, addr_info, uids);

  LOG(ERROR) << "=========================> create worker_server: "
                "worker_global_rank = "
             << worker_global_rank << ", world_size = " << world_size
             << ", dp_size = " << dp_size << ", ep_size = " << ep_size
             << ", options.task_type = " << options.task_type();
  CollectiveCommunicator comm(worker_global_rank, world_size, dp_size, ep_size);
  const ParallelArgs* parallel_args = comm.parallel_args();
#if defined(USE_MLU)
  comm.create_process_groups_cncl(master_node_addr, device);
#endif

  WorkerType worker_type =
      (options.task_type() == "generate") ? WorkerType::LLM : WorkerType::ELM;
  CHECK(worker_type == WorkerType::LLM || worker_type == WorkerType::ELM)
      << "Multi Node only support LLM and ELM Now, but get task type = "
      << options.task_type();
  std::unique_ptr<Worker> worker =
      std::make_unique<Worker>(*parallel_args, device, options, worker_type);
  worker_service->set_worker(std::move(worker));
  done.store(true);

  if (notify_fd != -1) {
    LOG(ERROR)
        << "============================> Child worker process write pipe";
    const char msg = '1';
    if (write(notify_fd, &msg, 1) == -1) {
      LOG(FATAL) << "Child worker process write pipe failed.";
      return;
    }
    close(notify_fd);
  }

  // Wait until Ctrl-C is pressed, then Stop() and Join() the server.
  worker_server->run();
}

void WorkerServer::create_fork_server(const runtime::Options& options,
                                      std::atomic<bool>& done,
                                      const std::string& master_node_addr,
                                      const torch::Device& d,
                                      int world_size,
                                      int global_rank,
                                      int32_t dp_size,
                                      int local_rank,
                                      int32_t ep_size) {
  pid_t pid = fork();
  if (pid == -1) {
    LOG(FATAL) << "create fork server failed.";
    return;
  }

  int pipefd[2];
  if (pipe(pipefd) == -1) {
    LOG(FATAL) << "Create pipe failed.";
    return;
  }

  if (pid == 0) {
    close(pipefd[0]);
    create_server(options,
                  done,
                  master_node_addr,
                  d,
                  world_size,
                  global_rank,
                  dp_size,
                  local_rank,
                  ep_size,
                  pipefd[1]);
  } else {
    close(pipefd[1]);
    char buf;
    ssize_t n = read(pipefd[0], &buf, 1);
    LOG(ERROR)
        << "============================> Parent worker process read pipe";
    close(pipefd[0]);
    done.store(true);
  }
}

void WorkerServer::create_spawn_server(int local_rank,
                                       const std::string& master_node_addr,
                                       std::atomic<bool>& done,
                                       const ParallelArgs& parallel_args,
                                       const torch::Device& d,
                                       const runtime::Options& options) {
  auto local_rank_str0 = std::to_string(local_rank);
  const char* local_rank_str = local_rank_str0.c_str();
  auto global_rank_str0 = std::to_string(parallel_args.rank());
  const char* global_rank_str = global_rank_str0.c_str();
  auto world_size_str0 = std::to_string(parallel_args.world_size());
  const char* world_size_str = world_size_str0.c_str();
  auto device_idx_str0 = std::to_string(d.index());
  LOG(ERROR) << "===========================> "
                "WorkerServer::create_spawn_server: d.index = "
             << device_idx_str0;
  const char* device_idx_str = device_idx_str0.c_str();
  auto num_decoding_tokens_str0 = std::to_string(options.num_decoding_tokens());
  const char* num_decoding_tokens_str = num_decoding_tokens_str0.c_str();
  auto block_size_str0 = std::to_string(options.block_size());
  const char* block_size_str = block_size_str0.c_str();
  std::string spawn_worker_bin_path =
      options.spawn_worker_path() + "/spawn_worker";
  LOG(INFO) << "Spawn worker path: " << spawn_worker_bin_path;
  const char* argv[] = {spawn_worker_bin_path.c_str(),
                        master_node_addr.c_str(),
                        local_rank_str,
                        global_rank_str,
                        world_size_str,
                        device_idx_str,
                        num_decoding_tokens_str,
                        block_size_str,
                        nullptr};
  pid_t pid;
  posix_spawn_file_actions_init(&file_actions_);
  posix_spawnattr_init(&spawn_attr_);
  int status = posix_spawnp(&pid,
                            argv[0],
                            &file_actions_,
                            &spawn_attr_,
                            const_cast<char**>(argv),
                            environ);
  if (status != 0) {
    LOG(ERROR) << "posix_spawnp failed: " << strerror(status);
    return;
  }
  use_spwan_worker_ = true;
  done.store(true);
}

WorkerServer::WorkerServer(int local_worker_idx,
                           const std::string& master_node_addr,
                           std::atomic<bool>& done,
                           const ParallelArgs& parallel_args,
                           const torch::Device& d,
                           const runtime::Options& options,
                           WorkerType worker_type,
                           bool use_spawn_worker) {
  if (worker_type == WorkerType::LLM || worker_type == WorkerType::ELM) {
    if (use_spawn_worker) {
      if (options.max_seqs_per_batch() % 2 != 0) {
        LOG(ERROR) << "========================> worker-server: "
                      "create spawn worker.  options.max_seqs_per_batch = "
                   << options.max_seqs_per_batch();
        // start worker in a spawn process
        create_spawn_server(local_worker_idx,
                            master_node_addr,
                            done,
                            parallel_args,
                            d,
                            options);
      } else {
        LOG(ERROR) << "========================> worker-server: "
                      "create fork worker.  options.max_seqs_per_batch = "
                   << options.max_seqs_per_batch();
        fork_worker_thread_.emplace_back(std::move(
            std::make_unique<std::thread>(&WorkerServer::create_fork_server,
                                          this,
                                          std::cref(options),
                                          std::ref(done),
                                          std::cref(master_node_addr),
                                          std::cref(d),
                                          parallel_args.world_size(),
                                          parallel_args.rank(),
                                          parallel_args.dp_size(),
                                          local_worker_idx,
                                          parallel_args.ep_size())));
      }
      //      sleep(5);
    } else {
      LOG(ERROR) << "========================> worker-server: worker_thread_";
      // start worker in a thread.
      worker_thread_ =
          std::make_unique<std::thread>(&WorkerServer::create_server,
                                        this,
                                        std::cref(options),
                                        std::ref(done),
                                        std::cref(master_node_addr),
                                        std::cref(d),
                                        parallel_args.world_size(),
                                        parallel_args.rank(),
                                        parallel_args.dp_size(),
                                        local_worker_idx,
                                        parallel_args.ep_size(),
                                        -1);
    }
  } else {
    // TODO: support other model type later.
    LOG(ERROR) << "Unsupported model type: " << worker_type;
  }
}

bool WorkerServer::sync_master_node(const std::string& master_node_addr,
                                    proto::AddressInfo& addr_info,
                                    proto::CommUniqueIdList& uids) {
  // Brpc connection resources
  brpc::Channel channel;
  brpc::ChannelOptions options;
  options.connection_type = "single";
  options.timeout_ms = 10000;
  options.max_retry = 3;
  if (channel.Init(master_node_addr.c_str(), "", &options) != 0) {
    LOG(ERROR) << "Failed to initialize BRPC channel to " << master_node_addr;
    return false;
  }
  proto::Collective_Stub stub(&channel);

  // Retry until master node ready
  int try_count = 0;
  brpc::Controller cntl;
  while (try_count < FLAGS_max_connect_count) {
    cntl.Reset();
    stub.Sync(&cntl, &addr_info, &uids, NULL);
    if (cntl.Failed()) {
      LOG(WARNING) << "Worker#" << addr_info.global_rank()
                   << " try connect to engine server error, try again."
                   << " Error message: " << cntl.ErrorText();
      std::this_thread::sleep_for(
          std::chrono::seconds(FLAGS_sleep_time_second));
    } else {
      LOG(INFO) << "Worker#" << addr_info.global_rank() << " connect to "
                << master_node_addr << " success.";
      break;
    }
    try_count++;
  }

  if (try_count >= FLAGS_max_connect_count) {
    LOG(ERROR) << "Worker#" << addr_info.global_rank() << " connect to "
               << master_node_addr << " failed."
               << " Error message: " << cntl.ErrorText();
    return false;
  }

  return true;
}

WorkerServer::~WorkerServer() {
  if (worker_thread_->joinable()) {
    worker_thread_->join();
  }

  if (use_spwan_worker_) {
    posix_spawn_file_actions_destroy(&file_actions_);
    posix_spawnattr_destroy(&spawn_attr_);
  }
}

}  // namespace xllm
