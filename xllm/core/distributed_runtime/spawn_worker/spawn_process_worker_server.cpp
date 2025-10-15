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

#include <absl/strings/str_split.h>
#include <acl/acl.h>
#include <signal.h>
#include <sys/prctl.h>

#include <cstdlib>

#include "core/distributed_runtime/worker_server.h"
#include "core/runtime/options.h"

static bool g_running = true;

void handle_signal(int signum) { g_running = false; }

// Worker argv from engine process:
// @master_node_addr
// @local_rank
// @global_rank
// @world_size
// @device_idx
// @num_decoding_tokens
// @block_size
int main(int argc, char* argv[]) {
  if (argc < 7) {
    LOG(ERROR)
        << "Spwan worker process receive wrong args. Need 7 args, receive "
        << argc;
    return 1;
  }

  // set PR_SET_PDEATHSIG flag that child should exit
  // when parent process exit
  if (prctl(PR_SET_PDEATHSIG, SIGHUP) == -1) {
    perror("prctl");
    return EXIT_FAILURE;
  }

  std::string master_node_addr = std::string(argv[1]);
  int local_rank = atoi(argv[2]);
  int global_rank = atoi(argv[3]);
  int world_size = atoi(argv[4]);
  int device_idx = atoi(argv[5]);
  int num_decoding_tokens = atoi(argv[6]);
  int block_size = atoi(argv[7]);

  LOG(INFO) << "Spwan worker: "
            << "master_node_addr = " << master_node_addr
            << ", local_rank = " << local_rank
            << ", world_size = " << world_size
            << ", device_idx = " << device_idx
            << ", num_decoding_tokens = " << num_decoding_tokens
            << ", block_size = " << block_size << "\n";

  // TODO: pass whole xllm::runtime::Options here from main process.
  xllm::runtime::Options runner_options;
  runner_options.block_size(block_size)
      .num_decoding_tokens(num_decoding_tokens)
      .enable_schedule_overlap(false)
      .enable_offline_inference(true)
      .master_node_addr(master_node_addr);
  FLAGS_enable_schedule_overlap = false;
  FLAGS_master_node_addr = master_node_addr;
  FLAGS_block_size = block_size;

  std::atomic<bool> done(false);
  // TODO: FIXME npu
  torch::Device device("npu:" + std::to_string(device_idx));
  torch_npu::init_npu(device);
  int ret = aclrtSetDevice(device_idx);
  if (ret != 0) {
    LOG(FATAL) << "ACL set device id: " << device_idx << " failed, ret:" << ret;
  }
#if defined(USE_NPU)
  FLAGS_enable_atb_comm_multiprocess = true;
#endif

  LOG(ERROR)
      << "===========================> spawn: before worker_server: device = "
      << device_idx << ", FLAGS_rank_tablefile = " << FLAGS_rank_tablefile
      << ", FLAGS_communication_backend = " << FLAGS_communication_backend
      << ", FLAGS_expert_parallel_degree = " << FLAGS_expert_parallel_degree
      << ", FLAGS_enable_eplb = " << FLAGS_enable_eplb
      << ", FLAGS_redundant_experts_num = " << FLAGS_redundant_experts_num
      << ", FLAGS_eplb_update_interval = " << FLAGS_eplb_update_interval
      << ", FLAGS_eplb_update_threshold = " << FLAGS_eplb_update_threshold;

  xllm::ParallelArgs parallel_args(global_rank, world_size, 1, nullptr, 1);
  xllm::WorkerServer worker_server(local_rank,
                                   master_node_addr,
                                   done,
                                   parallel_args,
                                   device,
                                   runner_options,
                                   xllm::WorkerType::LLM,
                                   false);

  signal(SIGINT, handle_signal);
  signal(SIGTERM, handle_signal);
  // main thread waiting here
  while (g_running) {
    sleep(5);
  }

  return 0;
}
