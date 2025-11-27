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
#include <folly/init/Init.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <pybind11/embed.h>
#include <torch/torch.h>

#include <csignal>
#include <filesystem>
#include <memory>

#include "api_service/api_service.h"
#include "core/common/global_flags.h"
#include "core/common/instance_name.h"
#include "core/common/metrics.h"
#include "core/common/options.h"
#include "core/common/types.h"
#include "core/runtime/master.h"
#include "core/runtime/master_coordinator.h"
#include "core/util/json_reader.h"
#include "core/util/net.h"
#include "core/util/utils.h"
#include "models/model_registry.h"
#include "server/xllm_server_registry.h"

using namespace xllm;

static std::atomic<uint32_t> signal_received{0};
void shutdown_handler(int signal) {
  // TODO: gracefully shutdown the server
  LOG(WARNING) << "Received signal " << signal << ", stopping server...";
  exit(1);
}

std::string get_model_backend(const std::filesystem::path& model_path) {
  JsonReader reader;
  // for llm, vlm and rec models, the config.json file is in the model path
  std::filesystem::path config_json_path = model_path / "config.json";
  // for dit models, the model_index.json file is in the model path
  std::filesystem::path model_index_json_path = model_path / "model_index.json";

  if (std::filesystem::exists(model_index_json_path)) {
    reader.parse(model_index_json_path);

    if (reader.value<std::string>("_diffusers_version").has_value()) {
      return "dit";
    } else {
      LOG(FATAL) << "Please check model_index.json file in model path: "
                 << model_path << ", it should contain _diffusers_version key.";
    }
  } else if (std::filesystem::exists(config_json_path)) {
    reader.parse(config_json_path);
    std::string model_type = reader.value<std::string>("model_type").value();
    if (model_type.empty()) {
      LOG(FATAL) << "Please check config.json file in model path: "
                 << model_path << ", it should contain model_type key.";
    }
    return ModelRegistry::get_model_backend(model_type);
  } else {
    LOG(FATAL) << "Please check config.json or model_index.json file, one of "
                  "them should exist in the model path: "
               << model_path;
  }
}

std::vector<std::string> parse_multi_model_path(const std::string& model_path) {
  std::vector<std::string> pathes = absl::StrSplit(model_path, ",");
  for (auto ith_path : pathes) {
    LOG(INFO) << "path : " << ith_path;
  }
  return pathes;
}

// TODO: refactor code
int run() {
  if (FLAGS_host.empty()) {
    // set the host to the local IP when the host is empty
    FLAGS_host = net::get_local_ip_addr();
  }

  auto master_ip = net::extract_ip(FLAGS_master_node_addr);
  auto master_port = net::extract_port(FLAGS_master_node_addr);
  bool is_local = false;
  if (FLAGS_host != "" && master_ip == FLAGS_host) {
    is_local = true;
  } else {
    is_local = false;
  }

  LOG(INFO) << "set worker role to "
            << (is_local ? "local worker" : "remote worker");

  if (FLAGS_backend == "vlm") {
    FLAGS_enable_prefix_cache = false;
    FLAGS_enable_chunked_prefill = false;
  }

  // if max_tokens_per_chunk_for_prefill is not set, set its value to
  // max_tokens_per_batch
  if (FLAGS_max_tokens_per_chunk_for_prefill < 0) {
    FLAGS_max_tokens_per_chunk_for_prefill = FLAGS_max_tokens_per_batch;
  }

  // TODO: handle multi-models case
  InstanceName::name()->set_name(FLAGS_host + ":" + std::to_string(FLAGS_port));

  // TODO: refactor
  std::vector<std::string> model_names;
  std::vector<std::string> model_versions;
  std::vector<std::string> models_path = parse_multi_model_path(FLAGS_model);
  std::vector<std::unique_ptr<LLMAssistantMaster>> assist_masters;
  std::vector<std::thread> threads;
  auto coordinator = std::make_shared<MasterCoordinator>(models_path.size());
  std::vector<std::unique_ptr<Master>> masters;
  for (auto i = 0; i < models_path.size(); ++i) {
    // TODO: FLAGS_backend support multi-models
    if (FLAGS_backend.empty()) {
      FLAGS_backend = get_model_backend(models_path[i]);
    }
    // check if model path exists
    if (!std::filesystem::exists(models_path[i])) {
      LOG(FATAL) << "Model path " << models_path[i] << " does not exist.";
    }

    std::filesystem::path model_path =
        std::filesystem::path(models_path[i]).lexically_normal();

    if (model_path.has_filename()) {
      model_names.emplace_back(
          std::filesystem::path(models_path[i]).filename());
      model_versions.emplace_back(
          std::filesystem::path(models_path[i]).filename());
    } else {
      model_names.emplace_back(
          std::filesystem::path(models_path[i]).parent_path().filename());
      model_versions.emplace_back(
          std::filesystem::path(models_path[i]).parent_path().filename());
    }

    int int_master_port = std::stoi(master_port);
    // Create Master
    Options options;
    options.model_path(models_path[i])
        .model_id(model_names[i])
        .task_type(FLAGS_task)
        .devices(FLAGS_devices)
        .draft_model_path(FLAGS_draft_model)
        .draft_devices(FLAGS_draft_devices)
        .backend(FLAGS_backend)
        .limit_image_per_prompt(FLAGS_limit_image_per_prompt)
        .block_size(FLAGS_block_size)
        .max_cache_size(FLAGS_max_cache_size)
        .max_memory_utilization(FLAGS_max_memory_utilization)
        .enable_prefix_cache(FLAGS_enable_prefix_cache)
        .max_tokens_per_batch(FLAGS_max_tokens_per_batch)
        .max_seqs_per_batch(FLAGS_max_seqs_per_batch)
        .max_tokens_per_chunk_for_prefill(
            FLAGS_max_tokens_per_chunk_for_prefill)
        .num_speculative_tokens(FLAGS_num_speculative_tokens)
        .num_request_handling_threads(FLAGS_num_request_handling_threads)
        .communication_backend(FLAGS_communication_backend)
        .enable_eplb(FLAGS_enable_eplb)
        .redundant_experts_num(FLAGS_redundant_experts_num)
        .eplb_update_interval(FLAGS_eplb_update_interval)
        .eplb_update_threshold(FLAGS_eplb_update_threshold)
        .rank_tablefile(FLAGS_rank_tablefile)
        .expert_parallel_degree(FLAGS_expert_parallel_degree)
        .enable_mla(FLAGS_enable_mla)
        .enable_chunked_prefill(FLAGS_enable_chunked_prefill)
        .master_node_addr(FLAGS_master_node_addr)
        .instance_role(InstanceRole(FLAGS_instance_role))
        .device_ip("")
        .transfer_listen_port(FLAGS_transfer_listen_port)
        .nnodes(FLAGS_nnodes)
        .node_rank(FLAGS_node_rank)
        .dp_size(FLAGS_dp_size)
        .ep_size(FLAGS_ep_size)
        .xservice_addr(FLAGS_xservice_addr)
        .instance_name(FLAGS_host + ":" + std::to_string(FLAGS_port))
        .enable_disagg_pd(FLAGS_enable_disagg_pd)
        .enable_pd_ooc(FLAGS_enable_pd_ooc)
        .enable_schedule_overlap(FLAGS_enable_schedule_overlap)
        .kv_cache_transfer_mode(FLAGS_kv_cache_transfer_mode)
        .etcd_addr(FLAGS_etcd_addr)
        .enable_service_routing(FLAGS_enable_service_routing)
        .tool_call_parser(FLAGS_tool_call_parser)
        .reasoning_parser(FLAGS_reasoning_parser)
        .priority_strategy(FLAGS_priority_strategy)
        .enable_online_preempt_offline(FLAGS_enable_online_preempt_offline)
        .enable_cache_upload(FLAGS_enable_prefix_cache &&
                             FLAGS_enable_service_routing &&
                             FLAGS_enable_cache_upload)
        .host_blocks_factor(FLAGS_host_blocks_factor)
        .enable_kvcache_store(FLAGS_enable_kvcache_store &&
                              FLAGS_enable_prefix_cache &&
                              (FLAGS_host_blocks_factor > 0.0))
        .store_protocol(FLAGS_store_protocol)
        .store_master_server_address(FLAGS_store_master_server_address)
        .store_metadata_server(FLAGS_store_metadata_server)
        .store_local_hostname(FLAGS_store_local_hostname)
        .enable_multi_stream_parallel(FLAGS_enable_multi_stream_parallel)
        .enable_profile_step_time(FLAGS_enable_profile_step_time)
        .enable_profile_token_budget(FLAGS_enable_profile_token_budget)
        .enable_latency_aware_schedule(FLAGS_enable_latency_aware_schedule)
        .profile_max_prompt_length(FLAGS_profile_max_prompt_length)
        .enable_profile_kv_blocks(FLAGS_enable_profile_kv_blocks)
        .disable_ttft_profiling(FLAGS_disable_ttft_profiling)
        .enable_forward_interruption(FLAGS_enable_forward_interruption)
        .max_global_ttft_ms(FLAGS_max_global_ttft_ms)
        .max_global_tpot_ms(FLAGS_max_global_tpot_ms)
        .max_requests_per_batch(FLAGS_max_requests_per_batch)
        .enable_continuous_kvcache(FLAGS_enable_continuous_kvcache)
        .enable_shm(FLAGS_enable_shm)
        .is_local(is_local)
        .serve_model_num(models_path.size())
        .current_model_idx(i)
        .master_node_addr(master_ip + ":" +
                          std::to_string(int_master_port + i));

    // working node
    if (options.node_rank() != 0) {
      // Create master for each model service.
      assist_masters.emplace_back(
          std::make_unique<LLMAssistantMaster>(options));
      // run master in a thread.
      threads.emplace_back(
          [master = assist_masters.back().get()]() { master->run(); });
    } else {
      // master node
      masters.emplace_back(create_master(FLAGS_backend, options, coordinator));
    }
  }

  // working node
  if (FLAGS_node_rank != 0) {
    // wait here
    for (auto& thread : threads) {
      if (thread.joinable()) {
        thread.join();
      }
    }
    return 0;
  }

  // master node
  // init all master
  for (auto& master : masters) {
    master->init();
  }
  // run master
  for (auto& master : masters) {
    master->run();
  }

  auto api_service =
      std::make_unique<APIService>(masters, model_names, model_versions);
  auto xllm_server =
      ServerRegistry::get_instance().register_server("HttpServer");

  // start brpc server
  if (!xllm_server->start(std::move(api_service))) {
    LOG(ERROR) << "Failed to start brpc server on port " << FLAGS_port;
    return -1;
  }

  return 0;
}

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = true;
  FLAGS_minloglevel = 0;
  google::ParseCommandLineFlags(&argc, &argv, true);

  google::InitGoogleLogging("xllm");

  return run();
}
