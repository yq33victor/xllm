#include <absl/strings/str_split.h>
#include <cstdlib>
#include <signal.h>
#include <sys/prctl.h>
#include "core/distributed_runtime/worker_server.h"
#include "core/runtime/options.h"

// Worker argv from engine process:
// @master_node_addr
// @local_rank
// @global_rank
// @world_size
// @device_idx
// @num_decoding_tokens
// @block_size
int main(int argc, char *argv[]) {
  if (argc < 7) {
    LOG(ERROR) << "Spwan worker process receive wrong args. Need 7 args, receive " << argc;
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
            //<< ", notify_file_name = " << notify_file_name
            << ", master_node_addr = " << master_node_addr
            << ", local_rank = " << local_rank
            << ", world_size = " << world_size
            << ", device_idx = " <<  device_idx
            << ", num_decoding_tokens = " << num_decoding_tokens
            << ", block_size = " << block_size << "\n";

  runtime::Options runner_options;
    runner_options.block_size(block_size)
        .num_decoding_tokens(num_decoding_tokens);
  std::atomic<bool> done(false);
  // TODO: FIXME npu
  torch::Device device("npu:"+std::to_string(device_idx));

  xllm::WorkerServer::create_server(runner_options,
                                    done,
                                    master_node_addr,
                                    device,
                                    world_size,
                                    global_rank,
                                    1,  // dp_size
                                    local_rank,
                                    1); // ep_size

  return 0;
}
