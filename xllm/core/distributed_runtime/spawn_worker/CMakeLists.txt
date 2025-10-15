include(cc_binary)

cc_binary(
  NAME 
    spawn_worker
  HDRS
  SRCS
    spawn_process_worker_server.cpp
  DEPS
    :distributed_runtime
    absl::strings
)
