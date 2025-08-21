#include "request.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <glog/logging.h>

#include <cstdint>
#include <string>
#include <vector>

#include "sequence.h"

namespace xllm {

Request::Request(const std::string& request_id,
                 const std::string& x_request_id,
                 const std::string& x_request_time,
                 const RequestState& state,
                 const std::string& service_request_id)
    : request_id_(request_id),
      service_request_id_(service_request_id),
      x_request_id_(x_request_id),
      x_request_time_(x_request_time),
      state_(std::move(state)),
      created_time_(absl::Now()) {
  create_sequences_group();
}

void Request::create_sequences_group() {
  SequenceParams sequence_params;
  sequence_params.seq_capacity = state_.seq_capacity;
  sequence_params.skip_special_tokens = state_.skip_special_tokens;
  sequence_params.echo = state_.echo;
  sequence_params.logprobs = state_.logprobs;
  sequence_params.n = state_.n;
  sequence_params.best_of = state_.best_of;
  sequence_params.streaming = state_.stream;
  sequence_params.enable_schedule_overlap = state_.enable_schedule_overlap;
  sequence_params.sampling_param = &(state_.sampling_param);
  sequence_params.stopping_checker = &(state_.stopping_checker);
  sequences_group_ = std::make_unique<SequencesGroup>(state_.prompt,
                                                      state_.prompt_tokens,
                                                      state_.input_embedding,
                                                      state_.mm_data,
                                                      sequence_params);
}

bool Request::finished() const { return sequences_group_->finished(); }

bool Request::expand_sequences(bool share_prefix) {
  return sequences_group_->expand_sequences(share_prefix);
}

void Request::log_statistic(double total_latency) {
  // log the request statistics
  int idx = 0;
  for (const auto& seq : sequences()) {
    LOG(INFO) << "x-request-id: " << x_request_id_ << ", "
              << "x-request-time: " << x_request_time_ << ", "
              << "request_id: " << request_id_ << ", "
              << "sequence " << idx++ << ", "
              << "max_tokens:       "
              << seq->stopping_checker()->get_max_generated_tokens() << ", "
              << "temperature: " << seq->sampling_param()->temperature << ", "
              << "finish_reason:     "
              << seq->finish_reason().to_string().value_or("") << ", "
              << "prompt_tokens: " << seq->num_prompt_tokens() << ", "
              << "generated_tokens: "
              << (state_.enable_schedule_overlap
                      ? seq->num_generated_tokens() - 1
                      : seq->num_generated_tokens())
              << ", " << std::fixed << std::setprecision(1)
              << "ttft: " << seq->time_to_first_token_latency_seconds()
              << "ms, "
              << "total_latency: " << total_latency * 1000 << "ms";
  }
}

void Request::log_error_statistic(Status status) {
  // log the request statistics
  int idx = 0;
  for (const auto& seq : sequences()) {
    LOG(INFO) << "x-request-id: " << x_request_id_ << ", "
              << "x-request-time: " << x_request_time_ << ", "
              << "request_id: " << request_id_ << ", "
              << "sequence " << idx++ << ", "
              << "max_tokens: "
              << seq->stopping_checker()->get_max_generated_tokens() << ", "
              << "temperature: " << seq->sampling_param()->temperature << ", "
              << "prompt_tokens: " << seq->num_prompt_tokens() << ", "
              << "status_code : " << static_cast<int32_t>(status.code()) << ", "
              << "status_msg : " << status.message();
  }
}

size_t Request::total_num_blocks() {
  size_t num = 0;
  for (auto& seq : sequences()) {
    num += seq->kv_state().num_kv_blocks();
  }
  return num;
}

RequestOutput Request::generate_output(const Tokenizer& tokenizer) {
  // summarize statistics for all sequences
  Usage usage;
  usage.num_prompt_tokens = state_.prompt_tokens.size();
  for (const auto& seq : sequences()) {
    usage.num_generated_tokens += seq->num_generated_tokens();
    // NOTE: Avoid counting the extra execution step in overlap scenario.
    if (state_.enable_schedule_overlap) {
      usage.num_generated_tokens--;
    }
  }
  usage.num_total_tokens = usage.num_prompt_tokens + usage.num_generated_tokens;

  RequestOutput output;
  output.request_id = request_id_;
       output.service_request_id = service_request_id_;
  output.usage = usage;
  output.status = Status(StatusCode::OK);

        output.finished = finished();
  output.cancelled = cancelled();
  sequences_group_->generate_outputs(output.outputs, tokenizer);

  return output;
}

}  // namespace xllm
