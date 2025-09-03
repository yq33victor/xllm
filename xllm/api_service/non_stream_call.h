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

#pragma once

#include <brpc/controller.h>
#include <butil/iobuf.h>
#include <glog/logging.h>
#include <json2pb/pb_to_json.h>

#include <atomic>
#include <memory>
#include <optional>
#include <string>

#include "call.h"
#include "core/common/types.h"

namespace xllm {

template <typename Request, typename Response>
class NonStreamCall : public Call {
 public:
  NonStreamCall(brpc::Controller* controller,
                ::google::protobuf::Closure* done,
                Request* request,
                Response* response)
      : Call(controller), done_(done), request_(request), response_(response) {
    controller_->http_response().SetHeader("Content-Type",
                                           "text/javascript; charset=utf-8");

    json_options_.bytes_to_base64 = false;
    json_options_.jsonify_empty_array = true;
    json_options_.always_print_primitive_fields = true;
  }

  ~NonStreamCall() override { done_->Run(); }

  // For non stream response
  bool write_and_finish(Response& response) {
    butil::IOBufAsZeroCopyOutputStream json_output(
        &controller_->response_attachment());
    std::string err_msg;
    if (!json2pb::ProtoMessageToJson(
            response, &json_output, json_options_, &err_msg)) {
      return finish_with_error(StatusCode::UNKNOWN, err_msg);
    }

    return true;
  }

  // For non stream response
  bool finish_with_error(const StatusCode& code,
                         const std::string& error_message) {
    controller_->SetFailed(error_message);
    return true;
  }

  bool is_disconnected() const override { return controller_->IsCanceled(); }

  const Request& request() const { return *request_; }
  Response& response() { return *response_; }
  ::google::protobuf::Closure* done() { return done_; }

 private:
  ::google::protobuf::Closure* done_; // not owned

  Request* request_; // not owned, will be deleted when brpc resp is deleted
  Response* response_; // not owned, will be deleted when brpc resp is deleted

  json2pb::Pb2JsonOptions json_options_;
};

};  // namespace xllm
