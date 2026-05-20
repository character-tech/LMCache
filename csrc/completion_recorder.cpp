// SPDX-License-Identifier: Apache-2.0

#include "completion_recorder.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <utility>

CompletionRecorder& CompletionRecorder::instance() {
  static CompletionRecorder recorder;
  return recorder;
}

void CompletionRecorder::push(PendingCompletion* completion) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    buffer_.push_back(std::move(*completion));
  }
  delete completion;
}

std::vector<PendingCompletion> CompletionRecorder::drain() {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<PendingCompletion> result;
  result.swap(buffer_);
  return result;
}

static bool trace_enabled() {
  static const bool enabled = [] {
    const char* v = std::getenv("LMCACHE_DEBUG_IPC_LIFECYCLE");
    return v && v[0] == '1';
  }();
  return enabled;
}

static std::atomic<uint64_t> g_callback_counter{0};

static void
#ifndef USE_ROCM
    CUDART_CB
#endif
    completion_host_callback(void* data) {
  if (trace_enabled()) {
    auto id = g_callback_counter.fetch_add(1, std::memory_order_relaxed);
    auto now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                      std::chrono::steady_clock::now().time_since_epoch())
                      .count();
    // Driver thread; no GIL — printf is safe and async-signal-ish.
    std::fprintf(stderr,
                 "ipc_event host_cb id=%llu monotonic_ns=%lld\n",
                 static_cast<unsigned long long>(id),
                 static_cast<long long>(now_ns));
  }
  auto* completion = static_cast<PendingCompletion*>(data);
  CompletionRecorder::instance().push(completion);
}

void record_completion_on_stream(int64_t cuda_stream_ptr,
                                 const std::string& kind,
                                 std::vector<std::string> payload) {
  auto* completion = new PendingCompletion{kind, std::move(payload)};
  auto stream = reinterpret_cast<lmcache_completion_stream_t>(
      static_cast<uintptr_t>(cuda_stream_ptr));
  auto err = LMCACHE_COMPLETION_LAUNCH_HOST_FUNC(
      stream, completion_host_callback, completion);
  // On failure the callback will never run, so we own the allocation.
  if (err != 0) {
    delete completion;
  }
}

CompletionDrainResult drain_recorded_completions() {
  auto completions = CompletionRecorder::instance().drain();
  CompletionDrainResult result;
  result.reserve(completions.size());
  for (auto& c : completions) {
    result.emplace_back(std::move(c.kind), std::move(c.payload));
  }
  return result;
}
