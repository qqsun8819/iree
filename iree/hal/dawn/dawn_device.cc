// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iree/hal/dawn/dawn_device.h"

#include "absl/memory/memory.h"
#include "iree/base/status.h"
#include "iree/base/tracing.h"
#include "iree/hal/command_queue.h"
#include "iree/hal/executable_cache.h"
#include "iree/hal/fence.h"

namespace iree {
namespace hal {
namespace dawn {

namespace {

// ExecutableCache implementation that compiles but does nothing.
// This will be replaced with something functional soon.
class NoopExecutableCache final : public ExecutableCache {
 public:
  explicit NoopExecutableCache() {}
  ~NoopExecutableCache() override = default;

  bool CanPrepareFormat(ExecutableFormat format) const override {
    return false;
  }

  StatusOr<ref_ptr<Executable>> PrepareExecutable(
      ExecutableLayout* executable_layout, ExecutableCachingModeBitfield mode,
      const ExecutableSpec& spec) override {
    return UnimplementedErrorBuilder(IREE_LOC) << "PrepareExecutable NYI";
  }
};

}  // namespace

DawnDevice::DawnDevice(const DeviceInfo& device_info,
                       ::wgpu::Device backend_device)
    : Device(device_info), backend_device_(backend_device) {
  IREE_TRACE_SCOPE0("DawnDevice::ctor");

  // TODO(scotttodd): construct command queues, perform other initialization

  // Log some basic device info.
  std::string backend_type_str;
  auto* adapter =
      reinterpret_cast<dawn_native::Adapter*>(device_info.device_id());
  switch (adapter->GetBackendType()) {
    case dawn_native::BackendType::D3D12:
      backend_type_str = "D3D12";
      break;
    case dawn_native::BackendType::Metal:
      backend_type_str = "Metal";
      break;
    case dawn_native::BackendType::Null:
      backend_type_str = "Null";
      break;
    case dawn_native::BackendType::OpenGL:
      backend_type_str = "OpenGL";
      break;
    case dawn_native::BackendType::Vulkan:
      backend_type_str = "Vulkan";
      break;
  }
  LOG(INFO) << "Created DawnDevice '" << device_info.name() << "' ("
            << backend_type_str << ")";
}

DawnDevice::~DawnDevice() = default;

StatusOr<ref_ptr<DescriptorSetLayout>> DawnDevice::CreateDescriptorSetLayout(
    DescriptorSetLayout::UsageType usage_type,
    absl::Span<const DescriptorSetLayout::Binding> bindings) {
  IREE_TRACE_SCOPE0("DawnDevice::CreateDescriptorSetLayout");
  return UnimplementedErrorBuilder(IREE_LOC) << "CreateDescriptorSetLayout NYI";
}

StatusOr<ref_ptr<DescriptorSet>> DawnDevice::CreateDescriptorSet(
    DescriptorSetLayout* set_layout,
    absl::Span<const DescriptorSet::Binding> bindings) {
  IREE_TRACE_SCOPE0("DawnDevice::CreateDescriptorSet");
  return UnimplementedErrorBuilder(IREE_LOC) << "CreateDescriptorSet NYI";
}

ref_ptr<ExecutableCache> DawnDevice::CreateExecutableCache() {
  IREE_TRACE_SCOPE0("DawnDevice::CreateExecutableCache");
  return make_ref<NoopExecutableCache>();
}

StatusOr<ref_ptr<ExecutableLayout>> DawnDevice::CreateExecutableLayout(
    absl::Span<DescriptorSetLayout* const> set_layouts, size_t push_constants) {
  IREE_TRACE_SCOPE0("DawnDevice::CreateExecutableLayout");
  return UnimplementedErrorBuilder(IREE_LOC) << "CreateExecutableLayout NYI";
}

StatusOr<ref_ptr<CommandBuffer>> DawnDevice::CreateCommandBuffer(
    CommandBufferModeBitfield mode,
    CommandCategoryBitfield command_categories) {
  return UnimplementedErrorBuilder(IREE_LOC) << "CreateCommandBuffer NYI";
}

StatusOr<ref_ptr<Event>> DawnDevice::CreateEvent() {
  return UnimplementedErrorBuilder(IREE_LOC) << "CreateEvent NYI";
}

StatusOr<ref_ptr<BinarySemaphore>> DawnDevice::CreateBinarySemaphore(
    bool initial_value) {
  IREE_TRACE_SCOPE0("DawnDevice::CreateBinarySemaphore");

  return UnimplementedErrorBuilder(IREE_LOC)
         << "Binary semaphores not yet implemented";
}

StatusOr<ref_ptr<TimelineSemaphore>> DawnDevice::CreateTimelineSemaphore(
    uint64_t initial_value) {
  IREE_TRACE_SCOPE0("DawnDevice::CreateTimelineSemaphore");

  return UnimplementedErrorBuilder(IREE_LOC)
         << "Timeline semaphores not yet implemented";
}

StatusOr<ref_ptr<Fence>> DawnDevice::CreateFence(uint64_t initial_value) {
  IREE_TRACE_SCOPE0("DawnDevice::CreateFence");

  return UnimplementedErrorBuilder(IREE_LOC) << "CreateFence NYI";
}

Status DawnDevice::WaitAllFences(absl::Span<const FenceValue> fences,
                                 absl::Time deadline) {
  IREE_TRACE_SCOPE0("DawnDevice::WaitAllFences");

  return UnimplementedErrorBuilder(IREE_LOC) << "WaitAllFences NYI";
}

StatusOr<int> DawnDevice::WaitAnyFence(absl::Span<const FenceValue> fences,
                                       absl::Time deadline) {
  IREE_TRACE_SCOPE0("DawnDevice::WaitAnyFence");

  return UnimplementedErrorBuilder(IREE_LOC) << "WaitAnyFence NYI";
}

Status DawnDevice::WaitIdle(absl::Time deadline) {
  return UnimplementedErrorBuilder(IREE_LOC) << "WaitIdle";
}

}  // namespace dawn
}  // namespace hal
}  // namespace iree
