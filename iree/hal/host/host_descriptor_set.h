// Copyright 2020 Google LLC
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

#ifndef IREE_HAL_HOST_HOST_DESCRIPTOR_SET_H_
#define IREE_HAL_HOST_HOST_DESCRIPTOR_SET_H_

#include "absl/container/inlined_vector.h"
#include "iree/hal/descriptor_set.h"
#include "iree/hal/descriptor_set_layout.h"

namespace iree {
namespace hal {

class HostDescriptorSet final : public DescriptorSet {
 public:
  HostDescriptorSet(DescriptorSetLayout* set_layout,
                    absl::Span<const DescriptorSet::Binding> bindings);
  ~HostDescriptorSet() override;

  absl::Span<const DescriptorSet::Binding> bindings() const {
    return absl::MakeConstSpan(bindings_);
  }

 private:
  absl::InlinedVector<DescriptorSet::Binding, 4> bindings_;
};

}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_HOST_HOST_DESCRIPTOR_SET_H_
