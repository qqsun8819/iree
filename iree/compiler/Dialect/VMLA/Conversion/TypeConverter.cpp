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

#include "iree/compiler/Dialect/VMLA/Conversion/TypeConverter.h"

#include "iree/compiler/Dialect/IREE/IR/IREETypes.h"
#include "iree/compiler/Dialect/VMLA/IR/VMLATypes.h"
#include "mlir/IR/StandardTypes.h"

namespace mlir {
namespace iree_compiler {

VMLATypeConverter::VMLATypeConverter() {
  addConversion([](Type type) -> Type {
    if (type.isInteger(1)) {
      // Widen i1 to i8.
      return IntegerType::get(8, type.getContext());
    }
    return type;
  });

  addConversion([](TensorType type) {
    // TODO(benvanik): composite-type conversion (buffer + dynamic dims).
    return IREE::VMLA::BufferType::get(type.getContext());
  });
}

}  // namespace iree_compiler
}  // namespace mlir
