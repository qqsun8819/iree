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

//===- Utils.cpp - GPU code-generation utility methods --------------------===//
//
// Implementation of utility methods that are used in passes that lower to
// SPIR-V.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Translation/SPIRV/LinalgToSPIRV/Utils.h"

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"

namespace mlir {
namespace iree_compiler {

/// Gets the number of outer parallel loops in a linalg operation.
unsigned getNumOuterParallelLoops(linalg::LinalgOp linalgOp) {
  if (auto convOp = dyn_cast<linalg::ConvOp>(linalgOp.getOperation())) return 0;
  // Find the number of leading parallel loops in the generic op
  unsigned numOuterParallelLoops = 0;
  for (auto iteratorType : linalgOp.iterator_types()) {
    if (iteratorType.cast<StringAttr>().getValue() !=
        getParallelIteratorTypeName()) {
      break;
    }
    numOuterParallelLoops++;
  }
  return numOuterParallelLoops;
}

}  // namespace iree_compiler
}  // namespace mlir
