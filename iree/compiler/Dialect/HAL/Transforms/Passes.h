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

#ifndef IREE_COMPILER_DIALECT_HAL_TRANSFORMS_PASSES_H_
#define IREE_COMPILER_DIALECT_HAL_TRANSFORMS_PASSES_H_

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Target/ExecutableTarget.h"
#include "llvm/ADT/StringMap.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

// Adds a set of passes to the given pass manager that run the required HAL
// transforms in the canonical order.
//
// Most translation code should prefer to use this instead of manually adding
// the passes themselves to ensure that expected pass ordering is observed.
//
// The expected usage is:
//   <run conversion to flow/sequencer/etc>
//   buildHALTransformPassPipeline & run
//   <run conversion from HAL to vm/etc>
void buildHALTransformPassPipeline(OpPassManager &passManager,
                                   ExecutableTargetOptions executableOptions);

//===----------------------------------------------------------------------===//
// Executable translation and optimization
//===----------------------------------------------------------------------===//

// Defines hal.executables and hal.interfaces for flow.executable ops based on
// usage within the module.
std::unique_ptr<OpPassBase<ModuleOp>> createMaterializeInterfacesPass(
    ExecutableTargetOptions executableOptions);

// Translates flow.executable ops to hal.executable ops using the provided
// options.
std::unique_ptr<OpPassBase<ModuleOp>> createTranslateExecutablesPass(
    ExecutableTargetOptions executableOptions);

// Rewrites hal.interface IO shims to look like the legacy IREE
// load_input/store_output form. This is incompatible with dynamic shapes and
// advanced descriptor set usage and will be removed as soon as the existing
// backends using it are ported to hal.interface.
std::unique_ptr<OpPassBase<IREE::Flow::ExecutableOp>>
createRewriteLegacyIOPass();

// For functions that contain reflection metadata in an
// iree.generateabi.reflection attribute, generate public ABI functions for
// typical clients to use.
std::unique_ptr<OpPassBase<ModuleOp>> createPublicABIGenerationPass();

//===----------------------------------------------------------------------===//
// Resource initialization, caching, and optimization
//===----------------------------------------------------------------------===//

// Finds all resource lookups (such as hal.executable.lookup), materializes
// their cache storage and initialization, and rewrites the lookups to
// references.
std::unique_ptr<OpPassBase<ModuleOp>> createMaterializeResourceCachesPass(
    ExecutableTargetOptions executableOptions);

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_HAL_TRANSFORMS_PASSES_H_
