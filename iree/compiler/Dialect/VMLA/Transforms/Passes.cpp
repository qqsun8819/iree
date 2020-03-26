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

#include "iree/compiler/Dialect/VMLA/Transforms/Passes.h"

#include <memory>

#include "iree/compiler/Dialect/IREE/Transforms/Passes.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"
#include "tensorflow/compiler/mlir/xla/transforms/passes.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace VMLA {

void buildVMLATransformPassPipeline(OpPassManager &passManager) {
  passManager.addPass(createCanonicalizerPass());

  // Flatten structured control flow to our CFG.
  passManager.addNestedPass<FuncOp>(xla_hlo::createLegalizeControlFlowPass());

  // Perform inlining and cleanup after CFG manipulation.
  passManager.addPass(createInlinerPass());
  passManager.addPass(createSymbolDCEPass());
  passManager.addNestedPass<FuncOp>(createCanonicalizerPass());
  passManager.addNestedPass<FuncOp>(createCSEPass());

  // Legalize input types.
  // TODO(benvanik): legalize input.
  // passManager.addPass(IREE::VMLA::createLegalizeInputTypesPass());

  // TODO(benvanik): preserve these hints during conversion.
  passManager.addNestedPass<FuncOp>(createDropCompilerHintsPass());

  // Unroll multi-dimensional reductions to one reduction per dimension.
  passManager.addNestedPass<FuncOp>(createUnrollReductionsPass());
  passManager.addNestedPass<FuncOp>(createCSEPass());

  // Convert from the various input dialects to the VMLA dialect.
  passManager.addPass(createConversionPass());

  // Cleanup identity ops that clutter up the IR and canonicalize.
  passManager.addNestedPass<FuncOp>(createCSEPass());
  passManager.addNestedPass<FuncOp>(createCanonicalizerPass());

  // TODO(benvanik): run symbol DCE pass.
}

static PassPipelineRegistration<> transformPassPipeline(
    "iree-vmla-transformation-pipeline",
    "Runs the full IREE VMLA dialect transformation pipeline",
    [](OpPassManager &passManager) {
      buildVMLATransformPassPipeline(passManager);
    });

}  // namespace VMLA
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
