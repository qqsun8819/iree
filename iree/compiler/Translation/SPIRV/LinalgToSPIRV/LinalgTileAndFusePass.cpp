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

//===- LinalgTilingOnBuffers.cpp - Tile and fuse Linalg on Buffers --------===//
//
// Implements a pass to tile and fuse linalg operations on buffers.
//
//===----------------------------------------------------------------------===//
#include "iree/compiler/Translation/CodegenUtils/CodegenUtils.h"
#include "iree/compiler/Translation/SPIRV/LinalgToSPIRV/Utils.h"
#include "mlir/Dialect/Linalg/Transforms/LinalgTransforms.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/FoldUtils.h"

namespace mlir {
namespace iree_compiler {

static StringRef getWorkGroupMarker() { return "spirv_workgroup"; }

/// Computes the tile sizes to be used in linalg based on the workgroup size.
static LogicalResult getTileSizes(linalg::LinalgOp linalgOp,
                                  SmallVectorImpl<int64_t> &tileSizes) {
  FuncOp funcOp = linalgOp.getParentOfType<FuncOp>();
  SmallVector<int64_t, 3> workGroupSizeVec;
  workGroupSizeVec.reserve(3);
  if (failed(getWorkGroupSize(funcOp, workGroupSizeVec))) return failure();
  ArrayRef<int64_t> workGroupSize = dropTrailingOnes(workGroupSizeVec);
  auto rev = reverse(workGroupSize);

  unsigned numOuterParallelLoops = getNumOuterParallelLoops(linalgOp);
  // Tile sizes to use are reverse of the workGroupSize.
  tileSizes.assign(rev.begin(), rev.end());
  // Linalg convention is to use 0 for no tiling. If the workgroup size is
  // 1, then dont tile along that dimension. So overriding 1 to 0.
  for (auto &tileSize : tileSizes)
    if (tileSize == 1) tileSize = 0;
  tileSizes.resize(numOuterParallelLoops, 0);
  return success();
}

namespace {
/// Function pass that implements tiling and fusion in Linalg.
struct LinalgTileAndFusePass : public FunctionPass<LinalgTileAndFusePass> {
  void runOnFunction() override;
};

/// If the linalg op has no outer parallel loops, inserts dummy one-trip loops
/// around it to execute it sequentially within a thread.
template <typename LinalgOp>
struct ExecuteSequentiallyPattern : public OpRewritePattern<LinalgOp> {
  using OpRewritePattern<LinalgOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(LinalgOp op,
                                PatternRewriter &rewriter) const override {
    auto tilingAttr = op.template getAttrOfType<StringAttr>(
        linalg::LinalgTransforms::kLinalgTransformMarker);
    if (tilingAttr && tilingAttr.getValue() == getWorkGroupMarker())
      return failure();
    OpBuilder::InsertionGuard guard(rewriter);
    linalg::LinalgOp linalgOp = cast<linalg::LinalgOp>(op.getOperation());
    unsigned numParallelLoops = getNumOuterParallelLoops(linalgOp);
    if (!linalgOp.hasBufferSemantics() || numParallelLoops) return failure();

    auto indexType = rewriter.getIndexType();
    auto loc = linalgOp.getLoc();
    auto zero =
        rewriter.create<ConstantOp>(loc, rewriter.getIntegerAttr(indexType, 0));
    auto one =
        rewriter.create<ConstantOp>(loc, rewriter.getIntegerAttr(indexType, 1));
    auto outerLoop = rewriter.create<loop::ForOp>(loc, zero, one, one);
    rewriter.setInsertionPoint(outerLoop.getBody(),
                               std::prev(outerLoop.getBody()->end()));
    auto innerLoop = rewriter.create<loop::ForOp>(loc, zero, one, one);
    rewriter.setInsertionPoint(innerLoop.getBody(),
                               std::prev(innerLoop.getBody()->end()));
    Operation *clonedOp = rewriter.clone(*linalgOp.getOperation());
    clonedOp->setAttr(linalg::LinalgTransforms::kLinalgTransformMarker,
                      rewriter.getStringAttr(getWorkGroupMarker()));
    rewriter.eraseOp(linalgOp);
    return success();
  }
};

/// If there is nothing to fuse the linalg op with, then just tiles it.
template <typename LinalgOp>
struct TileLinalgOpPattern : public OpRewritePattern<LinalgOp> {
  using OpRewritePattern<LinalgOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(LinalgOp op,
                                PatternRewriter &rewriter) const override {
    auto tilingAttr = op.template getAttrOfType<StringAttr>(
        linalg::LinalgTransforms::kLinalgTransformMarker);
    if (tilingAttr && tilingAttr.getValue() == getWorkGroupMarker())
      return failure();
    // Check that all input and outputs have a single use (this op). If so just
    // tile it.
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op.getOperation());
    SmallVector<int64_t, 3> tileSizes;
    linalg::LinalgOp linalgOp = cast<linalg::LinalgOp>(op.getOperation());
    if (!linalgOp.hasBufferSemantics() ||
        !llvm::all_of(linalgOp.getInputsAndOutputBuffers(),
                      [](Value arg) { return arg.hasOneUse(); }) ||
        failed(getTileSizes(linalgOp, tileSizes)))
      return failure();
    if (failed(linalg::tileLinalgOpAndSetMarker(rewriter, op.getOperation(),
                                                tileSizes, getWorkGroupMarker(),
                                                {}))) {
      return failure();
    }
    rewriter.eraseOp(op);
    return success();
  }
};

/// Tile and fuse linalg operations.
template <typename LinalgOp>
struct TileAndFuseLinalgOpPattern : public OpRewritePattern<LinalgOp> {
  using OpRewritePattern<LinalgOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(LinalgOp op,
                                PatternRewriter &rewriter) const override {
    auto tilingAttr = op.template getAttrOfType<StringAttr>(
        linalg::LinalgTransforms::kLinalgTransformMarker);
    if (tilingAttr && tilingAttr.getValue() == getWorkGroupMarker())
      return failure();
    linalg::LinalgOp linalgOp = cast<linalg::LinalgOp>(op.getOperation());
    SmallVector<int64_t, 3> tileSizes;
    if (!linalgOp.hasBufferSemantics() ||
        failed(getTileSizes(linalgOp, tileSizes)))
      return failure();

    SmallVector<int64_t, 1> operandIndicesToFuse;
    for (auto buffer : llvm::enumerate(linalgOp.getInputsAndOutputBuffers())) {
      // If a buffer has multiple uses, then it is a candidate for fusion.
      if (!buffer.value().hasOneUse())
        operandIndicesToFuse.push_back(buffer.index());
    }
    if (operandIndicesToFuse.empty()) return failure();
    if (failed(linalg::tileAndFuseLinalgOpAndSetMarker(
            rewriter, linalgOp, tileSizes, operandIndicesToFuse,
            getWorkGroupMarker())))
      return failure();
    rewriter.eraseOp(op);
    return success();
  }
};
}  // namespace

void LinalgTileAndFusePass::runOnFunction() {
  OwningRewritePatternList patterns;
  patterns.insert<ExecuteSequentiallyPattern<linalg::ConvOp>,
                  ExecuteSequentiallyPattern<linalg::IndexedGenericOp>,
                  TileLinalgOpPattern<linalg::GenericOp>,
                  TileLinalgOpPattern<linalg::IndexedGenericOp>,
                  TileLinalgOpPattern<linalg::MatmulOp>,
                  TileAndFuseLinalgOpPattern<linalg::GenericOp>>(&getContext());
  applyPatternsGreedily(getOperation(), patterns);
}

std::unique_ptr<OpPassBase<FuncOp>> createLinalgTileAndFusePass() {
  return std::make_unique<LinalgTileAndFusePass>();
}

static PassRegistration<LinalgTileAndFusePass> pass(
    "iree-linalg-tile-and-fuse", "Tile and Fuse Linalg operations on buffers");
}  // namespace iree_compiler
}  // namespace mlir
