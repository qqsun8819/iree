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
#include "mlir/Dialect/Linalg/Transforms/LinalgTransforms.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/FoldUtils.h"

namespace mlir {
namespace iree_compiler {

static StringRef getWorkGroupMarker() { return "spirv_workgroup"; }

/// Tile sizes to use by default based on number of dimension of parallelism.
static void getDefaultTileSizes(unsigned numDims,
                                SmallVectorImpl<int64_t> &tileSizes) {
  tileSizes.clear();
  switch (numDims) {
    case 0:
      return;
    case 1:
      tileSizes.push_back(32);
      return;
    case 2:
      tileSizes.push_back(4);
      tileSizes.push_back(32);
      return;
    default:
      tileSizes.push_back(2);
      tileSizes.push_back(2);
      tileSizes.push_back(32);
      return;
  }
}

/// Returns the number of "outer" parallel loops specified in the `linalgOp`.
static unsigned getNumOuterParallelLoops(linalg::LinalgOp linalgOp) {
  // Find the number of leading parallel loops in the generic op.
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

/// Method to compute the tile size of the linalg operation.
template <typename LinalgOp>
static void getTileSizesImpl(LinalgOp op, SmallVectorImpl<int64_t> &tileSizes) {
  return getDefaultTileSizes(getNumOuterParallelLoops(op.getOperation()),
                             tileSizes);
}

/// Method to compute the tile size for convolution operations.
template <>
void getTileSizesImpl<linalg::ConvOp>(linalg::ConvOp op,
                                      SmallVectorImpl<int64_t> &tileSizes) {
  // Disable tiling on convolutions.
  if (op.padding())
    return getDefaultTileSizes(op.getNumBatchDimensions(), tileSizes);
  return getDefaultTileSizes(getNumOuterParallelLoops(op.getOperation()),
                             tileSizes);
}

/// Method to get the tile size for a linalg operation that can be overriden by
/// the workGroup size passed in.
template <typename LinalgOp>
static void getTileSizes(LinalgOp linalgOp, ArrayRef<int64_t> workGroupSize,
                         SmallVectorImpl<int64_t> &tileSizes) {
  assert(tileSizes.empty());
  if (!workGroupSize.empty()) {
    auto rev = reverse(workGroupSize.take_front(3));
    tileSizes.assign(rev.begin(), rev.end());
    unsigned numParallelLoops = getNumOuterParallelLoops(
        cast<linalg::LinalgOp>(linalgOp.getOperation()));
    tileSizes.resize(numParallelLoops, 0);
    return;
  } else {
    getTileSizesImpl(linalgOp, tileSizes);
  }
  // Linalg convention is to use 0 for no tiling. If the workgroup size is
  // 1, then dont tile along that dimension. So overriding 1 to 0.
  for (auto &tileSize : tileSizes)
    if (tileSize == 1) tileSize = 0;
}

/// Checks if an operation already has an attribute with this marker. If set it
/// implies this op shouldnt be tiled with the same marker.
static bool hasMarker(Operation *op, StringRef marker) {
  auto tilingAttr = op->getAttrOfType<StringAttr>(
      linalg::LinalgTransforms::kLinalgTransformMarker);
  return tilingAttr && tilingAttr.getValue() == marker;
}

namespace {
/// Function pass that implements tiling and fusion in Linalg on buffers.
struct LinalgTileAndFusePass : public FunctionPass<LinalgTileAndFusePass> {
  LinalgTileAndFusePass(ArrayRef<int64_t> workGroupSize = {})
      : workGroupSize(workGroupSize.begin(), workGroupSize.end()) {}
  void runOnFunction() override;

 private:
  SmallVector<int64_t, 3> workGroupSize;
};

/// Base class for Linalg tiling patterns. All classes that derive from this
/// need to implement an apply method that will tile the operation with the
/// following signature.
///
/// LogicalResult apply(LinalgOp op, SmallVectorImpl<int64_t> &tileSizes,
///                     PatternRewriter &rewriter) const
template <typename DerivedClass, typename LinalgOp>
struct LinalgTilingPattern : public OpRewritePattern<LinalgOp> {
  LinalgTilingPattern(MLIRContext *context, ArrayRef<int64_t> workGroupSize,
                      PatternBenefit benefit = 1)
      : OpRewritePattern<LinalgOp>(context, benefit),
        workGroupSize(workGroupSize) {}

  LogicalResult matchAndRewrite(LinalgOp linalgOp,
                                PatternRewriter &rewriter) const override {
    if (!linalgOp.hasBufferSemantics()) return failure();
    // Currently we are only doing one-level tiling, so a single marker is
    // enough. This might need to move into derived classes.
    if (hasMarker(linalgOp.getOperation(), getWorkGroupMarker()))
      return failure();

    SmallVector<int64_t, 3> tileSizes;
    getTileSizes(linalgOp, workGroupSize, tileSizes);
    if (failed(static_cast<const DerivedClass *>(this)->apply(
            linalgOp, tileSizes, rewriter)))
      return failure();
    assert(tileSizes.size() <= 3 && "illegal tile sizes greater than 3");
    SmallVector<int32_t, 3> updatedWorkGroupSize(tileSizes.rbegin(),
                                                 tileSizes.rend());
    if (failed(
            updateWorkGroupSize(linalgOp.getOperation(), updatedWorkGroupSize)))
      return failure();
    rewriter.eraseOp(linalgOp);
    return success();
  }

 private:
  ArrayRef<int64_t> workGroupSize;
};

/// If the linalg op has no outer parallel loops, inserts dummy one-trip loops
/// around it to execute it sequentially within a thread.
template <typename LinalgOp>
struct ExecuteSequentiallyPattern
    : public LinalgTilingPattern<ExecuteSequentiallyPattern<LinalgOp>,
                                 LinalgOp> {
  using LinalgTilingPattern<ExecuteSequentiallyPattern<LinalgOp>,
                            LinalgOp>::LinalgTilingPattern;
  LogicalResult apply(LinalgOp linalgOp, ArrayRef<int64_t> tileSizes,
                      PatternRewriter &rewriter) const {
    if (!tileSizes.empty()) return failure();

    OpBuilder::InsertionGuard guard(rewriter);
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
    return success();
  }
};

/// If there is nothing to fuse the linalg op with, then just tiles it.
template <typename LinalgOp>
struct TileLinalgOpPattern
    : public LinalgTilingPattern<TileLinalgOpPattern<LinalgOp>, LinalgOp> {
  using LinalgTilingPattern<TileLinalgOpPattern<LinalgOp>,
                            LinalgOp>::LinalgTilingPattern;
  LogicalResult apply(LinalgOp linalgOp, ArrayRef<int64_t> tileSizes,
                      PatternRewriter &rewriter) const {
    // Check that all input and outputs have a single use (this op). If so just
    // tile it.
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(linalgOp.getOperation());
    if (!llvm::all_of(linalgOp.getInputsAndOutputBuffers(),
                      [](Value arg) { return arg.hasOneUse(); }))
      return failure();
    return linalg::tileLinalgOpAndSetMarker(rewriter, linalgOp.getOperation(),
                                            tileSizes, getWorkGroupMarker(),
                                            /*permutation=*/{});
  }
};

/// Tile and fuse linalg operations.
template <typename LinalgOp>
struct TileAndFuseLinalgOpPattern
    : public LinalgTilingPattern<TileAndFuseLinalgOpPattern<LinalgOp>,
                                 LinalgOp> {
  using LinalgTilingPattern<TileAndFuseLinalgOpPattern<LinalgOp>,
                            LinalgOp>::LinalgTilingPattern;
  LogicalResult apply(LinalgOp linalgOp, ArrayRef<int64_t> tileSizes,
                      PatternRewriter &rewriter) const {
    SmallVector<int64_t, 1> operandIndicesToFuse;
    for (auto buffer : llvm::enumerate(linalgOp.getInputsAndOutputBuffers())) {
      // If a buffer has multiple uses, then it is a candidate for fusion.
      if (!buffer.value().hasOneUse())
        operandIndicesToFuse.push_back(buffer.index());
    }
    if (operandIndicesToFuse.empty()) return failure();
    return linalg::tileAndFuseLinalgOpAndSetMarker(
        rewriter, linalgOp, tileSizes, operandIndicesToFuse,
        getWorkGroupMarker());
  }
};
}  // namespace

void LinalgTileAndFusePass::runOnFunction() {
  FuncOp funcOp = getFunction();
  if (!isDispatchFunctionImpl(funcOp)) return;

  // By default look at the number of "parallel" loops in the generic op.
  Region &body = funcOp.getBody();
  // Only handle single block functions.
  if (body.getBlocks().size() != 1) {
    funcOp.emitError("unhandled dispatch function with multiple blocks");
    return signalPassFailure();
  }
  Block &block = body.front();
  auto linalgOps = block.getOps<linalg::LinalgOp>();
  if (linalgOps.empty()) {
    // Nothing to do.
    return;
  }

  OwningRewritePatternList patterns;
  patterns.insert<ExecuteSequentiallyPattern<linalg::ConvOp>,
                  ExecuteSequentiallyPattern<linalg::GenericOp>,
                  ExecuteSequentiallyPattern<linalg::IndexedGenericOp>,
                  TileLinalgOpPattern<linalg::GenericOp>,
                  TileLinalgOpPattern<linalg::IndexedGenericOp>,
                  TileLinalgOpPattern<linalg::MatmulOp>,
                  TileLinalgOpPattern<linalg::ConvOp>,
                  TileAndFuseLinalgOpPattern<linalg::GenericOp>>(&getContext(),
                                                                 workGroupSize);
  applyPatternsGreedily(getOperation(), patterns);
}

std::unique_ptr<OpPassBase<FuncOp>> createLinalgTileAndFusePass(
    ArrayRef<int64_t> workGroupSize) {
  return std::make_unique<LinalgTileAndFusePass>(workGroupSize);
}

static PassRegistration<LinalgTileAndFusePass> pass(
    "iree-linalg-tile-and-fuse", "Tile and Fuse Linalg operations on buffers",
    [] { return std::make_unique<LinalgTileAndFusePass>(); });
}  // namespace iree_compiler
}  // namespace mlir
