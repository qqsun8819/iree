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

#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"

#include "iree/compiler/Dialect/Shape/IR/ShapeTypes.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/SMLoc.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/STLExtras.h"

namespace mlir {
namespace iree_compiler {
namespace Shape {

/// Make a range of the DimType for a RankedShapeType with the number of
/// elements equal to the rank of the shaped type.
static RepeatRange<Type> getDimTypeRange(RankedShapeType type) {
  return make_repeated_range(
      static_cast<Type>(IndexType::get(type.getContext())), type.getRank());
}

/// Make a range of the DimType for a RankedShapeType with the number of
/// elements equal to the dynamic dimensions of the shaped type.
static RepeatRange<Type> getDimTypeDynamicRange(RankedShapeType type) {
  return make_repeated_range(
      static_cast<Type>(IndexType::get(type.getContext())),
      type.getNumDynamicDims());
}

//===----------------------------------------------------------------------===//
// Canonicalization
//===----------------------------------------------------------------------===//

class SafeCastCompatibleShapePattern
    : public OpRewritePattern<CastCompatibleShapeOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(CastCompatibleShapeOp op,
                                PatternRewriter &rewriter) const override {
    // TODO(laurenzo): This is just eliding if everything is the same. Make
    // it generic.
    auto resultRs = op.result().getType().dyn_cast<RankedShapeType>();
    if (resultRs) {
      // Casting to a ranked shape.
      for (auto operandType : op.getOperandTypes()) {
        auto operandRs = operandType.dyn_cast<RankedShapeType>();
        if (!operandRs || operandRs != resultRs) {
          return failure();
        }
      }
      rewriter.replaceOp(op, op.operands()[0]);
      return success();
    }

    return failure();
  }
};

class ElideTiedGetRankedShapePattern
    : public OpRewritePattern<GetRankedShapeOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(GetRankedShapeOp op,
                                PatternRewriter &rewriter) const override {
    // If the immediate predecessor is a TieShapeOp, then this op can be
    // erased in favor of the input to the tie op.
    auto tieOp = dyn_cast_or_null<TieShapeOp>(op.operand().getDefiningOp());
    if (!tieOp) return failure();

    rewriter.replaceOp(op, tieOp.shape());

    return success();
  }
};

class ElideDuplicateGetRankedShapePattern
    : public OpRewritePattern<GetRankedShapeOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(GetRankedShapeOp op,
                                PatternRewriter &rewriter) const override {
    // If the immediate predecessor is a GetRankedShapeOp, then this op can be
    // erased in favor of the input to the tie op.
    auto precedingGetRankedShapeOp =
        dyn_cast_or_null<GetRankedShapeOp>(op.operand().getDefiningOp());
    if (!precedingGetRankedShapeOp) return failure();

    rewriter.replaceOp(op, precedingGetRankedShapeOp.shape());
    return success();
  }
};

class ElideStaticGetRankedShapePattern
    : public OpRewritePattern<GetRankedShapeOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(GetRankedShapeOp op,
                                PatternRewriter &rewriter) const override {
    auto operandType = op.operand().getType().dyn_cast<RankedTensorType>();
    auto shapeType = op.shape().getType().dyn_cast<RankedShapeType>();
    if (!operandType || !shapeType || !operandType.hasStaticShape()) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<ConstRankedShapeOp>(op, shapeType);
    return success();
  }
};

class IdentityMakeRankedShapePattern
    : public OpRewritePattern<MakeRankedShapeOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(MakeRankedShapeOp op,
                                PatternRewriter &rewriter) const override {
    if (op.dynamic_dimensions().empty()) {
      // Do not match static shapes.
      return failure();
    }

    // Detects make_ranked_shape ops whose dynamic dimensions are provided by
    // ranked_dim ops that extract dimensions from an identical ranked_shape.
    auto rankedShape = op.getRankedShapeType();
    RankedDimOp commonRankedDimOp;
    unsigned previousProvidingIndex = 0;
    for (auto providingDim : op.dynamic_dimensions()) {
      auto rankedDimOp =
          llvm::dyn_cast_or_null<RankedDimOp>(providingDim.getDefiningOp());
      if (!rankedDimOp) return failure();

      // Shapes must match and refer to a dynamic index.
      unsigned providingIndex = rankedDimOp.getIndex();
      if (rankedDimOp.getRankedShapeType() != rankedShape ||
          !rankedShape.isDimDynamic(providingIndex)) {
        return failure();
      }

      if (commonRankedDimOp) {
        // Not first dim: verify same providing shape and indexes into next
        // dynamic dim.
        if (rankedDimOp.shape() != commonRankedDimOp.shape() ||
            providingIndex <= previousProvidingIndex) {
          return failure();
        }
      }

      commonRankedDimOp = rankedDimOp;
      previousProvidingIndex = rankedDimOp.getIndex();
    }

    // Fall-through: this op produces an identical shape as
    // commonRankedDimOp.
    assert(commonRankedDimOp &&
           "dynamic ranked_shape did not find a common provider");

    rewriter.replaceOp(op, commonRankedDimOp.shape());
    return success();
  }
};

class DynamicMakeRankedShapeDimPattern : public OpRewritePattern<RankedDimOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(RankedDimOp op,
                                PatternRewriter &rewriter) const override {
    // If the immediate predecessor is a MakeRankedShapeOp, then this op can be
    // erased in favor of the corresponding input to that op.
    auto makeRsOp =
        dyn_cast_or_null<MakeRankedShapeOp>(op.shape().getDefiningOp());
    if (!makeRsOp) return failure();

    RankedShapeType rsType = op.getRankedShapeType();
    unsigned index = op.getIndex();
    auto allDims = rsType.getAllDims();
    assert(index < allDims.size());
    if (allDims[index] >= 0) {
      // Not dynamic.
      return failure();
    }

    // Map the overall index to the dynamic dim index.
    int dynamicDimIndex = 0;
    for (unsigned i = 0; i < index; ++i) {
      if (allDims[i] < 0) dynamicDimIndex++;
    }

    assert(dynamicDimIndex < makeRsOp.dynamic_dimensions().size());
    rewriter.replaceOp(op, makeRsOp.dynamic_dimensions()[dynamicDimIndex]);
    return success();
  }
};

// Expands a shape.ranked_dims op into multiple shape.ranked_dim ops.
// This allows better folding of static dimensions.
class ExpandRankedShapeDimsPattern : public OpRewritePattern<RankedDimsOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(RankedDimsOp op,
                                PatternRewriter &rewriter) const override {
    auto rsType = op.getRankedShapeType();
    SmallVector<Value, 4> dims(rsType.getRank());
    for (int i = 0; i < rsType.getRank(); ++i) {
      dims[i] = rewriter.createOrFold<RankedDimOp>(
          op.getLoc(), op.getResult(i).getType(), op.shape(), i);
    }
    rewriter.replaceOp(op, dims);
    return success();
  }
};

class ElideDuplicateTieShapePattern : public OpRewritePattern<TieShapeOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(TieShapeOp op,
                                PatternRewriter &rewriter) const override {
    // If the immediate predecessor is a TieShapeOp, then it can be possible
    // to merge these. This can often happen when function/block tie_shape
    // placeholders are inserted prior to materializing later parts of the
    // computation.
    auto precedingTieShapeOp =
        dyn_cast_or_null<TieShapeOp>(op.operand().getDefiningOp());
    if (!precedingTieShapeOp) return failure();

    if (op.shape() != precedingTieShapeOp.shape()) {
      // This can happen in intermediate states before all shape calculations
      // are collapsed (i.e. the shapes may actually be equivalent but
      // constructed through different branches).
      return failure();
    }

    rewriter.replaceOp(op, precedingTieShapeOp.result());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// shape.tie_shape
//===----------------------------------------------------------------------===//

void TieShapeOp::build(Builder *builder, OperationState &result, Value operand,
                       Value shape) {
  result.types.push_back(operand.getType());
  result.addOperands({operand, shape});
}

static LogicalResult verifyTieShapeOp(TieShapeOp op) {
  if (op.operand().getType() != op.result().getType()) {
    return op.emitOpError("must have the same operand and result type");
  }

  // Validate RankedTensorType and ranked_shape_type conservatively in this
  // case (tie_shape supports arbitrary operand() but we constrain it if
  // it is specific enough.
  auto rankedTensorType = op.operand().getType().dyn_cast<RankedTensorType>();
  auto rsType = op.shape().getType().dyn_cast<RankedShapeType>();
  if (rankedTensorType && rsType) {
    if (!rankedTensorType.getShape().equals(rsType.getAllDims())) {
      return op.emitOpError("dims must match between tensor and shape");
    }
  }

  return success();
}

void TieShapeOp::getCanonicalizationPatterns(OwningRewritePatternList &patterns,
                                             MLIRContext *context) {
  patterns.insert<ElideDuplicateTieShapePattern>(context);
}

//===----------------------------------------------------------------------===//
// shape.cast_compatible_shape
//===----------------------------------------------------------------------===//

static LogicalResult verifyCastCompatibleShapeOp(CastCompatibleShapeOp op) {
  if (op.operands().empty()) {
    return op.emitOpError() << "Must have at least one operand";
  }

  auto resultRs = op.result().getType().dyn_cast<RankedShapeType>();
  if (resultRs) {
    // TODO(laurenzo): Expand this to check true compatibility instead of
    // just equality.
    // Casting to a ranked shape.
    for (auto operandType : op.getOperandTypes()) {
      auto operandRs = operandType.dyn_cast<RankedShapeType>();
      if (!operandRs || operandRs != resultRs) {
        return op.emitOpError()
               << "Incompatible static shape cast: " << operandRs << " -> "
               << resultRs;
      }
    }
    return success();
  }

  return failure();
}

void CastCompatibleShapeOp::getCanonicalizationPatterns(
    OwningRewritePatternList &patterns, MLIRContext *context) {
  patterns.insert<SafeCastCompatibleShapePattern>(context);
}

//===----------------------------------------------------------------------===//
// shape.get_ranked_shape
//===----------------------------------------------------------------------===//

void GetRankedShapeOp::build(Builder *builder, OperationState &result,
                             Value operand) {
  auto rankedOperandType = operand.getType().dyn_cast<RankedTensorType>();
  if (rankedOperandType) {
    result.types.push_back(RankedShapeType::get(rankedOperandType.getShape(),
                                                builder->getContext()));
  }
  result.addOperands(operand);
}

static LogicalResult verifyGetRankedShapeOp(GetRankedShapeOp op) {
  auto tensorType = op.operand().getType().cast<TensorType>();
  auto rsType = op.shape().getType().cast<RankedShapeType>();
  if (tensorType.getRank() != rsType.getRank()) {
    return op.emitOpError("operand and result must be of same rank");
  }
  auto rsDims = rsType.getAllDims();
  if (!std::equal(rsDims.begin(), rsDims.end(),
                  tensorType.getShape().begin())) {
    return op.emitOpError("operand tensor and result shape must be equal");
  }
  return success();
}

void GetRankedShapeOp::getCanonicalizationPatterns(
    OwningRewritePatternList &patterns, MLIRContext *context) {
  patterns
      .insert<ElideTiedGetRankedShapePattern, ElideStaticGetRankedShapePattern,
              ElideDuplicateGetRankedShapePattern>(context);
}

//===----------------------------------------------------------------------===//
// shape.const_ranked_shape
//===----------------------------------------------------------------------===//

void ConstRankedShapeOp::build(Builder *builder, OperationState &result,
                               Type type) {
  assert(type.cast<RankedShapeType>().isFullyStatic());
  result.types.push_back(type);
}

static LogicalResult verifyConstRankedShapeOp(ConstRankedShapeOp op) {
  auto rsType = op.result().getType().dyn_cast<RankedShapeType>();
  if (!rsType || !rsType.isFullyStatic()) {
    return op.emitOpError("must be a fully static ranked_shape");
  }
  return success();
}

void ConstRankedShapeOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  auto rankedShape = result().getType().cast<RankedShapeType>();
  SmallString<32> buffer;
  llvm::raw_svector_ostream os(buffer);
  os << "rs";
  interleave(
      rankedShape.getAllDims(), os, [&](int64_t dim) { os << dim; }, "_");
  setNameFn(getResult(), os.str());
}

//===----------------------------------------------------------------------===//
// shape.make_ranked_shape
//===----------------------------------------------------------------------===//

static LogicalResult verifyMakeRankedShapeOp(MakeRankedShapeOp op) {
  if (op.getRankedShapeType().getNumDynamicDims() != op.getNumOperands()) {
    return op.emitError()
           << "number of dynamic dims doesn't match number of operands";
  }
  return success();
}

void MakeRankedShapeOp::getCanonicalizationPatterns(
    OwningRewritePatternList &patterns, MLIRContext *context) {
  patterns.insert<IdentityMakeRankedShapePattern>(context);
}

//===----------------------------------------------------------------------===//
// shape.ranked_dim
//===----------------------------------------------------------------------===//

void RankedDimOp::build(Builder *builder, OperationState &result, Type dimType,
                        Value shape, int index) {
  result.addOperands(shape);
  result.addAttribute("index",
                      builder->getIntegerAttr(builder->getIndexType(), index));
  result.addTypes(dimType);
}

void RankedDimOp::build(Builder *builder, OperationState &result, Value shape,
                        int index) {
  RankedDimOp::build(builder, result, builder->getIndexType(), shape, index);
}

ParseResult parseRankedDimOp(OpAsmParser &parser, OperationState &state) {
  OpAsmParser::OperandType operand;
  Type operandType;
  IntegerAttr indexAttr;
  Type indexType = parser.getBuilder().getIndexType();
  SmallVector<Type, 1> resultTypes;
  if (parser.parseOperand(operand) || parser.parseLSquare() ||
      parser.parseAttribute(indexAttr, indexType, "index", state.attributes) ||
      parser.parseRSquare() || parser.parseColonType(operandType) ||
      parser.parseArrowTypeList(resultTypes) || resultTypes.empty() ||
      parser.resolveOperand(operand, operandType, state.operands)) {
    return failure();
  }

  auto rsType = operandType.dyn_cast<RankedShapeType>();
  if (!rsType) {
    return parser.emitError(parser.getNameLoc());
  }
  state.types.push_back(resultTypes[0]);
  return success();
}

static void printRankedDimOp(OpAsmPrinter &p, RankedDimOp op) {
  p << op.getOperationName() << " ";
  p.printOperand(op.shape());
  p << "[" << op.getIndex() << "]";
  p << " : ";
  p.printType(op.shape().getType());
  p << " -> ";
  p.printType(op.getType());
}

static LogicalResult verifyRankedDimOp(RankedDimOp op) {
  auto rsType = op.shape().getType().dyn_cast<RankedShapeType>();
  auto index = static_cast<int64_t>(op.getIndex());
  if (index < 0 || index >= rsType.getRank()) {
    return op.emitOpError() << "index out of bounds of shape";
  }
  return success();
}

OpFoldResult RankedDimOp::fold(ArrayRef<Attribute> operand) {
  auto rsType = shape().getType().cast<RankedShapeType>();
  int index = getIndex();
  if (!rsType.isDimDynamic(index)) {
    auto dimSize = rsType.getStaticDim(index);
    return IntegerAttr::get(getType(), dimSize);
  }

  return {};
}

void RankedDimOp::getCanonicalizationPatterns(
    OwningRewritePatternList &patterns, MLIRContext *context) {
  patterns.insert<DynamicMakeRankedShapeDimPattern>(context);
}

//===----------------------------------------------------------------------===//
// shape.ranked_dims
//===----------------------------------------------------------------------===//

void RankedDimsOp::build(Builder *builder, OperationState &result, Type dimType,
                         Value shape) {
  result.addOperands(shape);
  auto rankedShapeType = shape.getType().cast<RankedShapeType>();
  for (int i = 0; i < rankedShapeType.getRank(); ++i) {
    result.types.push_back(dimType);
  }
}

void RankedDimsOp::build(Builder *builder, OperationState &result,
                         Value shape) {
  RankedDimsOp::build(builder, result, builder->getIndexType(), shape);
}

void RankedDimsOp::getCanonicalizationPatterns(
    OwningRewritePatternList &patterns, MLIRContext *context) {
  patterns.insert<ExpandRankedShapeDimsPattern>(context);
}

#define GET_OP_CLASSES
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.cpp.inc"

}  // namespace Shape
}  // namespace iree_compiler
}  // namespace mlir
