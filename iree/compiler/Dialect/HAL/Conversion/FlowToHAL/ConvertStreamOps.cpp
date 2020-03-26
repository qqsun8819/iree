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

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/Conversion/FlowToHAL/ConvertFlowToHAL.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/HAL/Utils/TypeUtils.h"
#include "iree/compiler/Dialect/IREE/IR/IREETypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Module.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "iree-hal"

namespace mlir {
namespace iree_compiler {
namespace {

struct BufferRange {
  BufferRange() = default;
  explicit BufferRange(Value buffer) : buffer(buffer) {}

  Value buffer = nullptr;
};

// Allocated buffers used within the stream.
struct BufferSet {
  explicit BufferSet(Value allocator) : allocator(allocator) {}

  // Allocator instance the buffers come from.
  Value allocator = nullptr;

  // All output buffers in the same order as the original results.
  SmallVector<Value, 4> outputBuffers;

  // Maps tensor values within the stream to a buffer range that stores them.
  DenseMap<Value, BufferRange> rangeMap;
};

// Allocates a buffer for the given stream output value.
// |streamValue| is the Value used within the stream region and
// |externalValue| is the returned value from the stream region in the parent
// block.
static Value allocateOutputBuffer(Value streamValue, Value externalValue,
                                  Value allocator,
                                  ConversionPatternRewriter &rewriter) {
  // TODO(benvanik): compute from SSA use-def chain uses.
  IREE::HAL::MemoryTypeBitfield memoryTypes =
      IREE::HAL::MemoryTypeBitfield::DeviceLocal |
      IREE::HAL::MemoryTypeBitfield::HostVisible;
  IREE::HAL::BufferUsageBitfield bufferUsage =
      IREE::HAL::BufferUsageBitfield::All;

  // Compute the allocation size for the value.
  auto elementType = IREE::HAL::getElementTypeValue(
      streamValue.getType().cast<ShapedType>().getElementType());
  if (!elementType) {
    return {};
  }
  auto shape = IREE::HAL::getShapeDims(streamValue, rewriter);
  auto allocationSize =
      rewriter
          .create<IREE::HAL::AllocatorComputeSizeOp>(
              externalValue.getLoc(), allocator, shape, elementType.getValue())
          .getResult();

  auto buffer = rewriter
                    .create<IREE::HAL::AllocatorAllocateOp>(
                        externalValue.getLoc(), allocator, memoryTypes,
                        bufferUsage, allocationSize)
                    .getResult();

  // TODO(benvanik): implement resource sets.
  rewriter.create<IREE::HAL::ExDeferReleaseOp>(externalValue.getLoc(), buffer);

  return buffer;
}

// Allocates all output buffers for the stream and populates the |bufferSet|
// with the new mappings.
static void allocateOutputBuffers(IREE::Flow::ExStreamFragmentOp streamOp,
                                  BufferSet &bufferSet,
                                  ConversionPatternRewriter &rewriter) {
  // Allocate output buffers and replace the original uses with the buffers.
  auto returnOp = cast<IREE::Flow::ReturnOp>(streamOp.body().front().back());
  for (auto result : llvm::enumerate(streamOp.getResults())) {
    auto streamValue = returnOp.getOperand(result.index());
    auto externalValue = result.value();
    auto buffer = allocateOutputBuffer(streamValue, externalValue,
                                       bufferSet.allocator, rewriter);
    auto bufferRange = BufferRange{buffer};
    bufferSet.rangeMap[externalValue] = bufferRange;
    bufferSet.rangeMap[streamValue] = bufferRange;
    bufferSet.outputBuffers.push_back(buffer);
  }
}

// Allocates a transient buffer for use entirely within the command buffer.
static Value allocateTransientBuffer(Value streamValue, Value allocator,
                                     ConversionPatternRewriter &rewriter) {
  // TODO(benvanik): compute from SSA use-def chain uses.
  IREE::HAL::MemoryTypeBitfield memoryTypes =
      IREE::HAL::MemoryTypeBitfield::DeviceLocal;
  IREE::HAL::BufferUsageBitfield bufferUsage =
      IREE::HAL::BufferUsageBitfield::Dispatch |
      IREE::HAL::BufferUsageBitfield::Transfer;

  // Compute the allocation size for the value.
  auto elementType = IREE::HAL::getElementTypeValue(
      streamValue.getType().cast<ShapedType>().getElementType());
  if (!elementType) {
    return {};
  }
  auto shape = IREE::HAL::getShapeDims(streamValue, rewriter);
  auto allocationSize =
      rewriter
          .create<IREE::HAL::AllocatorComputeSizeOp>(
              streamValue.getLoc(), allocator, shape, elementType.getValue())
          .getResult();

  auto buffer = rewriter
                    .create<IREE::HAL::AllocatorAllocateOp>(
                        streamValue.getLoc(), allocator, memoryTypes,
                        bufferUsage, allocationSize)
                    .getResult();

  // TODO(benvanik): implement resource sets.
  rewriter.create<IREE::HAL::ExDeferReleaseOp>(streamValue.getLoc(), buffer);

  return buffer;
}

// Allocates transient buffers to store the intra-stream results and populates
// the |bufferSet| with the new mappings.
static void allocateTransientBuffers(IREE::Flow::ExStreamFragmentOp streamOp,
                                     BufferSet &bufferSet,
                                     ConversionPatternRewriter &rewriter) {
  for (auto &op : streamOp.body().front()) {
    for (auto result : op.getResults()) {
      // If the result is an output buffer we can just use that directly.
      if (bufferSet.rangeMap[result].buffer) continue;

      auto buffer =
          allocateTransientBuffer(result, bufferSet.allocator, rewriter);
      bufferSet.rangeMap[result] = BufferRange{buffer};
    }
  }
}

// Returns a the (x, y, z) workgroup counts calculated from the given |workload|
// (invocation count) and the workgroup size of the dispatch |entryPointOp|.
static std::array<Value, 3> getDispatchWorkgroupCounts(
    IREE::HAL::ExecutableEntryPointOp entryPointOp, Value workload,
    ConversionPatternRewriter &rewriter) {
  std::array<Value, 3> result;
  auto loc = entryPointOp.getLoc();
  auto i32Type = rewriter.getIntegerType(32);
  auto constantOne = rewriter.createOrFold<mlir::ConstantOp>(
      loc, rewriter.getI32IntegerAttr(1));
  workload = rewriter.createOrFold<mlir::IndexCastOp>(loc, i32Type, workload);
  for (int i = 0; i < 3; ++i) {
    // Round up: (workload + workgroup_size - 1) / workgroup_size;
    auto workgroupSizeI = rewriter.createOrFold<mlir::ConstantOp>(
        loc, rewriter.getI32IntegerAttr(
                 entryPointOp.workgroup_size().getValue<int32_t>(
                     {static_cast<uint64_t>(i)})));
    auto rounded = rewriter.createOrFold<mlir::SubIOp>(
        loc, rewriter.createOrFold<mlir::AddIOp>(loc, workload, workgroupSizeI),
        constantOne);
    auto workgroupCountI = rewriter.createOrFold<mlir::UnsignedDivIOp>(
        loc, rounded, workgroupSizeI);
    result[i] = workgroupCountI;

    // Multiply back out and subtract from invocations.
    workload = rewriter.createOrFold<SubIOp>(
        loc, workload,
        rewriter.createOrFold<MulIOp>(loc, workgroupCountI, rounded));

    // Ensure > 0.
    auto workloadGreaterZero =
        rewriter.create<CmpIOp>(loc, CmpIPredicate::sge, workload, constantOne);
    workload = rewriter.create<SelectOp>(loc, workloadGreaterZero, workload,
                                         constantOne);
  }
  return result;
}

// Records a full execution barrier that forces visibility of all buffers.
static void recordFullExecutionBarrier(Value commandBuffer, Location loc,
                                       ConversionPatternRewriter &rewriter) {
  auto memoryBarrier =
      rewriter
          .create<IREE::HAL::MakeMemoryBarrierOp>(
              loc, IREE::HAL::AccessScopeBitfield::DispatchWrite,
              IREE::HAL::AccessScopeBitfield::DispatchRead)
          .getResult();
  rewriter.create<IREE::HAL::CommandBufferExecutionBarrierOp>(
      loc, commandBuffer,
      IREE::HAL::ExecutionStageBitfield::CommandRetire |
          IREE::HAL::ExecutionStageBitfield::Dispatch,
      IREE::HAL::ExecutionStageBitfield::CommandIssue |
          IREE::HAL::ExecutionStageBitfield::Dispatch,
      ArrayRef<Value>{memoryBarrier}, ArrayRef<Value>{});
}

static void recordPushBindings(Value device, Value commandBuffer,
                               IREE::Flow::DispatchOp &dispatchOp,
                               IREE::HAL::ExecutableOp &executableOp,
                               IREE::HAL::ExecutableEntryPointOp &entryPointOp,
                               BufferSet &bufferSet,
                               ConversionPatternRewriter &rewriter) {
  int bindingOrdinal = 0;
  auto pushBinding = [&](Value tensorValue) {
    auto &bufferRange = bufferSet.rangeMap[tensorValue];
    IREE::HAL::TensorRewriteAdaptor value(dispatchOp.getLoc(), tensorValue,
                                          bufferRange.buffer, rewriter);
    rewriter.create<IREE::HAL::ExPushBindingOp>(
        dispatchOp.getLoc(), commandBuffer,
        rewriter.getI32IntegerAttr(bindingOrdinal++), value.getBuffer(),
        value.getShapeDims(), value.getElementTypeAttr());
    rewriter.create<IREE::HAL::ExDeferReleaseOp>(dispatchOp.getLoc(),
                                                 bufferRange.buffer);
  };
  for (auto tensorValue : dispatchOp.operands()) {
    pushBinding(tensorValue);
  }
  for (auto tensorValue : dispatchOp.results()) {
    pushBinding(tensorValue);
  }
}

// Records a dispatch operation.
static void recordDispatch(Value device, Value commandBuffer,
                           IREE::Flow::DispatchOp &dispatchOp,
                           BufferSet &bufferSet,
                           ConversionPatternRewriter &rewriter) {
  // Get the handle to the executable that is compatible with our device.
  auto executable =
      rewriter
          .create<IREE::HAL::ExecutableLookupOp>(dispatchOp.getLoc(), device,
                                                 dispatchOp.executable())
          .getResult();
  auto executableOp =
      cast<IREE::HAL::ExecutableOp>(SymbolTable::lookupNearestSymbolFrom(
          dispatchOp, dispatchOp.executable()));

  // Compute the workgroup count based on the executable's tiling.
  auto entryPointOp = cast<IREE::HAL::ExecutableEntryPointOp>(
      SymbolTable::lookupSymbolIn(executableOp, dispatchOp.entry_point()));
  auto workgroupCounts = getDispatchWorkgroupCounts(
      entryPointOp, rewriter.getRemappedValue(dispatchOp.workload()), rewriter);

  // Setup bindings, right now pushed immediately but soon to be replaced with
  // descriptor sets (or something better, anyway).
  recordPushBindings(device, commandBuffer, dispatchOp, executableOp,
                     entryPointOp, bufferSet, rewriter);

  rewriter.create<IREE::HAL::CommandBufferDispatchOp>(
      dispatchOp.getLoc(), commandBuffer, executable, entryPointOp,
      workgroupCounts[0], workgroupCounts[1], workgroupCounts[2]);

  // Full barriers for now as we aren't scheduling things.
  // TODO(benvanik): don't add at the end of the command buffer (we could
  // also do a canonicalization step that removed trailing barriers).
  recordFullExecutionBarrier(commandBuffer, dispatchOp.getLoc(), rewriter);
}

static void recordTensorUpdate(Value device, Value commandBuffer,
                               IREE::Flow::TensorUpdateOp &updateOp,
                               BufferSet &bufferSet,
                               ConversionPatternRewriter &rewriter) {
  auto &updateBuffer = bufferSet.rangeMap[updateOp.update()];
  auto &targetBuffer = bufferSet.rangeMap[updateOp.target()];
  auto &resultBuffer = bufferSet.rangeMap[updateOp.result()];

  // TODO(benvanik): use something other than the BufferRange::buffer?
  // This may require us to subview the buffer first.
  IREE::HAL::TensorRewriteAdaptor update(updateOp.getLoc(), updateOp.update(),
                                         updateBuffer.buffer, rewriter);
  IREE::HAL::TensorRewriteAdaptor target(updateOp.getLoc(), updateOp.target(),
                                         targetBuffer.buffer, rewriter);
  IREE::HAL::TensorRewriteAdaptor result(updateOp.getLoc(), updateOp.result(),
                                         resultBuffer.buffer, rewriter);

  auto zeroOffset = rewriter.createOrFold<mlir::ConstantOp>(
      updateOp.getLoc(), rewriter.getI32IntegerAttr(0));

  // Compute the size of the update range.
  auto startIndices = llvm::to_vector<4>(llvm::map_range(
      updateOp.start_indices(),
      [&](Value value) { return rewriter.getRemappedValue(value); }));
  auto targetRange = target.computeRange(startIndices, update.getShapeDims());

  // TODO(benvanik): actual buffer allocation so we aren't doing this copy.
  rewriter.create<IREE::HAL::CommandBufferCopyBufferOp>(
      updateOp.getLoc(), commandBuffer, target.getBuffer(), zeroOffset,
      result.getBuffer(), zeroOffset, target.getByteLength());
  // TODO(benvanik): slice left/mid/right, but really just don't do this.
  recordFullExecutionBarrier(commandBuffer, updateOp.getLoc(), rewriter);
  rewriter.create<IREE::HAL::CommandBufferCopyBufferOp>(
      updateOp.getLoc(), commandBuffer, update.getBuffer(), zeroOffset,
      result.getBuffer(), targetRange.offset, targetRange.length);

  // TODO(benvanik): implement resource sets.
  rewriter.create<IREE::HAL::ExDeferReleaseOp>(updateOp.getLoc(),
                                               targetBuffer.buffer);
  rewriter.create<IREE::HAL::ExDeferReleaseOp>(updateOp.getLoc(),
                                               updateBuffer.buffer);
  rewriter.create<IREE::HAL::ExDeferReleaseOp>(updateOp.getLoc(),
                                               resultBuffer.buffer);

  // Full barriers for now as we aren't scheduling things.
  // TODO(benvanik): don't add at the end of the command buffer (we could
  // also do a canonicalization step that removed trailing barriers).
  recordFullExecutionBarrier(commandBuffer, updateOp.getLoc(), rewriter);
}

static LogicalResult recordStreamCommands(Value device, Value commandBuffer,
                                          Block &streamBlock,
                                          BufferSet &bufferSet,
                                          ConversionPatternRewriter &rewriter) {
  for (auto &op : streamBlock) {
    if (auto dispatchOp = dyn_cast<IREE::Flow::DispatchOp>(op)) {
      recordDispatch(device, commandBuffer, dispatchOp, bufferSet, rewriter);
    } else if (auto updateOp = dyn_cast<IREE::Flow::TensorUpdateOp>(op)) {
      recordTensorUpdate(device, commandBuffer, updateOp, bufferSet, rewriter);
    } else if (auto returnOp = dyn_cast<IREE::Flow::ReturnOp>(op)) {
      // No-op; handled by the buffer allocation.
    } else {
      return op.emitOpError() << "unexpected in stream";
    }
  }
  return success();
}

class ExStreamFragmentOpConversion
    : public OpConversionPattern<IREE::Flow::ExStreamFragmentOp> {
 public:
  using OpConversionPattern<
      IREE::Flow::ExStreamFragmentOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Flow::ExStreamFragmentOp streamOp, llvm::ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    // TODO(benvanik): choose buffer mode/category based on stream commands.
    auto mode = IREE::HAL::CommandBufferModeBitfield::OneShot;
    auto category = IREE::HAL::CommandCategoryBitfield::Dispatch |
                    IREE::HAL::CommandCategoryBitfield::Transfer;

    // We'll use this buffer set to track the original and converted tensors
    // and buffers during conversion. Ideally we'd run some fancy allocation
    // analysis first to produce it.
    auto device =
        rewriter.createOrFold<IREE::HAL::ExSharedDeviceOp>(streamOp.getLoc());
    auto allocator =
        rewriter.create<IREE::HAL::DeviceAllocatorOp>(streamOp.getLoc(), device)
            .getResult();
    BufferSet bufferSet{allocator};

    // Remap non-tensor operands (such as workloads).
    auto &entryBlock = streamOp.body().front();
    for (int i = 0; i < operands.size(); ++i) {
      if (operands[i].getType().isa<IREE::HAL::BufferType>()) {
        bufferSet.rangeMap[entryBlock.getArgument(i)] =
            BufferRange{operands[i]};
      } else {
        rewriter.replaceUsesOfBlockArgument(entryBlock.getArgument(i),
                                            operands[i]);
      }
    }

    // Allocate buffers for outputs and transient buffers.
    allocateOutputBuffers(streamOp, bufferSet, rewriter);
    allocateTransientBuffers(streamOp, bufferSet, rewriter);

    // Allocate and begin the command buffer.
    // In a real version we would want to pick the device based on the placement
    // information attached to the stream.
    auto commandBuffer =
        rewriter.createOrFold<IREE::HAL::CommandBufferCreateOp>(
            streamOp.getLoc(), device, mode, category);
    rewriter.create<IREE::HAL::CommandBufferBeginOp>(streamOp.getLoc(),
                                                     commandBuffer);

    // Record all of the commands into the command buffer.
    if (failed(recordStreamCommands(device, commandBuffer, entryBlock,
                                    bufferSet, rewriter))) {
      return failure();
    }

    // End and submit the command buffer.
    // In a real version we'd want to setup a semaphore chain instead of
    // submitting and waiting.
    rewriter.create<IREE::HAL::CommandBufferEndOp>(streamOp.getLoc(),
                                                   commandBuffer);
    rewriter.create<IREE::HAL::ExSubmitAndWaitOp>(streamOp.getLoc(), device,
                                                  commandBuffer);

    // It's annoying, but we need to do this replacement at the very end as
    // otherwise we lose access to the original values (which we need for
    // shape information).
    for (int i = 0; i < operands.size(); ++i) {
      if (operands[i].getType().isa<IREE::HAL::BufferType>()) {
        rewriter.replaceUsesOfBlockArgument(entryBlock.getArgument(i),
                                            operands[i]);
      }
    }

    rewriter.replaceOp(streamOp, bufferSet.outputBuffers);
    return success();
  }
};

}  // namespace

void populateFlowStreamToHALPatterns(MLIRContext *context,
                                     OwningRewritePatternList &patterns,
                                     TypeConverter &converter) {
  patterns.insert<ExStreamFragmentOpConversion>(context);
}

}  // namespace iree_compiler
}  // namespace mlir
