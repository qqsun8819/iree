// RUN: iree-opt -split-input-file -iree-index-computation -simplify-spirv-affine-exprs=false -verify-diagnostics -o - %s | IreeFileCheck %s

module {
  // CHECK: func @concatenate
  // CHECK-SAME: [[ARG0:%[a-zA-Z0-9_]*]]: memref<1x64xf32>
  // CHECK-SAME: iree.index_computation_info
  // CHECK-SAME: operand_indices
  // CHECK-SAME: []
  // CHECK-SAME: result_index
  // CHECK-SAME: [affine_map<(d0, d1) -> (0, d0)>]
  // CHECK-SAME: [[ARG1:%[a-zA-Z0-9_]*]]: memref<1x10xf32>
  // CHECK-SAME: iree.index_computation_info
  // CHECK-SAME: operand_indices
  // CHECK-SAME: []
  // CHECK-SAME: result_index
  // CHECK-SAME: [affine_map<(d0, d1) -> (0, d0 - 64)>]
  func @concatenate(%arg0: memref<1x64xf32>, %arg1 : memref<1x10xf32>, %arg2 : memref<1x74xf32>)
  attributes  {iree.executable.export, iree.executable.workgroup_size = [1 : index, 1 : index, 1 : index], iree.ordinal = 0 : i32} {
    %0 = iree.load_input(%arg0 : memref<1x64xf32>) : tensor<1x64xf32>
    %1 = iree.load_input(%arg1 : memref<1x10xf32>) : tensor<1x10xf32>
    // CHECK: xla_hlo.concatenate
    // CHECK-SAME: iree.index_computation_info
    // CHECK-SAME: operand_indices
    // CHECK-SAME: [affine_map<(d0, d1) -> (0, d0)>]
    // CHECK-SAME: [affine_map<(d0, d1) -> (0, d0 - 64)>]
    // CHECK-SAME: result_index
    // CHECK-SAME: [affine_map<(d0, d1) -> (0, d0)>]
    %2 = "xla_hlo.concatenate"(%0, %1) {dimension = 1 : i64} : (tensor<1x64xf32>, tensor<1x10xf32>) -> tensor<1x74xf32>
    iree.store_output(%2 : tensor<1x74xf32>, %arg2 : memref<1x74xf32>)
    return
  }
}
