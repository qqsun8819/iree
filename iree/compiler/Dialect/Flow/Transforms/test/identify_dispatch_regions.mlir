// RUN: iree-opt -split-input-file -iree-flow-dispatchability-analysis -iree-flow-identify-dispatch-regions %s | IreeFileCheck %s

// CHECK-LABEL: @empty
func @empty() {
  // CHECK-NEXT: return
  return
}

// -----

// CHECK-LABEL: @simpleMath
func @simpleMath(%arg0 : tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NEXT: %[[WORKLOAD:.+]] = constant 4
  // CHECK-NEXT: %[[R1:.+]] = flow.dispatch.region
  // CHECK-SAME: [%[[WORKLOAD]] : index]
  // CHECK-SAME: (%arg1 = %arg0 : tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NEXT:   %1 = xla_hlo.add %arg1, %arg1 : tensor<4xf32>
  %0 = xla_hlo.add %arg0, %arg0 : tensor<4xf32>
  // CHECK-NEXT:   flow.return %1 : tensor<4xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: return %[[R1]] : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: @stdElementwiseOps
func @stdElementwiseOps(%arg0 : tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NEXT: %[[WORKLOAD:.+]] = constant 4
  // CHECK-NEXT: %[[R1:.+]] = flow.dispatch.region
  // CHECK-SAME: [%[[WORKLOAD]] : index]
  // CHECK-SAME: (%arg1 = %arg0 : tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NEXT:   %1 = addf %arg1, %arg1 : tensor<4xf32>
  %0 = addf %arg0, %arg0 : tensor<4xf32>
  // CHECK-NEXT:   %2 = subf %1, %arg1 : tensor<4xf32>
  %1 = subf %0, %arg0 : tensor<4xf32>
  // CHECK-NEXT:   %3 = mulf %2, %arg1 : tensor<4xf32>
  %2 = mulf %1, %arg0 : tensor<4xf32>
  // CHECK-NEXT:   flow.return %3 : tensor<4xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: return %[[R1]] : tensor<4xf32>
  return %2 : tensor<4xf32>
}

// -----

// CHECK-LABEL: @hloElementwiseOps
func @hloElementwiseOps(%arg0 : tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NEXT: %[[WORKLOAD:.+]] = constant 4
  // CHECK-NEXT: %0 = flow.dispatch.region
  // CHECK-SAME: [%[[WORKLOAD]] : index]
  // CHECK-SAME: (%arg1 = %arg0 : tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NEXT:   %1 = xla_hlo.add %arg1, %arg1 : tensor<4xf32>
  %0 = xla_hlo.add %arg0, %arg0 : tensor<4xf32>
  // CHECK-NEXT:   %2 = xla_hlo.subtract %1, %arg1 : tensor<4xf32>
  %1 = xla_hlo.subtract %0, %arg0 : tensor<4xf32>
  // CHECK-NEXT:   %3 = xla_hlo.multiply %2, %arg1 : tensor<4xf32>
  %2 = xla_hlo.multiply %1, %arg0 : tensor<4xf32>
  // CHECK-NEXT:   flow.return %3 : tensor<4xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: return %0 : tensor<4xf32>
  return %2 : tensor<4xf32>
}

// -----

// CHECK-LABEL: @interleavedDot
func @interleavedDot(%arg0 : tensor<4x4xf32>) -> tensor<4x4xf32> {
  // NOTE: Fragile ordering. Workload constants are emitted in order a the
  // top of the block.
  // CHECK: %[[WORKLOAD0:.+]] = constant 16 : index
  // CHECK: %[[WORKLOAD1:.+]] = constant 16 : index
  // CHECK: %[[WORKLOAD2:.+]] = constant 16 : index
  // CHECK: %[[R0:.+]] = flow.dispatch.region
  // CHECK-SAME: [%[[WORKLOAD0]] : index]
  // CHECK-SAME: (%arg1 = %arg0 : tensor<4x4xf32>) -> tensor<4x4xf32> {
  // CHECK-NEXT:   %3 = xla_hlo.add %arg1, %arg1 : tensor<4x4xf32>
  %0 = xla_hlo.add %arg0, %arg0 : tensor<4x4xf32>
  // CHECK-NEXT: flow.return %3 : tensor<4x4xf32>
  // CHECK-NEXT: }
  // CHECK: %[[R1:.+]] = flow.dispatch.region
  // CHECK-SAME: [%[[WORKLOAD1]] : index]
  // CHECK-SAME: (%arg1 = %[[R0]] : tensor<4x4xf32>, %arg2 = %arg0 : tensor<4x4xf32>) -> tensor<4x4xf32> {
  // CHECK-NEXT:   %3 = "xla_hlo.dot"(%arg1, %arg2) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %1 = "xla_hlo.dot"(%0, %arg0) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  // CHECK-NEXT:   flow.return %3 : tensor<4x4xf32>
  // CHECK-NEXT: }
  // CHECK: %[[R2:.+]] = flow.dispatch.region
  // CHECK-SAME: [%[[WORKLOAD2]] : index]
  // CHECK-SAME: (%arg1 = %[[R1]] : tensor<4x4xf32>, %arg2 = %arg0 : tensor<4x4xf32>) -> tensor<4x4xf32> {
  // CHECK-NEXT:   %3 = xla_hlo.multiply %arg1, %arg2 : tensor<4x4xf32>
  %2 = xla_hlo.multiply %1, %arg0 : tensor<4x4xf32>
  // CHECK-NEXT:   flow.return %3 : tensor<4x4xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: return %[[R2]] : tensor<4x4xf32>
  return %2 : tensor<4x4xf32>
}

// -----

// CHECK-LABEL: func @caller
func @caller(%arg0 : tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NEXT: %[[WORKLOAD0:.+]] = constant 4 : index
  // CHECK-NEXT: %[[R0:.+]] = flow.dispatch.region
  // CHECK-SAME: [%[[WORKLOAD0]] : index]
  // CHECK-SAME: (%arg1 = %arg0 : tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NEXT:   %1 = xla_hlo.add %arg1, %arg1 : tensor<4xf32>
  %0 = xla_hlo.add %arg0, %arg0 : tensor<4xf32>
  // CHECK-NEXT:   %2 = call @callee(%1) : (tensor<4xf32>) -> tensor<4xf32>
  %1 = call @callee(%0) : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK-NEXT:   %3 = xla_hlo.multiply %2, %arg1 : tensor<4xf32>
  %2 = xla_hlo.multiply %1, %arg0 : tensor<4xf32>
  // CHECK-NEXT:   flow.return %3 : tensor<4xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: return %[[R0]] : tensor<4xf32>
  return %2 : tensor<4xf32>
}
// CHECK-LABEL: func @callee
func @callee(%arg0 : tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: %[[WORKLOAD0:.+]] = constant 4 : index
  // CHECK: %[[R0:.+]] = flow.dispatch.region
  // CHECK-SAME: [%[[WORKLOAD0]] : index]
  // CHECK-SAME: (%arg1 = %arg0 : tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NEXT:   %1 = xla_hlo.multiply %arg1, %arg1 : tensor<4xf32>
  %0 = xla_hlo.multiply %arg0, %arg0 : tensor<4xf32>
  // CHECK-NEXT:   flow.return %1 : tensor<4xf32>
  // CHECK-NEXT: }
  // CHECK: return %[[R0]] : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: @single_reduction
func @single_reduction(%arg0 : tensor<4x8xf32>) -> tensor<4xf32> {
  // CHECK-DAG: %[[INITIAL:.+]] = constant dense<0.000000e+00>
  %0 = constant dense<0.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[WORKLOAD0:.+]] = constant 4 : index
  // CHECK: %[[RESULT:.+]] = flow.dispatch.region
  // CHECK-SAME: [%[[WORKLOAD0]] : index]
  // CHECK-SAME: (%arg1 = %arg0 : tensor<4x8xf32>, %arg2 = %[[INITIAL]] : tensor<f32>) -> tensor<4xf32>
  // CHECK-NEXT: = "xla_hlo.reduce"(%arg1, %arg2)
  %1 = "xla_hlo.reduce"(%arg0, %0) ( {
  ^bb0(%arg1 : tensor<f32>, %arg2 : tensor<f32>):
    %2 = xla_hlo.add %arg1, %arg2 : tensor<f32>
    "xla_hlo.return"(%2) : (tensor<f32>) -> ()
  }) {dimensions = dense<[1]> : tensor<1xi64>} : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
  // CHECK: flow.return
  // CHECK: return %[[RESULT]] : tensor<4xf32>
  return %1 : tensor<4xf32>
}

// -----

// CHECK-LABEL: @multi_reduction
func @multi_reduction(%arg0 : tensor<4x8xf32>, %arg1 : tensor<4x8xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  // CHECK-DAG: %[[INITIALA:.+]] = constant dense<0.000000e+00>
  %0 = constant dense<0.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[INITIALB:.+]] = constant dense<1.000000e+00>
  %1 = constant dense<1.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[WORKLOAD0:.+]] = constant 4 : index
  // CHECK: %[[RESULT:.+]]:2 = flow.dispatch.region
  // CHECK-SAME: [%[[WORKLOAD0]] : index]
  // CHECK-SAME: (%arg2 = %arg0 : tensor<4x8xf32>, %arg3 = %arg1 : tensor<4x8xf32>, %arg4 = %[[INITIALA]] : tensor<f32>, %arg5 = %[[INITIALB]] : tensor<f32>) -> (tensor<4xf32>, tensor<4xf32>)
  // CHECK-NEXT: = "xla_hlo.reduce"(%arg2, %arg3, %arg4, %arg5)
  %2, %3 = "xla_hlo.reduce"(%arg0, %arg1, %0, %1) ( {
  ^bb0(%arg0_lhs : tensor<f32>, %arg1_lhs : tensor<f32>, %arg0_rhs : tensor<f32>, %arg1_rhs : tensor<f32>):
    %4 = xla_hlo.add %arg0_lhs, %arg0_rhs : tensor<f32>
    %5 = xla_hlo.add %arg1_lhs, %arg1_rhs : tensor<f32>
    "xla_hlo.return"(%4, %5) : (tensor<f32>, tensor<f32>) -> ()
  }) {dimensions = dense<[1]> : tensor<1xi64>} : (tensor<4x8xf32>, tensor<4x8xf32>, tensor<f32>, tensor<f32>) -> (tensor<4xf32>, tensor<4xf32>)
  // CHECK: flow.return
  // CHECK: return %[[RESULT]]#0, %[[RESULT]]#1 : tensor<4xf32>, tensor<4xf32>
  return %2, %3 : tensor<4xf32>, tensor<4xf32>
}

// -----

// TODO(benvanik): windowed reduction.
