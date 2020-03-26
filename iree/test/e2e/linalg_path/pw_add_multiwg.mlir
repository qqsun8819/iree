// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-run-mlir -iree-hal-target-backends=vulkan-spirv -iree-use-linalg-to-spirv-path -iree-linalg-to-spirv-workgroup-size=2,2,1 %s | IreeFileCheck %s)

// CHECK-LABEL: EXEC @pw_add
// CHECK: 4x8xi32=[3 6 9 12 15 18 21 24][27 30 33 36 39 42 45 48][51 54 57 60 63 66 69 72][75 78 81 84 87 90 93 96]
module {
  func @pw_add() -> tensor<4x8xi32> {
    %0 = iree.unfoldable_constant dense<[[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16], [17, 18, 19, 20, 21, 22, 23, 24], [25, 26, 27, 28, 29, 30, 31, 32]]> : tensor<4x8xi32>
    %1 = iree.unfoldable_constant dense<[[2, 4, 6, 8, 10, 12, 14, 16], [18, 20, 22, 24, 26, 28, 30, 32], [34, 36, 38, 40, 42, 44, 46, 48], [50, 52, 54, 56, 58, 60, 62, 64]]> : tensor<4x8xi32>
    %2 = "xla_hlo.add"(%0, %1) : (tensor<4x8xi32>, tensor<4x8xi32>) -> tensor<4x8xi32>
    return %2 : tensor<4x8xi32>
  }
}
