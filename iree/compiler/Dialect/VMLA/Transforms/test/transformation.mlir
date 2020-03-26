// RUN: iree-opt -split-input-file -iree-vmla-transformation-pipeline %s | IreeFileCheck %s

func @simpleMath_rgn_dispatch_0() {
  %c0 = constant 0 : index
  %0 = hal.interface.load.tensor @legacy_io::@arg0, offset = %c0 : tensor<4xf32>
  %1 = call @simpleMath_rgn_dispatch_0_impl(%0) : (tensor<4xf32>) -> tensor<4xf32>
  hal.interface.store.tensor %1, @legacy_io::@ret0, offset = %c0 : tensor<4xf32>
  return
}
func @simpleMath_rgn_dispatch_0_impl(%arg0: tensor<4xf32>) -> tensor<4xf32> attributes {sym_visibility = "private"} {
  %0 = xla_hlo.add %arg0, %arg0 : tensor<4xf32>
  return %0 : tensor<4xf32>
}
hal.interface @legacy_io attributes {sym_visibility = "private"} {
  hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
  hal.interface.binding @ret0, set=0, binding=1, type="StorageBuffer", access="Write|Discard"
}

//      CHECK: func @simpleMath_rgn_dispatch_0(%arg0: !vmla.interface) {
// CHECK-NEXT:   %c0 = constant 0 : index
// CHECK-NEXT:   %c16 = constant 16 : index
// CHECK-NEXT:   %0 = "vmla.interface.binding"(%arg0) {binding = 0 : i32, set = 0 : i32} : (!vmla.interface) -> !vmla.buffer
// CHECK-NEXT:   %1 = "vmla.buffer.view"(%0, %c0, %c16) : (!vmla.buffer, index, index) -> !vmla.buffer
// CHECK-NEXT:   %2 = "vmla.buffer.alloc"(%c16) : (index) -> !vmla.buffer
// CHECK-NEXT:   "vmla.add"(%1, %1, %2) {element_type = f32} : (!vmla.buffer, !vmla.buffer, !vmla.buffer) -> ()
// CHECK-NEXT:   %3 = "vmla.interface.binding"(%arg0) {binding = 1 : i32, set = 0 : i32} : (!vmla.interface) -> !vmla.buffer
// CHECK-NEXT:   "vmla.buffer.copy"(%2, %c0, %3, %c0, %c16) : (!vmla.buffer, index, !vmla.buffer, index, index) -> ()
// CHECK-NEXT:   return
// CHECK-NEXT: }
