// RUN: iree-opt -split-input-file -iree-hal-transformation-pipeline -iree-hal-target-backends=vmla %s | IreeFileCheck %s

flow.executable @simpleMath_ex_dispatch_0 {
  flow.dispatch.entry @simpleMath_rgn_dispatch_0 attributes {
    workload = dense<[4, 1, 1]> : vector<3xi32>
  }
  module {
    func @simpleMath_rgn_dispatch_0(%arg0: tensor<4xf32>) -> tensor<4xf32> {
      %0 = xla_hlo.add %arg0, %arg0 : tensor<4xf32>
      return %arg0 : tensor<4xf32>
    }
  }
}

// CHECK-LABEL: hal.executable @simpleMath_ex_dispatch_0 {
//  CHECK-NEXT:   hal.interface @legacy_io {
//  CHECK-NEXT:     hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
//  CHECK-NEXT:     hal.interface.binding @ret0, set=0, binding=1, type="StorageBuffer", access="Write|Discard"
//  CHECK-NEXT:   }
//  CHECK-NEXT:   hal.executable.entry_point @simpleMath_rgn_dispatch_0 attributes {interface = @legacy_io, ordinal = 0 : i32, signature = (tensor<4xf32>) -> tensor<4xf32>, workgroup_size = dense<1> : vector<3xi32>}
//  CHECK-NEXT:   hal.executable.binary attributes {
//  CHECK-SAME:       data = dense<
//  CHECK-SAME:       format = 1447906369 : i32} {
//  CHECK-NEXT:     vm.module @module {
//  CHECK-NEXT:       vm.func @simpleMath_rgn_dispatch_0(%arg0: !vm.ref<!vmla.interface>) attributes {ordinal = 0 : i32} {
//  CHECK-NEXT:         %zero = vm.const.i32.zero : i32
//  CHECK-NEXT:         %c16 = vm.const.i32 16 : i32
//  CHECK-NEXT:         %c1 = vm.const.i32 1 : i32
//  CHECK-NEXT:         %ref = vm.call @vmla.interface.binding(%arg0, %zero, %zero) : (!vm.ref<!vmla.interface>, i32, i32) -> !vm.ref<!vmla.buffer>
//  CHECK-NEXT:         %ref_0 = vm.call @vmla.buffer.view(%ref, %zero, %c16) : (!vm.ref<!vmla.buffer>, i32, i32) -> !vm.ref<!vmla.buffer>
//  CHECK-NEXT:         %ref_1 = vm.call @vmla.interface.binding(%arg0, %zero, %c1) : (!vm.ref<!vmla.interface>, i32, i32) -> !vm.ref<!vmla.buffer>
//  CHECK-NEXT:         vm.call @vmla.buffer.copy(%ref_0, %zero, %ref_1, %zero, %c16) : (!vm.ref<!vmla.buffer>, i32, !vm.ref<!vmla.buffer>, i32, i32) -> ()
//  CHECK-NEXT:         vm.return
//  CHECK-NEXT:       }
//  CHECK-NEXT:       vm.export @simpleMath_rgn_dispatch_0 attributes {ordinal = 0 : i32}
//  CHECK-NEXT:       vm.import @vmla.interface.binding(%interface : !vm.ref<!vmla.interface>, %set : i32, %binding : i32) -> !vm.ref<!vmla.buffer> attributes {nosideeffects, ordinal = 0 : i32, sym_visibility = "private"}
//  CHECK-NEXT:       vm.import @vmla.buffer.view(%src : !vm.ref<!vmla.buffer>, %byte_offset : i32, %byte_length : i32) -> !vm.ref<!vmla.buffer> attributes {nosideeffects, ordinal = 1 : i32, sym_visibility = "private"}
//  CHECK-NEXT:       vm.import @vmla.buffer.copy(%src : !vm.ref<!vmla.buffer>, %src_byte_offset : i32, %dst : !vm.ref<!vmla.buffer>, %dst_byte_offset : i32, %byte_length : i32) attributes {ordinal = 2 : i32, sym_visibility = "private"}
//  CHECK-NEXT:     }
//  CHECK-NEXT:   }
//  CHECK-NEXT: }

// -----

flow.executable @reduction_ex_dispatch_0 {
  flow.dispatch.entry @reduction_ex_dispatch_0 attributes {workload = dense<[4, 1, 1]> : vector<3xi32>}
  module {
    func @reduction_ex_dispatch_0(%arg0: tensor<4x8xf32>) -> tensor<4xf32> {
      %cst = constant dense<0.000000e+00> : tensor<f32>
      %0 = "xla_hlo.reduce"(%arg0, %cst) ( {
      ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):	// no predecessors
        %1 = xla_hlo.add %arg1, %arg2 : tensor<f32>
        "xla_hlo.return"(%1) : (tensor<f32>) -> ()
      }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
      return %0 : tensor<4xf32>
    }
  }
}

// CHECK-LABEL: hal.executable @reduction_ex_dispatch_0 {
//  CHECK-NEXT:   hal.interface @legacy_io {
//  CHECK-NEXT:     hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
//  CHECK-NEXT:     hal.interface.binding @ret0, set=0, binding=1, type="StorageBuffer", access="Write|Discard"
//  CHECK-NEXT:   }
//  CHECK-NEXT:   hal.executable.entry_point @reduction_ex_dispatch_0 attributes {interface = @legacy_io, ordinal = 0 : i32, signature = (tensor<4x8xf32>) -> tensor<4xf32>, workgroup_size = dense<1> : vector<3xi32>}
//  CHECK-NEXT:   hal.executable.binary attributes {
//  CHECK-SAME:       data = dense<
//  CHECK-SAME:       format = 1447906369 : i32} {
//  CHECK-NEXT:     vm.module @module {
//  CHECK-NEXT:       vm.rodata @reduction_ex_dispatch_0_const_0 dense<0.000000e+00> : tensor<f32>
//  CHECK-NEXT:       vm.func @reduction_ex_dispatch_0(%arg0: !vm.ref<!vmla.interface>) attributes {ordinal = 0 : i32} {
//  CHECK-NEXT:         %zero = vm.const.i32.zero : i32
//  CHECK-NEXT:         %c128 = vm.const.i32 128 : i32
//  CHECK-NEXT:         %c16 = vm.const.i32 16 : i32
//  CHECK-NEXT:         %c4 = vm.const.i32 4 : i32
//  CHECK-NEXT:         %c8 = vm.const.i32 8 : i32
//  CHECK-NEXT:         %c1 = vm.const.i32 1 : i32
//  CHECK-NEXT:         %reduction_ex_dispatch_0_const_0 = vm.const.ref.rodata @reduction_ex_dispatch_0_const_0 : !vm.ref<!iree.byte_buffer>
//  CHECK-NEXT:         %ref = vm.call @vmla.buffer.const(%reduction_ex_dispatch_0_const_0) : (!vm.ref<!iree.byte_buffer>) -> !vm.ref<!vmla.buffer>
//  CHECK-NEXT:         %ref_0 = vm.call @vmla.interface.binding(%arg0, %zero, %zero) : (!vm.ref<!vmla.interface>, i32, i32) -> !vm.ref<!vmla.buffer>
//  CHECK-NEXT:         %ref_1 = vm.call @vmla.buffer.view(%ref_0, %zero, %c128) : (!vm.ref<!vmla.buffer>, i32, i32) -> !vm.ref<!vmla.buffer>
//  CHECK-NEXT:         %ref_2 = vm.call @vmla.buffer.alloc(%c16) : (i32) -> !vm.ref<!vmla.buffer>
//  CHECK-NEXT:         vm.call.variadic @vmla.reduce.sum.f32(%ref_1, [%c4, %c8], %ref, [], %c1, %ref_2, [%c4]) : (!vm.ref<!vmla.buffer>, i32..., !vm.ref<!vmla.buffer>, i32..., i32, !vm.ref<!vmla.buffer>, i32...)
//  CHECK-NEXT:         %ref_3 = vm.call @vmla.interface.binding(%arg0, %zero, %c1) : (!vm.ref<!vmla.interface>, i32, i32) -> !vm.ref<!vmla.buffer>
//  CHECK-NEXT:         vm.call @vmla.buffer.copy(%ref_2, %zero, %ref_3, %zero, %c16) : (!vm.ref<!vmla.buffer>, i32, !vm.ref<!vmla.buffer>, i32, i32) -> ()
//  CHECK-NEXT:         vm.return
//  CHECK-NEXT:       }
//  CHECK-NEXT:       vm.export @reduction_ex_dispatch_0 attributes {ordinal = 0 : i32}
//  CHECK-NEXT:       vm.import @vmla.interface.binding(%interface : !vm.ref<!vmla.interface>, %set : i32, %binding : i32) -> !vm.ref<!vmla.buffer> attributes {nosideeffects, ordinal = 0 : i32, sym_visibility = "private"}
//  CHECK-NEXT:       vm.import @vmla.buffer.const(%value : !vm.ref<!iree.byte_buffer>) -> !vm.ref<!vmla.buffer> attributes {nosideeffects, ordinal = 1 : i32, sym_visibility = "private"}
//  CHECK-NEXT:       vm.import @vmla.buffer.alloc(%byte_length : i32) -> !vm.ref<!vmla.buffer> attributes {nosideeffects, ordinal = 2 : i32, sym_visibility = "private"}
//  CHECK-NEXT:       vm.import @vmla.buffer.view(%src : !vm.ref<!vmla.buffer>, %byte_offset : i32, %byte_length : i32) -> !vm.ref<!vmla.buffer> attributes {nosideeffects, ordinal = 3 : i32, sym_visibility = "private"}
//  CHECK-NEXT:       vm.import @vmla.buffer.copy(%src : !vm.ref<!vmla.buffer>, %src_byte_offset : i32, %dst : !vm.ref<!vmla.buffer>, %dst_byte_offset : i32, %byte_length : i32) attributes {ordinal = 4 : i32, sym_visibility = "private"}
//  CHECK-NEXT:       vm.import @vmla.reduce.sum.f32(%src : !vm.ref<!vmla.buffer>, %src_shape : i32..., %init : !vm.ref<!vmla.buffer>, %init_shape : i32..., %dimension : i32, %dst : !vm.ref<!vmla.buffer>, %dst_shape : i32...) attributes {ordinal = 5 : i32, sym_visibility = "private"}
//  CHECK-NEXT:     }
//  CHECK-NEXT:   }
//  CHECK-NEXT: }
