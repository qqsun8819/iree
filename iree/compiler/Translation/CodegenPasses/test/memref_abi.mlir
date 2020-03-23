// RUN: iree-opt -pass-pipeline='hal.executable(hal.executable.source(module(iree-convert-hal-interface-to-memref)))' %s

hal.executable @pw_add_ex_dispatch_0 {
  hal.interface @legacy_io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.entry_point @pw_add_ex_dispatch_0 attributes {interface = @legacy_io, ordinal = 0 : i32, signature = (tensor<4x8xi32>, tensor<4x8xi32>) -> tensor<4x8xi32>, workgroup_size = dense<1> : vector<3xi32>}
  hal.executable.source {
    module {
      flow.executable @pw_add_ex_dispatch_0 {
        flow.dispatch.entry @pw_add_ex_dispatch_0
        module {
          // CHECK-LABEL: func @pw_add_ex_dispatch_0(
          func @pw_add_ex_dispatch_0() {
            %c0_i32 = constant 0 : i32
	    //  CHECK-DAG: %[[MEMREF0:.*]] = hal.interface.get_memref @legacy_io::@arg0
	    // CHECK-SAME: memref<4x8xi32>
	    //  CHECK-DAG: %[[MEMREF1:.*]] = hal.interface.get_memref @legacy_io::@arg1
	    // CHECK-SAME: memref<4x8xi32>
	    //  CHECK-DAG: %[[MEMREF2:.*]] = hal.interface.get_memref @legacy_io::@ret0
	    // CHECK-SAME: memref<4x8xi32>
	    //      CHECK: call @pw_add_ex_dispatch_0_impl(%[[MEMREF0]], %[[MEMREF1]], %[[MEMREF2]])
            %0 = hal.interface.load.tensor @legacy_io::@arg0, offset = %c0_i32 : tensor<4x8xi32>
            %1 = hal.interface.load.tensor @legacy_io::@arg1, offset = %c0_i32 : tensor<4x8xi32>
            %2 = call @pw_add_ex_dispatch_0_impl(%0, %1) : (tensor<4x8xi32>, tensor<4x8xi32>) -> tensor<4x8xi32>
            hal.interface.store.tensor %2, @legacy_io::@ret0, offset = %c0_i32 : tensor<4x8xi32>
            return
          }
	  //  CHECK-DAG: func @pw_add_ex_dispatch_0_impl(
	  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]*]]: memref<4x8xi32>
	  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9_]*]]: memref<4x8xi32>
	  // CHECK-SAME: %[[ARG2:[a-zA-Z0-9_]*]]: memref<4x8xi32>)
          func @pw_add_ex_dispatch_0_impl(%arg0: tensor<4x8xi32>, %arg1: tensor<4x8xi32>) -> tensor<4x8xi32> attributes {sym_visibility = "private"} {
	    //  CHECK-DAG: %[[T0:.*]] = iree.load_tensor(%[[ARG0]]
	    //  CHECK-DAG: %[[T1:.*]] = iree.load_tensor(%[[ARG1]]
	    //      CHECK: %[[T2:.*]] = xla_hlo.add %[[T0]], %[[T1]]
	    //      CHECK: iree.store_output(
	    // CHECK-SAME: %[[T2]]
	    // CHECK-SAME: %[[ARG2]]
	    //      CHECK: return
            %0 = xla_hlo.add %arg0, %arg1 : tensor<4x8xi32>
            return %0 : tensor<4x8xi32>
          }
          hal.interface @legacy_io attributes {sym_visibility = "private"} {
            hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
            hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
            hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
          }
        }
      }
    }
  }
}
