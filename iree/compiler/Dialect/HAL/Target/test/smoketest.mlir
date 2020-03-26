// RUN: iree-opt -split-input-file -iree-hal-transformation-pipeline -iree-hal-target-backends=vulkan-spirv %s | IreeFileCheck %s -check-prefix=VKSPV

flow.executable @simpleMath_ex_dispatch_0 {
  flow.dispatch.entry @simpleMath_rgn_dispatch_0 attributes {
      workload = 4 : index
  }
  module {
    func @simpleMath_rgn_dispatch_0(%arg0: tensor<4xf32>) -> tensor<4xf32> {
      %0 = xla_hlo.add %arg0, %arg0 : tensor<4xf32>
      return %0 : tensor<4xf32>
    }
  }
}

// VKSPV-LABEL: hal.executable @simpleMath_ex_dispatch_0
// VKSPV-NEXT:   hal.interface @legacy_io {
// VKSPV-DAG:      hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
// VKSPV-DAG:      hal.interface.binding @ret0, set=0, binding=1, type="StorageBuffer", access="Write|Discard"
// VKSPV-NEXT:   }
// VKSPV-NEXT:   hal.executable.entry_point @simpleMath_rgn_dispatch_0 attributes {interface = @legacy_io, ordinal = 0 : i32, signature = (tensor<4xf32>) -> tensor<4xf32>, workgroup_size = [32 : index, 1 : index, 1 : index]
// VKSPV-NEXT:   hal.executable.binary attributes {
// VKSPV-SAME:     data = dense
// VKSPV-SAME:     format = 1397773893 : i32} {
// VKSPV-NEXT:     module attributes {spv.target_env = #spv.target_env<#spv.vce<v1.3, [Shader], [SPV_KHR_storage_buffer_storage_class]>, {max_compute_workgroup_invocations = 128 : i32, max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>} {
// VKSPV-NEXT:       spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]> {
//  VKSPV-DAG:         spv.globalVariable [[GLOBALID:@.*]] built_in("GlobalInvocationId")
//  VKSPV-DAG:         spv.globalVariable [[NUMWORKGROUPS:@.*]] built_in("NumWorkgroups")
//      VKSPV:         spv.EntryPoint "GLCompute" @simpleMath_rgn_dispatch_0, [[GLOBALID]], [[NUMWORKGROUPS]]
// VKSPV-NEXT:         spv.ExecutionMode @simpleMath_rgn_dispatch_0 "LocalSize", 32, 1, 1
// VKSPV-NEXT:       }
// VKSPV-NEXT:     }
// VKSPV-NEXT:   }

// -----

flow.executable @reduction_ex_reduce_0_dim_0 {
  flow.dispatch.entry @reduction_rgn_reduce_0_dim_0_entry attributes {
    workload = 4 : index
  }
  module {
    func @reduction_rgn_reduce_0_dim_0_entry(%arg0 : tensor<4x8xf32>, %arg1 : tensor<f32>) -> tensor<4xf32> {
      %0 = "xla_hlo.reduce"(%arg0, %arg1) ( {
      ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
        %1 = xla_hlo.add %arg3, %arg4 : tensor<f32>
        "xla_hlo.return"(%1) : (tensor<f32>) -> ()
      }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
      return %0 : tensor<4xf32>
    }
  }
}

// VKSPV-LABEL: hal.executable @reduction_ex_reduce_0_dim_0
// VKSPV-NEXT:   hal.interface @legacy_io {
// VKSPV-DAG:      hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
// VKSPV-DAG:      hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
// VKSPV-DAG:      hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
// VKSPV-NEXT:   }
// VKSPV-NEXT:   hal.executable.entry_point @reduction_rgn_reduce_0_dim_0_entry attributes {
// VKSPV-SAME:     interface = @legacy_io,
// VKSPV-SAME:     ordinal = 0 : i32,
// VKSPV-SAME:     signature = (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>,
// VKSPV-SAME:     workgroup_size = [32 : index, 1 : index, 1 : index]
// VKSPV-SAME:   }
// VKSPV-NEXT:   hal.executable.binary attributes {
// VKSPV-SAME:     data = dense
// VKSPV-SAME:     format = 1397773893 : i32} {
