// RUN: iree-opt -split-input-file -iree-convert-flow-to-hal -canonicalize %s | IreeFileCheck %s

hal.executable @ex0 {
  hal.interface @interface {
    hal.interface.binding @s0b0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @s0b1, set=0, binding=1, type="StorageBuffer", access="Read|Write"
  }
  hal.executable.entry_point @entry0 attributes {
    interface = @interface,
    ordinal = 0 : i32,
    signature = (tensor<128xf32>) -> tensor<128xf32>,
    workgroup_size = [32 : index, 1 : index, 1 : index]
  }
}

// CHECK-LABEL: func @multipleDispatches
func @multipleDispatches(%arg0: tensor<128xf32>) -> tensor<128xf32> {
  // CHECK-DAG: [[C1:%.+]] = constant 1
  // CHECK-DAG: [[C4:%.+]] = constant 4
  // CHECK-DAG: [[C128:%.+]] = constant 128
  %cst = constant 128 : index
  // CHECK: [[RET_BUF:%.+]] = hal.allocator.allocate {{.+}}, "HostVisible|DeviceVisible|DeviceLocal", "Constant|Transfer|Mapping|Dispatch"
  // CHECK-NEXT: hal.ex.defer_release [[RET_BUF]]
  // CHECK: [[TMP_BUF:%.+]] = hal.allocator.allocate {{.+}}, "DeviceVisible|DeviceLocal", "Transfer|Dispatch"
  // CHECK-NEXT: hal.ex.defer_release [[TMP_BUF]]
  // CHECK: [[CMD:%.+]] = hal.command_buffer.create {{.+}}, "OneShot", "Transfer|Dispatch"
  // CHECK-NEXT: hal.command_buffer.begin [[CMD]]
  %0 = flow.ex.stream.fragment(%arg1 = %cst : index, %arg2 = %arg0 : tensor<128xf32>) -> tensor<128xf32> {
    //  CHECK-DAG: [[EXE:%.+]] = hal.executable.lookup {{.+}}, @ex0 : !hal.executable
    //  CHECK-DAG: [[EXE_LAYOUT:%.+]] = hal.executable_layout.lookup
    //      CHECK: hal.command_buffer.push_descriptor_set [[CMD]], [[EXE_LAYOUT]], set=0, bindings=[0 = (%arg0, %c0, %sz_3), 1 = (%buffer_1, %c0, %sz_4)]
    // CHECK-NEXT: hal.command_buffer.dispatch [[CMD]], [[EXE]], entry_point = 0, workgroup_xyz = [
    // CHECK-SAME:   [[C4]], [[C1]], [[C1]]
    // CHECK-SAME: ]
    //      CHECK: hal.command_buffer.execution_barrier
    %1 = flow.dispatch @ex0::@entry0[%arg1 : index](%arg2) : (tensor<128xf32>) -> tensor<128xf32>
    //      CHECK: hal.command_buffer.push_descriptor_set
    // CHECK-NEXT: hal.command_buffer.dispatch [[CMD]], {{.+}}, entry_point = 0, workgroup_xyz = [
    // CHECK-SAME:   [[C4]], [[C1]], [[C1]]
    // CHECK-SAME: ]
    //      CHECK: hal.command_buffer.execution_barrier
    %2 = flow.dispatch @ex0::@entry0[%arg1 : index](%1) : (tensor<128xf32>) -> tensor<128xf32>
    flow.return %2 : tensor<128xf32>
  }
  // CHECK: hal.command_buffer.end [[CMD]]
  // CHECK-NEXT: hal.ex.submit_and_wait {{.+}}, [[CMD]]
  // CHECK-NEXT: return [[RET_BUF]]
  return %0 : tensor<128xf32>
}

// -----

// CHECK-LABEL: @tensorUpdate
// CHECK-SAME: ([[UBUF:%.+]]:{{.+}}, [[TBUF:%.+]]:{{.+}})
func @tensorUpdate(%arg0 : tensor<1x1x10xf32>, %arg1 : tensor<5x1x10xf32>) -> tensor<5x1x10xf32> {
  // CHECK: [[C0:%.+]] = constant 0
  %c4 = constant 4 : index
  %c1 = constant 1 : index
  // CHECK: [[RET_BUF:%.+]] = hal.allocator.allocate
  // CHECK: [[CMD:%.+]] = hal.command_buffer.create
  // CHECK-NEXT: hal.command_buffer.begin [[CMD]]
  %0 = flow.ex.stream.fragment(%arg2 = %arg0 : tensor<1x1x10xf32>, %arg3 = %arg1 : tensor<5x1x10xf32>, %arg4 = %c4 : index, %arg5 = %c1 : index) -> tensor<5x1x10xf32> {
    // CHECK: [[UOFF:%.+]], [[ULEN:%.+]] = hal.allocator.compute_range {{%.+}}
    // CHECK: [[TLEN:%.+]] = hal.allocator.compute_size {{%.+}}
    // CHECK-NEXT: hal.command_buffer.copy_buffer [[CMD]], [[TBUF]], [[C0]], [[RET_BUF]], [[C0]], [[TLEN]]
    // CHECK: hal.command_buffer.execution_barrier
    // CHECK-NEXT: hal.command_buffer.copy_buffer [[CMD]], [[UBUF]], [[C0]], [[RET_BUF]], [[UOFF]], [[ULEN]]
    %1 = flow.tensor.update %arg2, %arg3[%arg4, %arg5, %arg5] : tensor<1x1x10xf32> -> tensor<5x1x10xf32>
    flow.return %1 : tensor<5x1x10xf32>
  }
  // CHECK: hal.command_buffer.end [[CMD]]
  // CHECK: return [[RET_BUF]]
  return %0 : tensor<5x1x10xf32>
}
