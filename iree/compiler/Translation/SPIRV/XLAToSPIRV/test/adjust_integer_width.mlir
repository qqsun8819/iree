// RUN: iree-opt -iree-spirv-adjust-integer-width -verify-diagnostics -o - %s | IreeFileCheck %s

module{
  spv.module Logical GLSL450 {
    spv.globalVariable @globalInvocationID built_in("GlobalInvocationId") : !spv.ptr<vector<3xi32>, Input>
    // CHECK: spv.globalVariable @constant_arg_0 bind(0, 0) : !spv.ptr<!spv.struct<i32 [0]>, StorageBuffer>
    // CHECK: spv.globalVariable @constant_arg_1 bind(0, 1) : !spv.ptr<!spv.struct<i32 [0]>, StorageBuffer>
    spv.globalVariable @constant_arg_0 bind(0, 0) : !spv.ptr<!spv.struct<i64 [0]>, StorageBuffer>
    spv.globalVariable @constant_arg_1 bind(0, 1) : !spv.ptr<!spv.struct<i64 [0]>, StorageBuffer>
    spv.func @foo_i64(%arg0 : i64, %arg1 : i64) -> () "None" {
      // CHECK: spv._address_of {{.*}} : !spv.ptr<!spv.struct<i32 [0]>, StorageBuffer>
      // CHECK: spv.AccessChain {{.*}} : !spv.ptr<!spv.struct<i32 [0]>, StorageBuffer>
      // CHECK: spv.Load "StorageBuffer" %{{.*}} : i32
      // CHECK: spv._address_of {{.*}} : !spv.ptr<!spv.struct<i32 [0]>, StorageBuffer>
      // CHECK: spv.AccessChain {{.*}} : !spv.ptr<!spv.struct<i32 [0]>, StorageBuffer>
      // CHECK: spv.Store "StorageBuffer" %{{.*}} %{{.*}} : i32
      %0 = spv._address_of @constant_arg_0 : !spv.ptr<!spv.struct<i64 [0]>, StorageBuffer>
      %1 = spv.constant 0 : i32
      %2 = spv.AccessChain %0[%1] : !spv.ptr<!spv.struct<i64 [0]>, StorageBuffer>
      %3 = spv.Load "StorageBuffer" %2 : i64
      %4 = spv._address_of @constant_arg_1 : !spv.ptr<!spv.struct<i64 [0]>, StorageBuffer>
      %5 = spv.constant 0 : i32
      %6 = spv.AccessChain %4[%5] : !spv.ptr<!spv.struct<i64 [0]>, StorageBuffer>
      spv.Store "StorageBuffer" %6, %3 : i64
      spv.Return
    }
  }

  spv.module Logical GLSL450 {
    spv.globalVariable @globalInvocationID built_in("GlobalInvocationId") : !spv.ptr<vector<3xi32>, Input>
    // CHECK: spv.globalVariable @constant_arg_0 bind(0, 0) : !spv.ptr<!spv.struct<i32 [0]>, StorageBuffer>
    // CHECK: spv.globalVariable @constant_arg_1 bind(0, 1) : !spv.ptr<!spv.struct<i32 [0]>, StorageBuffer>
    spv.globalVariable @constant_arg_0 bind(0, 0) : !spv.ptr<!spv.struct<i16 [0]>, StorageBuffer>
    spv.globalVariable @constant_arg_1 bind(0, 1) : !spv.ptr<!spv.struct<i16 [0]>, StorageBuffer>
    spv.func @foo_i16(%arg0 : i16, %arg1 : i16) -> () "None" {
      // CHECK: spv._address_of {{.*}} : !spv.ptr<!spv.struct<i32 [0]>, StorageBuffer>
      // CHECK: spv.AccessChain {{.*}} : !spv.ptr<!spv.struct<i32 [0]>, StorageBuffer>
      // CHECK: spv.Load "StorageBuffer" %{{.*}} : i32
      // CHECK: spv.AtomicAnd
      // CHECK: spv.AtomicOr
      %0 = spv._address_of @constant_arg_0 : !spv.ptr<!spv.struct<i16 [0]>, StorageBuffer>
      %1 = spv.constant 0 : i32
      %2 = spv.AccessChain %0[%1] : !spv.ptr<!spv.struct<i16 [0]>, StorageBuffer>
      %3 = spv.Load "StorageBuffer" %2 : i16
      %4 = spv._address_of @constant_arg_1 : !spv.ptr<!spv.struct<i16 [0]>, StorageBuffer>
      %5 = spv.constant 0 : i32
      %6 = spv.AccessChain %4[%5] : !spv.ptr<!spv.struct<i16 [0]>, StorageBuffer>
      spv.Store "StorageBuffer" %6, %3 : i16
      spv.Return
    }
  }

  spv.module Logical GLSL450 {
    spv.globalVariable @globalInvocationID built_in("GlobalInvocationId") : !spv.ptr<vector<3xi32>, Input>
    // CHECK: spv.globalVariable @constant_arg_0 bind(0, 0) : !spv.ptr<!spv.struct<i32 [0]>, StorageBuffer>
    // CHECK: spv.globalVariable @constant_arg_1 bind(0, 1) : !spv.ptr<!spv.struct<i32 [0]>, StorageBuffer>
    spv.globalVariable @constant_arg_0 bind(0, 0) : !spv.ptr<!spv.struct<i8 [0]>, StorageBuffer>
    spv.globalVariable @constant_arg_1 bind(0, 1) : !spv.ptr<!spv.struct<i8 [0]>, StorageBuffer>
    spv.func @foo_i8(%arg0 : i8, %arg1 : i8) -> () "None" {
      // CHECK: spv._address_of {{.*}} : !spv.ptr<!spv.struct<i32 [0]>, StorageBuffer>
      // CHECK: spv.AccessChain {{.*}} : !spv.ptr<!spv.struct<i32 [0]>, StorageBuffer>
      // CHECK: spv.Load "StorageBuffer" %{{.*}} : i32
      // CHECK: spv.AtomicAnd
      // CHECK: spv.AtomicOr
      %0 = spv._address_of @constant_arg_0 : !spv.ptr<!spv.struct<i8 [0]>, StorageBuffer>
      %1 = spv.constant 0 : i32
      %2 = spv.AccessChain %0[%1] : !spv.ptr<!spv.struct<i8 [0]>, StorageBuffer>
      %3 = spv.Load "StorageBuffer" %2 : i8
      %4 = spv._address_of @constant_arg_1 : !spv.ptr<!spv.struct<i8 [0]>, StorageBuffer>
      %5 = spv.constant 0 : i32
      %6 = spv.AccessChain %4[%5] : !spv.ptr<!spv.struct<i8 [0]>, StorageBuffer>
      spv.Store "StorageBuffer" %6, %3 : i8
      spv.Return
    }
  }

  spv.module Logical GLSL450 {
    spv.globalVariable @globalInvocationID built_in("GlobalInvocationId") : !spv.ptr<vector<3xi32>, Input>
    // CHECK: spv.globalVariable @constant_arg_0 bind(0, 0) : !spv.ptr<!spv.struct<i32 [0]>, StorageBuffer>
    // CHECK: spv.globalVariable @constant_arg_1 bind(0, 1) : !spv.ptr<!spv.struct<i32 [0]>, StorageBuffer>
    spv.globalVariable @constant_arg_0 bind(0, 0) : !spv.ptr<!spv.struct<i1 [0]>, StorageBuffer>
    spv.globalVariable @constant_arg_1 bind(0, 1) : !spv.ptr<!spv.struct<i1 [0]>, StorageBuffer>
    spv.func @foo_i1(%arg0 : i1, %arg1 : i1) -> () "None" {
      // CHECK: spv._address_of {{.*}} : !spv.ptr<!spv.struct<i32 [0]>, StorageBuffer>
      // CHECK: spv.AccessChain {{.*}} : !spv.ptr<!spv.struct<i32 [0]>, StorageBuffer>
      // CHECK: spv.Load "StorageBuffer" %{{.*}} : i32
      // CHECK-NEXT: spv.BitwiseAnd
      // CHECK-NEXT: spv.INotEqual {{.*}} : i32
      // CHECK: spv.Select {{.*}} : i1, i32
      // CHECK: spv.AtomicAnd
      // CHECK: spv.AtomicOr
      %0 = spv._address_of @constant_arg_0 : !spv.ptr<!spv.struct<i1 [0]>, StorageBuffer>
      %1 = spv.constant 0 : i32
      %2 = spv.AccessChain %0[%1] : !spv.ptr<!spv.struct<i1 [0]>, StorageBuffer>
      %3 = spv.Load "StorageBuffer" %2 : i1
      %4 = spv._address_of @constant_arg_1 : !spv.ptr<!spv.struct<i1 [0]>, StorageBuffer>
      %5 = spv.constant 0 : i32
      %6 = spv.AccessChain %4[%5] : !spv.ptr<!spv.struct<i1 [0]>, StorageBuffer>
      spv.Store "StorageBuffer" %6, %3 : i1
      spv.Return
    }
  }

  spv.module Logical GLSL450 {
    spv.globalVariable @globalInvocationID built_in("GlobalInvocationId") : !spv.ptr<vector<3xi32>, Input>
    spv.globalVariable @arg_0 bind(0, 0) : !spv.ptr<!spv.struct<i64 [0]>, StorageBuffer>
    spv.globalVariable @arg_1 bind(0, 1) : !spv.ptr<!spv.struct<i64 [0]>, StorageBuffer>
    spv.func @add(%arg0: i64, %arg1: i64) -> () "None" {
      %0 = spv._address_of @arg_0 : !spv.ptr<!spv.struct<i64 [0]>, StorageBuffer>
      %1 = spv.constant 0 : i32
      %2 = spv.AccessChain %0[%1] : !spv.ptr<!spv.struct<i64 [0]>, StorageBuffer>
      %3 = spv.Load "StorageBuffer" %2 : i64
      // CHECK: spv.IAdd {{.*}} : i32
      %4 = spv.IAdd %3, %3 : i64
      %5 = spv._address_of @arg_1 : !spv.ptr<!spv.struct<i64 [0]>, StorageBuffer>
      %6 = spv.AccessChain %5[%1] : !spv.ptr<!spv.struct<i64 [0]>, StorageBuffer>
      spv.Store "StorageBuffer" %6, %4 : i64
      spv.Return
    }

    spv.func @sub(%arg0: i64, %arg1: i64) -> () "None" {
      %0 = spv._address_of @arg_0 : !spv.ptr<!spv.struct<i64 [0]>, StorageBuffer>
      %1 = spv.constant 0 : i32
      %2 = spv.AccessChain %0[%1] : !spv.ptr<!spv.struct<i64 [0]>, StorageBuffer>
      %3 = spv.Load "StorageBuffer" %2 : i64
      %cst = spv.constant 1337 : i64
      // CHECK: spv.ISub {{.*}} : i32
      %4 = spv.ISub %3, %cst : i64
      %5 = spv._address_of @arg_1 : !spv.ptr<!spv.struct<i64 [0]>, StorageBuffer>
      %6 = spv.AccessChain %5[%1] : !spv.ptr<!spv.struct<i64 [0]>, StorageBuffer>
      spv.Store "StorageBuffer" %6, %4 : i64
      spv.Return
    }

    spv.func @mul(%arg0: i64, %arg1: i64) -> () "None" {
      %0 = spv._address_of @arg_0 : !spv.ptr<!spv.struct<i64 [0]>, StorageBuffer>
      %1 = spv.constant 0 : i32
      %2 = spv.AccessChain %0[%1] : !spv.ptr<!spv.struct<i64 [0]>, StorageBuffer>
      %3 = spv.Load "StorageBuffer" %2 : i64
      // CHECK: spv.IMul {{.*}} : i32
      %4 = spv.IMul %3, %3 : i64
      %5 = spv._address_of @arg_1 : !spv.ptr<!spv.struct<i64 [0]>, StorageBuffer>
      %6 = spv.AccessChain %5[%1] : !spv.ptr<!spv.struct<i64 [0]>, StorageBuffer>
      spv.Store "StorageBuffer" %6, %4 : i64
      spv.Return
    }

    spv.func @sdiv(%arg0: i64, %arg1: i64) -> () "None" {
      %0 = spv._address_of @arg_0 : !spv.ptr<!spv.struct<i64 [0]>, StorageBuffer>
      %1 = spv.constant 0 : i32
      %2 = spv.AccessChain %0[%1] : !spv.ptr<!spv.struct<i64 [0]>, StorageBuffer>
      %3 = spv.Load "StorageBuffer" %2 : i64
      // CHECK: spv.SDiv {{.*}} : i32
      %4 = spv.SDiv %3, %3 : i64
      %5 = spv._address_of @arg_1 : !spv.ptr<!spv.struct<i64 [0]>, StorageBuffer>
      %6 = spv.AccessChain %5[%1] : !spv.ptr<!spv.struct<i64 [0]>, StorageBuffer>
      spv.Store "StorageBuffer" %6, %4 : i64
      spv.Return
    }

    spv.func @smod(%arg0: i64, %arg1: i64) -> () "None" {
      %0 = spv._address_of @arg_0 : !spv.ptr<!spv.struct<i64 [0]>, StorageBuffer>
      %1 = spv.constant 0 : i32
      %2 = spv.AccessChain %0[%1] : !spv.ptr<!spv.struct<i64 [0]>, StorageBuffer>
      %3 = spv.Load "StorageBuffer" %2 : i64
      // CHECK: spv.SMod {{.*}} : i32
      %4 = spv.SMod %3, %3 : i64
      %5 = spv._address_of @arg_1 : !spv.ptr<!spv.struct<i64 [0]>, StorageBuffer>
      %6 = spv.AccessChain %5[%1] : !spv.ptr<!spv.struct<i64 [0]>, StorageBuffer>
      spv.Store "StorageBuffer" %6, %4 : i64
      spv.Return
    }

    spv.func @srem(%arg0: i64, %arg1: i64) -> () "None" {
      %0 = spv._address_of @arg_0 : !spv.ptr<!spv.struct<i64 [0]>, StorageBuffer>
      %1 = spv.constant 0 : i32
      %2 = spv.AccessChain %0[%1] : !spv.ptr<!spv.struct<i64 [0]>, StorageBuffer>
      %3 = spv.Load "StorageBuffer" %2 : i64
      // CHECK: spv.SRem {{.*}} : i32
      %4 = spv.SRem %3, %3 : i64
      %5 = spv._address_of @arg_1 : !spv.ptr<!spv.struct<i64 [0]>, StorageBuffer>
      %6 = spv.AccessChain %5[%1] : !spv.ptr<!spv.struct<i64 [0]>, StorageBuffer>
      spv.Store "StorageBuffer" %6, %4 : i64
      spv.Return
    }

    spv.func @udiv(%arg0: i64, %arg1: i64) -> () "None" {
      %0 = spv._address_of @arg_0 : !spv.ptr<!spv.struct<i64 [0]>, StorageBuffer>
      %1 = spv.constant 0 : i32
      %2 = spv.AccessChain %0[%1] : !spv.ptr<!spv.struct<i64 [0]>, StorageBuffer>
      %3 = spv.Load "StorageBuffer" %2 : i64
      // CHECK: spv.UDiv {{.*}} : i32
      %4 = spv.UDiv %3, %3 : i64
      %5 = spv._address_of @arg_1 : !spv.ptr<!spv.struct<i64 [0]>, StorageBuffer>
      %6 = spv.AccessChain %5[%1] : !spv.ptr<!spv.struct<i64 [0]>, StorageBuffer>
      spv.Store "StorageBuffer" %6, %4 : i64
      spv.Return
    }

    spv.func @umod(%arg0: i64, %arg1: i64) -> () "None" {
      %0 = spv._address_of @arg_0 : !spv.ptr<!spv.struct<i64 [0]>, StorageBuffer>
      %1 = spv.constant 0 : i32
      %2 = spv.AccessChain %0[%1] : !spv.ptr<!spv.struct<i64 [0]>, StorageBuffer>
      %3 = spv.Load "StorageBuffer" %2 : i64
      // CHECK: spv.UMod {{.*}} : i32
      %4 = spv.UMod %3, %3 : i64
      %5 = spv._address_of @arg_1 : !spv.ptr<!spv.struct<i64 [0]>, StorageBuffer>
      %6 = spv.AccessChain %5[%1] : !spv.ptr<!spv.struct<i64 [0]>, StorageBuffer>
      spv.Store "StorageBuffer" %6, %4 : i64
      spv.Return
    }

    spv.func @abs(%arg0: i64, %arg1: i64) -> () "None" {
      %0 = spv._address_of @arg_0 : !spv.ptr<!spv.struct<i64 [0]>, StorageBuffer>
      %1 = spv.constant 0 : i32
      %2 = spv.AccessChain %0[%1] : !spv.ptr<!spv.struct<i64 [0]>, StorageBuffer>
      %3 = spv.Load "StorageBuffer" %2 : i64
      // CHECK: spv.GLSL.SAbs {{.*}} : i32
      %4 = spv.GLSL.SAbs %3 : i64
      %5 = spv._address_of @arg_1 : !spv.ptr<!spv.struct<i64 [0]>, StorageBuffer>
      %6 = spv.AccessChain %5[%1] : !spv.ptr<!spv.struct<i64 [0]>, StorageBuffer>
      spv.Store "StorageBuffer" %6, %4 : i64
      spv.Return
    }

    spv.func @smax(%arg0: i64, %arg1: i64) -> () "None" {
      %0 = spv._address_of @arg_0 : !spv.ptr<!spv.struct<i64 [0]>, StorageBuffer>
      %1 = spv.constant 0 : i32
      %2 = spv.AccessChain %0[%1] : !spv.ptr<!spv.struct<i64 [0]>, StorageBuffer>
      %3 = spv.Load "StorageBuffer" %2 : i64
      // CHECK: spv.GLSL.SMax {{.*}} : i32
      %4 = spv.GLSL.SMax %3, %3 : i64
      %5 = spv._address_of @arg_1 : !spv.ptr<!spv.struct<i64 [0]>, StorageBuffer>
      %6 = spv.AccessChain %5[%1] : !spv.ptr<!spv.struct<i64 [0]>, StorageBuffer>
      spv.Store "StorageBuffer" %6, %4 : i64
      spv.Return
    }

    spv.func @smin(%arg0: i64, %arg1: i64) -> () "None" {
      %0 = spv._address_of @arg_0 : !spv.ptr<!spv.struct<i64 [0]>, StorageBuffer>
      %1 = spv.constant 0 : i32
      %2 = spv.AccessChain %0[%1] : !spv.ptr<!spv.struct<i64 [0]>, StorageBuffer>
      %3 = spv.Load "StorageBuffer" %2 : i64
      // CHECK: spv.GLSL.SMin {{.*}} : i32
      %4 = spv.GLSL.SMin %3, %3 : i64
      %5 = spv._address_of @arg_1 : !spv.ptr<!spv.struct<i64 [0]>, StorageBuffer>
      %6 = spv.AccessChain %5[%1] : !spv.ptr<!spv.struct<i64 [0]>, StorageBuffer>
      spv.Store "StorageBuffer" %6, %4 : i64
      spv.Return
    }

    spv.func @sign(%arg0: i64, %arg1: i64) -> () "None" {
      %0 = spv._address_of @arg_0 : !spv.ptr<!spv.struct<i64 [0]>, StorageBuffer>
      %1 = spv.constant 0 : i32
      %2 = spv.AccessChain %0[%1] : !spv.ptr<!spv.struct<i64 [0]>, StorageBuffer>
      %3 = spv.Load "StorageBuffer" %2 : i64
      // CHECK: spv.GLSL.SSign {{.*}} : i32
      %4 = spv.GLSL.SSign %3 : i64
      %5 = spv._address_of @arg_1 : !spv.ptr<!spv.struct<i64 [0]>, StorageBuffer>
      %6 = spv.AccessChain %5[%1] : !spv.ptr<!spv.struct<i64 [0]>, StorageBuffer>
      spv.Store "StorageBuffer" %6, %4 : i64
      spv.Return
    }

    spv.func @constant_i64(%arg1: i64) -> () "None" {
      // CHECK: spv.constant 1337 : i32
      %0 = spv.constant 1337 : i64
      %1 = spv.constant 0 : i32
      %2 = spv._address_of @arg_1 : !spv.ptr<!spv.struct<i64 [0]>, StorageBuffer>
      %3 = spv.AccessChain %2[%1] : !spv.ptr<!spv.struct<i64 [0]>, StorageBuffer>
      spv.Store "StorageBuffer" %3, %0 : i64
      spv.Return
    }
  }

  spv.module Logical GLSL450 {
    spv.globalVariable @globalInvocationID built_in("GlobalInvocationId") : !spv.ptr<vector<3xi32>, Input>
    spv.globalVariable @constant_arg_0 bind(0, 0) : !spv.ptr<!spv.struct<i1 [0]>, StorageBuffer>
    spv.globalVariable @constant_arg_1 bind(0, 1) : !spv.ptr<!spv.struct<i8 [0]>, StorageBuffer>
    spv.func @select(%arg0 : i1, %arg1 : i8) -> () "None" {
      %0 = spv._address_of @constant_arg_0 : !spv.ptr<!spv.struct<i1 [0]>, StorageBuffer>
      %1 = spv.constant 0 : i32
      %2 = spv.AccessChain %0[%1] : !spv.ptr<!spv.struct<i1 [0]>, StorageBuffer>
      %3 = spv.Load "StorageBuffer" %2 : i1
      %4 = spv.constant 0 : i8
      %5 = spv.constant 1 : i8
      // CHECK: spv.Select {{.*}} : i1, i32
      %6 = spv.Select %3, %5, %4 : i1, i8
      %7 = spv._address_of @constant_arg_1 : !spv.ptr<!spv.struct<i8 [0]>, StorageBuffer>
      %8 = spv.AccessChain %7[%1] : !spv.ptr<!spv.struct<i8 [0]>, StorageBuffer>
      spv.Store "StorageBuffer" %8, %6 : i8
      spv.Return
    }
  }
}
