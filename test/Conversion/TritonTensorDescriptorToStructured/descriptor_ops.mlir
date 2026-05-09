// RUN: triton-shared-opt --triton-tensor-descriptor-to-structured --canonicalize --cse %s | FileCheck %s

module {
  tt.func @descriptor_ops(%src_ptr: !tt.ptr<i32>, %dst_ptr: !tt.ptr<i32>,
                          %m: i32, %n: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c4_i64 = arith.constant 4 : i64
    %c1_i64 = arith.constant 1 : i64

    %src_desc = tt.make_tensor_descriptor %src_ptr, [%m, %n], [%c4_i64, %c1_i64] : <i32>, <2x4xi32>
    %dst_desc = tt.make_tensor_descriptor %dst_ptr, [%m, %n], [%c4_i64, %c1_i64] : <i32>, <2x4xi32>
    %val = tt.descriptor_load %src_desc[%c0_i32, %c0_i32] : !tt.tensordesc<2x4xi32> -> tensor<2x4xi32>
    tt.descriptor_store %dst_desc[%c0_i32, %c0_i32], %val : !tt.tensordesc<2x4xi32>, tensor<2x4xi32>
    tt.return
  }

  tt.func @descriptor_reduce(%dst_ptr: !tt.ptr<i32>, %src: tensor<2x4xi32>,
                             %m: i32, %n: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c4_i64 = arith.constant 4 : i64
    %c1_i64 = arith.constant 1 : i64
    %dst_desc = tt.make_tensor_descriptor %dst_ptr, [%m, %n], [%c4_i64, %c1_i64] : <i32>, <2x4xui32>
    tt.descriptor_reduce add, %dst_desc[%c0_i32, %c0_i32], %src : !tt.tensordesc<2x4xui32>, tensor<2x4xi32>
    tt.return
  }

  tt.func @descriptor_gather(%src_ptr: !tt.ptr<i32>, %x_offsets: tensor<8xi32>,
                             %m: i32, %n: i32, %y: i32) -> tensor<8x8xi32> {
    %c4_i64 = arith.constant 4 : i64
    %c1_i64 = arith.constant 1 : i64
    %src_desc = tt.make_tensor_descriptor %src_ptr, [%m, %n], [%c4_i64, %c1_i64] : <i32>, <1x8xi32>
    %val = tt.descriptor_gather %src_desc[%x_offsets, %y] : (!tt.tensordesc<1x8xi32>, tensor<8xi32>, i32) -> tensor<8x8xi32>
    tt.return %val : tensor<8x8xi32>
  }

  tt.func @descriptor_scatter(%dst_ptr: !tt.ptr<i32>, %x_offsets: tensor<8xi32>,
                              %src: tensor<8x8xi32>, %m: i32, %n: i32,
                              %y: i32) {
    %c4_i64 = arith.constant 4 : i64
    %c1_i64 = arith.constant 1 : i64
    %dst_desc = tt.make_tensor_descriptor %dst_ptr, [%m, %n], [%c4_i64, %c1_i64] : <i32>, <1x8xi32>
    tt.descriptor_scatter %dst_desc[%x_offsets, %y], %src : !tt.tensordesc<1x8xi32>, tensor<8xi32>, i32, tensor<8x8xi32>
    tt.return
  }

  tt.func @descriptor_arg_load(%desc: !tt.tensordesc<2x4xi32>,
                               %i: i32, %j: i32) -> tensor<2x4xi32> {
    %val = tt.descriptor_load %desc[%i, %j] : !tt.tensordesc<2x4xi32> -> tensor<2x4xi32>
    tt.return %val : tensor<2x4xi32>
  }

  tt.func public @descriptor_signature(%desc: !tt.tensordesc<2x4xi32>) -> !tt.tensordesc<2x4xi32> {
    tt.return %desc : !tt.tensordesc<2x4xi32>
  }
  
  tt.func public @kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c32_i32 = arith.constant 32 : i32
    %c8_i32 = arith.constant 8 : i32
    %0:5 = tt.call @test_tensor_descriptor.tensor_descriptor_return_helper__Pfp32_i32_i32_c8_c32(%arg1, %arg2, %arg3) : (!tt.ptr<f32>, i32, i32) -> (!tt.tensordesc<8x32xf32>, i32, i32, i64, i64)
    %1:5 = tt.call @test_tensor_descriptor.tensor_descriptor_return_helper__Pfp32_i32_i32_c8_c32(%arg0, %arg2, %arg3) : (!tt.ptr<f32>, i32, i32) -> (!tt.tensordesc<8x32xf32>, i32, i32, i64, i64)
    %2 = tt.get_program_id x : i32
    %3 = arith.muli %2, %c8_i32 : i32
    %4 = tt.get_program_id y : i32
    %5 = arith.muli %4, %c32_i32 : i32
    %6 = tt.descriptor_load %0#0[%3, %5] : !tt.tensordesc<8x32xf32> -> tensor<8x32xf32>
    %7 = math.absf %6 : tensor<8x32xf32>
    tt.descriptor_store %1#0[%3, %5], %7 : !tt.tensordesc<8x32xf32>, tensor<8x32xf32>
    tt.return
  }
  tt.func private @test_tensor_descriptor.tensor_descriptor_return_helper__Pfp32_i32_i32_c8_c32(%arg0: !tt.ptr<f32>, %arg1: i32, %arg2: i32) -> (!tt.tensordesc<8x32xf32>, i32, i32, i64, i64) attributes {noinline = true} {
    %c1_i64 = arith.constant 1 : i64
    %0 = arith.extsi %arg2 : i32 to i64
    %1 = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%0, %c1_i64] : <f32>, <8x32xf32>
    tt.return %1, %arg1, %arg2, %0, %c1_i64 : !tt.tensordesc<8x32xf32>, i32, i32, i64, i64
  }
}

// CHECK-LABEL: tt.func @descriptor_ops(
// CHECK-SAME: %[[SRC_PTR:.*]]: !tt.ptr<i32>, %[[DST_PTR:.*]]: !tt.ptr<i32>, %[[M:.*]]: i32, %[[N:.*]]: i32
// CHECK: arith.maxsi
// CHECK: %[[SRC_TPTR:.*]] = tts.make_tptr %[[SRC_PTR]]
// CHECK-SAME: to sizes: [2, 4]
// CHECK-SAME: shape: [0, 0]
// CHECK: %[[VAL:.*]] = "tts.load"(%[[SRC_TPTR]]
// CHECK-SAME: tensor<2x4x!tt.ptr<i32>>
// CHECK: %[[DST_TPTR:.*]] = tts.make_tptr %[[DST_PTR]]
// CHECK-SAME: to sizes: [2, 4]
// CHECK: "tts.store"(%[[DST_TPTR]], %[[VAL]]
// CHECK-NOT: tt.make_tensor_descriptor
// CHECK-NOT: tt.descriptor_load
// CHECK-NOT: tt.descriptor_store
// CHECK: tt.return

// CHECK-LABEL: tt.func @descriptor_reduce(
// CHECK: %[[REDUCE_TPTR:.*]] = tts.make_tptr
// CHECK-SAME: to sizes: [2, 4]
// CHECK-SAME: shape: [0, 0]
// CHECK: "tts.reduce"(%[[REDUCE_TPTR]], {{.*}}) <{is_unsigned = true, kind = 1 : i32
// CHECK-SAME: static_mask_dims = array<i64: -9223372036854775808, -9223372036854775808>
// CHECK-SAME: }> : (tensor<2x4x!tt.ptr<i32>>, tensor<2x4xi32>, index, index) -> ()
// CHECK-NOT: tt.descriptor_reduce
// CHECK: tt.return

// CHECK-LABEL: tt.func @descriptor_gather(
// CHECK: arith.cmpi sge
// CHECK: tt.splat
// CHECK: arith.cmpi slt
// CHECK: arith.andi
// CHECK: %[[GATHER_PTR:.*]] = tts.make_gather_scatter_tptr
// CHECK-SAME: gather_scatter_dim: 0
// CHECK-SAME: gather_scatter_offset:
// CHECK-SAME: gather_scatter_mask:
// CHECK: %[[GATHER:.*]] = "tts.load"(%[[GATHER_PTR]]
// CHECK-NOT: tt.descriptor_gather
// CHECK: tt.return %[[GATHER]]

// CHECK-LABEL: tt.func @descriptor_scatter(
// CHECK: arith.cmpi sge
// CHECK: tt.splat
// CHECK: arith.cmpi slt
// CHECK: arith.andi
// CHECK: %[[SCATTER_PTR:.*]] = tts.make_gather_scatter_tptr
// CHECK-SAME: gather_scatter_dim: 0
// CHECK-SAME: gather_scatter_offset:
// CHECK-SAME: gather_scatter_mask:
// CHECK: "tts.store"(%[[SCATTER_PTR]]
// CHECK-NOT: tt.descriptor_scatter
// CHECK: tt.return

// CHECK-LABEL: tt.func @descriptor_arg_load(
// CHECK-SAME: %[[DESC_BASE:[^:]+]]: !tt.ptr<i32>, %[[SHAPE0:[^:]+]]: i64, %[[SHAPE1:[^:]+]]: i64, %[[STRIDE0:[^:]+]]: i64, %[[STRIDE1:[^:]+]]: i64, %[[PADDING:[^:]+]]: i1, %[[TF32:[^:]+]]: i1
// CHECK: %[[DESC_TPTR:.*]] = tts.make_tptr %[[DESC_BASE]]
// CHECK-SAME: to sizes: [2, 4]
// CHECK-SAME: strides: [
// CHECK-SAME: shape: [
// CHECK-SAME: : <i32> to tensor<2x4x!tt.ptr<i32>>
// CHECK: "tts.load"(%[[DESC_TPTR]]
// CHECK-NOT: tt.descriptor_load
// CHECK: tt.return

// CHECK-LABEL: tt.func public @descriptor_signature(
// CHECK-SAME: !tt.ptr<i32>
// CHECK-SAME: i64
// CHECK-SAME: i64
// CHECK-SAME: i64
// CHECK-SAME: i64
// CHECK-SAME: i1
// CHECK-SAME: i1
// CHECK-SAME: -> (!tt.ptr<i32>, i64, i64, i64, i64, i1, i1)
// CHECK: tt.return
// CHECK-SAME: !tt.ptr<i32>, i64, i64, i64, i64, i1, i1

// CHECK-LABEL: tt.func public @kernel(
// CHECK: %[[SRC_DESC:.*]]:11 = tt.call @test_tensor_descriptor.tensor_descriptor_return_helper__Pfp32_i32_i32_c8_c32
// CHECK-SAME: -> (!tt.ptr<f32>, i64, i64, i64, i64, i1, i1, i32, i32, i64, i64)
// CHECK: %[[DST_DESC:.*]]:11 = tt.call @test_tensor_descriptor.tensor_descriptor_return_helper__Pfp32_i32_i32_c8_c32
// CHECK-SAME: -> (!tt.ptr<f32>, i64, i64, i64, i64, i1, i1, i32, i32, i64, i64)
// CHECK: %[[SRC_TPTR:.*]] = tts.make_tptr %[[SRC_DESC]]#0
// CHECK-SAME: to sizes: [8, 32]
// CHECK-SAME: strides:
// CHECK-SAME: offsets:
// CHECK-SAME: shape: [0, 0]
// CHECK-SAME: : <f32> to tensor<8x32x!tt.ptr<f32>>
// CHECK: %[[LOAD:.*]] = "tts.load"(%[[SRC_TPTR]]
// CHECK-SAME: tensor<8x32x!tt.ptr<f32>>
// CHECK: %[[ABS:.*]] = math.absf
// CHECK: %[[DST_TPTR:.*]] = tts.make_tptr %[[DST_DESC]]#0
// CHECK-SAME: to sizes: [8, 32]
// CHECK: "tts.store"(%[[DST_TPTR]], %[[ABS]]
// CHECK-NOT: tt.descriptor_load
// CHECK-NOT: tt.descriptor_store
// CHECK: tt.return

// CHECK-LABEL: tt.func private @test_tensor_descriptor.tensor_descriptor_return_helper__Pfp32_i32_i32_c8_c32(
// CHECK-SAME: %[[BASE:[^:]+]]: !tt.ptr<f32>, %[[M:[^:]+]]: i32, %[[N:[^:]+]]: i32
// CHECK-SAME: -> (!tt.ptr<f32>, i64, i64, i64, i64, i1, i1, i32, i32, i64, i64)
// CHECK: %[[FALSE:.*]] = arith.constant false
// CHECK: %[[C1_I64:.*]] = arith.constant 1 : i64
// CHECK: %[[N_I64:.*]] = arith.extsi %[[N]] : i32 to i64
// CHECK: %[[M_I64:.*]] = arith.extsi %[[M]] : i32 to i64
// CHECK-NOT: tt.make_tensor_descriptor
// CHECK: tt.return %[[BASE]], %[[M_I64]], %[[N_I64]], %[[N_I64]], %[[C1_I64]], %[[FALSE]], %[[FALSE]], %[[M]], %[[N]], %[[N_I64]], %[[C1_I64]]
// CHECK-SAME: !tt.ptr<f32>, i64, i64, i64, i64, i1, i1, i32, i32, i64, i64
