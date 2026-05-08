// RUN: triton-shared-opt --triton-tensor-descriptor-to-memref --canonicalize --cse %s | FileCheck %s

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
    %dst_desc = tt.make_tensor_descriptor %dst_ptr, [%m, %n], [%c4_i64, %c1_i64] : <i32>, <2x4xi32>
    tt.descriptor_reduce add, %dst_desc[%c0_i32, %c0_i32], %src : !tt.tensordesc<2x4xi32>, tensor<2x4xi32>
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

  tt.func public @descriptor_signature(%desc: !tt.tensordesc<2x4xi32>) -> !tt.tensordesc<2x4xi32> {
    tt.return %desc : !tt.tensordesc<2x4xi32>
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
// CHECK: "tts.reduce"(%[[REDUCE_TPTR]]
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

// CHECK-LABEL: tt.func public @descriptor_signature(
// CHECK-SAME: !tt.ptr<i32>
// CHECK-SAME: i1
// CHECK-SAME: i1
// CHECK-SAME: -> (!tt.ptr<i32>, i1, i1)
// CHECK: tt.return
// CHECK-SAME: !tt.ptr<i32>, i1, i1
