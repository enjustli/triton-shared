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
// CHECK: %[[SRC:.*]] = builtin.unrealized_conversion_cast %[[SRC_PTR]] : !tt.ptr<i32> to memref<*xi32>
// CHECK: %[[SRC_DESC:.*]] = memref.reinterpret_cast %[[SRC]]
// CHECK-SAME: sizes: [{{[^]]+}}], strides: [4, 1]
// CHECK-SAME: to memref<?x?xi32, strided<[4, 1]>>
// CHECK: %[[DST:.*]] = builtin.unrealized_conversion_cast %[[DST_PTR]] : !tt.ptr<i32> to memref<*xi32>
// CHECK: %[[DST_DESC:.*]] = memref.reinterpret_cast %[[DST]]
// CHECK-SAME: sizes: [{{[^]]+}}], strides: [4, 1]
// CHECK-SAME: to memref<?x?xi32, strided<[4, 1]>>
// CHECK: arith.maxsi
// CHECK: %[[LOAD_ALLOC:.*]] = memref.alloc() : memref<2x4xi32>
// CHECK: scf.if
// CHECK: memref.copy
// CHECK: %[[VAL:.*]] = bufferization.to_tensor %[[LOAD_ALLOC]] restrict writable : memref<2x4xi32> to tensor<2x4xi32>
// CHECK: scf.if
// CHECK: bufferization.materialize_in_destination
// CHECK-NOT: tt.make_tensor_descriptor
// CHECK-NOT: tt.descriptor_load
// CHECK-NOT: tt.descriptor_store
// CHECK: tt.return

// CHECK-LABEL: tt.func @descriptor_reduce(
// CHECK: %[[SUBVIEW:.*]] = memref.{{(subview|cast)}}
// CHECK-SAME: to memref<2x4xi32
// CHECK: %[[CAST:.*]] = memref.cast %[[SUBVIEW]]
// CHECK: memref.alloc() : memref<2x4xi32>
// CHECK: %[[INIT:.*]] = bufferization.to_tensor
// CHECK: linalg.reduce
// CHECK-SAME: arith.addi
// CHECK-SAME: ins(%{{[^ ]*}} : tensor<2x4xi32>)
// CHECK-SAME: outs(%[[INIT]]
// CHECK-SAME: dimensions = []
// CHECK: bufferization.materialize_in_destination
// CHECK-SAME: in writable %[[CAST]]
// CHECK-NOT: tt.descriptor_reduce
// CHECK: tt.return

// CHECK-LABEL: tt.func @descriptor_gather(
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<8x8xi32>
// CHECK: %[[GENERIC:.*]] = linalg.generic
// CHECK-SAME: ins(%{{.*}} : tensor<8xi32>)
// CHECK-SAME: outs(%[[EMPTY]] : tensor<8x8xi32>)
// CHECK:   memref.load
// CHECK:   linalg.yield
// CHECK-NOT: tt.descriptor_gather
// CHECK: tt.return %[[GENERIC]]

// CHECK-LABEL: tt.func @descriptor_scatter(
// CHECK: linalg.generic
// CHECK-SAME: ins(%{{.*}}, %{{.*}} : tensor<8xi32>, tensor<8x8xi32>)
// CHECK:   memref.store
// CHECK:   linalg.yield
// CHECK-NOT: tt.descriptor_scatter
// CHECK: tt.return

// CHECK-LABEL: tt.func public @descriptor_signature(
// CHECK-SAME: memref<*xi32>
// CHECK-SAME: i1
// CHECK-SAME: i1
// CHECK-SAME: -> (memref<*xi32>, i1, i1)
// CHECK: tt.return
// CHECK-SAME: memref<*xi32>, i1, i1
