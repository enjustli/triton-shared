// RUN: triton-shared-opt --structured-to-memref --canonicalize --cse %s | FileCheck %s

module {
  tt.func public @reduce_add(%arg0: !tt.ptr<i32>, %arg1: tensor<2x4xi32>) {
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = tts.make_tptr %arg0 to sizes: [2, 4], strides: [%c4, %c1], offsets: [%c0, %c0], shape: [0, 0], order: [] : <i32> to tensor<2x4x!tt.ptr<i32>>
    "tts.reduce"(%0, %arg1) <{kind = 1 : i32, static_mask_dims = array<i64>}> : (tensor<2x4x!tt.ptr<i32>>, tensor<2x4xi32>) -> ()
    tt.return
  }
}

// CHECK-LABEL: tt.func public @reduce_add(
// CHECK-SAME: %[[PTR:.*]]: !tt.ptr<i32>, %[[SRC:.*]]: tensor<2x4xi32>
// CHECK: %[[BASE:.*]] = builtin.unrealized_conversion_cast %[[PTR]] : !tt.ptr<i32> to memref<*xi32>
// CHECK: %[[DST:.*]] = memref.reinterpret_cast %[[BASE]]
// CHECK-SAME: offset: [0], sizes: [2, 4], strides: [4, 1]
// CHECK: %[[DST_CAST:.*]] = memref.cast %[[DST]]
// CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<2x4xi32>
// CHECK: memref.copy %[[DST]], %[[ALLOC]]
// CHECK: %[[INIT:.*]] = bufferization.to_tensor %[[ALLOC]] restrict writable : memref<2x4xi32> to tensor<2x4xi32>
// CHECK: %[[REDUCED:.*]] = linalg.reduce
// CHECK-SAME: arith.addi
// CHECK-SAME: ins(%[[SRC]] : tensor<2x4xi32>)
// CHECK-SAME: outs(%[[INIT]] : tensor<2x4xi32>)
// CHECK-SAME: dimensions = []
// CHECK: bufferization.materialize_in_destination %[[REDUCED]] in writable %[[DST_CAST]]
// CHECK-NOT: tts.reduce
// CHECK: tt.return
