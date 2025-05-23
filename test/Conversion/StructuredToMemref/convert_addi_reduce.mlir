// RUN: triton-shared-opt --split-input-file --triton-to-linalg-experimental  %s | FileCheck %s

module {
  tt.func public @addi(%arg0: !tt.ptr<i32>) {
    %cst_0 = arith.constant dense<0> : tensor<4096xi32>
    %63 = "tt.reduce"(%cst_0) ({
    ^bb0(%arg14: i32, %arg15: i32):
      %69 = arith.addi %arg14, %arg15 : i32
      tt.reduce.return %69 : i32
    }) {axis = 0 : i32} : (tensor<4096xi32>) -> i32
    tt.store %arg0, %63 : !tt.ptr<i32>
    tt.return
  }
}

// CHECK-LABEL:  func.func @addi
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xi32>, [[PARAM_1_:%.+]]: i32, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32) {
// CHECK-DAG:       %[[CST_0_:.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[VAR_0_:%.+]] = tensor.empty() : tensor<4096xi32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_1_:%.+]] = linalg.fill ins([[CST_0_1_]] : i32) outs([[VAR_0_]] : tensor<4096xi32>) -> tensor<4096xi32>
// CHECK-DAG:       [[VAR_2_:%.+]] = bufferization.alloc_tensor() : tensor<i32>
// CHECK:           [[VAR_inserted_:%.+]] = tensor.insert [[CST_0_1_]] into [[VAR_2_]][] : tensor<i32>
// CHECK:           [[VAR_reduced_:%.+]] = linalg.reduce ins([[VAR_1_]] : tensor<4096xi32>) outs([[VAR_inserted_]] : tensor<i32>) dimensions = [0]
// CHECK:             ([[in_:.+]]: i32, [[init_:.+]]: i32) {
// CHECK:               [[VAR_3_:%.+]] = arith.addi [[in_]], [[init_]] : i32
// CHECK:               linalg.yield [[VAR_3_]] : i32
// CHECK:             }
// CHECK-DAG:       [[VAR_extracted_:%.+]] = tensor.extract [[VAR_reduced_]][] : tensor<i32>
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: [%[[CST_0_]]], sizes: [1], strides: [1] : memref<*xi32> to memref<1xi32, strided<[1], offset: ?>>
// CHECK:           memref.store [[VAR_extracted_]], [[VAR_reinterpret_cast_]][%[[CST_0_]]] : memref<1xi32, strided<[1], offset: ?>>
// CHECK:           return
// CHECK:         }
