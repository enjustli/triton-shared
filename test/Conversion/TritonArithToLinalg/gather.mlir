// RUN: triton-shared-opt --triton-arith-to-linalg --split-input-file %s | FileCheck %s
module {
  tt.func public @gather_test_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<4> : tensor<8x1xi32>
    %cst_0 = arith.constant dense<4> : tensor<4x1xi32>
    %0 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %2 = arith.muli %1, %cst_0 : tensor<4x1xi32>
    %3 = tt.expand_dims %0 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %4 = tt.broadcast %2 : tensor<4x1xi32> -> tensor<4x4xi32>
    %5 = tt.broadcast %3 : tensor<1x4xi32> -> tensor<4x4xi32>
    %6 = arith.addi %4, %5 : tensor<4x4xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x4x!tt.ptr<f32>>
    %8 = tt.addptr %7, %6 : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
    %9 = tt.load %8 : tensor<4x4x!tt.ptr<f32>>
    %10 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %11 = tt.expand_dims %10 {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
    %12 = arith.muli %11, %cst : tensor<8x1xi32>
    %13 = tt.broadcast %12 : tensor<8x1xi32> -> tensor<8x4xi32>
    %14 = tt.broadcast %3 : tensor<1x4xi32> -> tensor<8x4xi32>
    %15 = arith.addi %13, %14 : tensor<8x4xi32>
    %16 = tt.splat %arg1 : !tt.ptr<i64> -> tensor<8x4x!tt.ptr<i64>>
    %17 = tt.addptr %16, %15 : tensor<8x4x!tt.ptr<i64>>, tensor<8x4xi32>
    %18 = tt.load %17 : tensor<8x4x!tt.ptr<i64>>
    %19 = tt.gather %9[%18] {axis = 0 : i32} : (tensor<4x4xf32>, tensor<8x4xi64>) -> tensor<8x4xf32>
    %20 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<8x4x!tt.ptr<f32>>
    %21 = tt.addptr %20, %15 : tensor<8x4x!tt.ptr<f32>>, tensor<8x4xi32>
    tt.store %21, %19 : tensor<8x4x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0) -> (d0)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0, d1) -> (d0, 0)>
// CHECK: #[[$ATTR_3:.+]] = affine_map<(d0, d1) -> (0, d1)>
// CHECK-LABEL:   func.func @gather_test_kernel(
// CHECK-SAME:      %[[ARG0:.*]]: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %[[ARG1:.*]]: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %[[ARG2:.*]]: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %[[ARG3:.*]]: i32, %[[ARG4:.*]]: i32, %[[ARG5:.*]]: i32, %[[ARG6:.*]]: i32, %[[ARG7:.*]]: i32, %[[ARG8:.*]]: i32) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 4 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0 : i64
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 4 : i32
// CHECK:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<8x1xi32>
// CHECK:           %[[FILL_0:.*]] = linalg.fill ins(%[[CONSTANT_2]] : i32) outs(%[[EMPTY_0]] : tensor<8x1xi32>) -> tensor<8x1xi32>
// CHECK:           %[[EMPTY_1:.*]] = tensor.empty() : tensor<4x1xi32>
// CHECK:           %[[FILL_1:.*]] = linalg.fill ins(%[[CONSTANT_2]] : i32) outs(%[[EMPTY_1]] : tensor<4x1xi32>) -> tensor<4x1xi32>
// CHECK:           %[[EMPTY_2:.*]] = tensor.empty() : tensor<4xi32>
// CHECK:           %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]]], iterator_types = ["parallel"]} outs(%[[EMPTY_2]] : tensor<4xi32>) {
// CHECK:           ^bb0(%[[VAL_0:.*]]: i32):
// CHECK:             %[[INDEX_0:.*]] = linalg.index 0 : index
// CHECK:             %[[INDEX_CAST_0:.*]] = arith.index_cast %[[INDEX_0]] : index to i32
// CHECK:             linalg.yield %[[INDEX_CAST_0]] : i32
// CHECK:           } -> tensor<4xi32>
// CHECK:           %[[EXPAND_SHAPE_0:.*]] = tensor.expand_shape %[[GENERIC_0]] {{\[\[}}0, 1]] output_shape [4, 1] : tensor<4xi32> into tensor<4x1xi32>
// CHECK:           %[[GENERIC_1:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[EXPAND_SHAPE_0]], %[[FILL_1]] : tensor<4x1xi32>, tensor<4x1xi32>) outs(%[[EXPAND_SHAPE_0]] : tensor<4x1xi32>) {
// CHECK:           ^bb0(%[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i32):
// CHECK:             %[[MULI_0:.*]] = arith.muli %[[VAL_1]], %[[VAL_2]] : i32
// CHECK:             linalg.yield %[[MULI_0]] : i32
// CHECK:           } -> tensor<4x1xi32>
// CHECK:           %[[EXPAND_SHAPE_1:.*]] = tensor.expand_shape %[[GENERIC_0]] {{\[\[}}0, 1]] output_shape [1, 4] : tensor<4xi32> into tensor<1x4xi32>
// CHECK:           %[[EMPTY_3:.*]] = tensor.empty() : tensor<4x4xi32>
// CHECK:           %[[GENERIC_2:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_2]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[GENERIC_1]] : tensor<4x1xi32>) outs(%[[EMPTY_3]] : tensor<4x4xi32>) attrs =  {broadcastDims = array<i64: 1>} {
// CHECK:           ^bb0(%[[VAL_4:.*]]: i32, %[[VAL_5:.*]]: i32):
// CHECK:             linalg.yield %[[VAL_4]] : i32
// CHECK:           } -> tensor<4x4xi32>
// CHECK:           %[[EMPTY_4:.*]] = tensor.empty() : tensor<4x4xi32>
// CHECK:           %[[GENERIC_3:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_3]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[EXPAND_SHAPE_1]] : tensor<1x4xi32>) outs(%[[EMPTY_4]] : tensor<4x4xi32>) attrs =  {broadcastDims = array<i64: 0>} {
// CHECK:           ^bb0(%[[VAL_6:.*]]: i32, %[[VAL_7:.*]]: i32):
// CHECK:             linalg.yield %[[VAL_6]] : i32
// CHECK:           } -> tensor<4x4xi32>
// CHECK:           %[[GENERIC_4:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[GENERIC_2]], %[[GENERIC_3]] : tensor<4x4xi32>, tensor<4x4xi32>) outs(%[[GENERIC_2]] : tensor<4x4xi32>) {
// CHECK:           ^bb0(%[[VAL_8:.*]]: i32, %[[VAL_9:.*]]: i32, %[[VAL_10:.*]]: i32):
// CHECK:             %[[ADDI_0:.*]] = arith.addi %[[VAL_8]], %[[VAL_9]] : i32
// CHECK:             linalg.yield %[[ADDI_0]] : i32
// CHECK:           } -> tensor<4x4xi32>
// CHECK:           %[[SPLAT_0:.*]] = tensor.splat %[[ARG0]] : tensor<4x4x!tt.ptr<f32>>
// CHECK:           %[[GENERIC_5:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[SPLAT_0]], %[[GENERIC_4]] : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>) outs(%[[SPLAT_0]] : tensor<4x4x!tt.ptr<f32>>) {
// CHECK:           ^bb0(%[[VAL_11:.*]]: !tt.ptr<f32>, %[[VAL_12:.*]]: i32, %[[VAL_13:.*]]: !tt.ptr<f32>):
// CHECK:             %[[ADDPTR_0:.*]] = tt.addptr %[[VAL_11]], %[[VAL_12]] : !tt.ptr<f32>, i32
// CHECK:             linalg.yield %[[ADDPTR_0]] : !tt.ptr<f32>
// CHECK:           } -> tensor<4x4x!tt.ptr<f32>>
// CHECK:           %[[LOAD_0:.*]] = tt.load %[[GENERIC_5]] : tensor<4x4x!tt.ptr<f32>>
// CHECK:           %[[EMPTY_5:.*]] = tensor.empty() : tensor<8xi32>
// CHECK:           %[[GENERIC_6:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]]], iterator_types = ["parallel"]} outs(%[[EMPTY_5]] : tensor<8xi32>) {
// CHECK:           ^bb0(%[[VAL_14:.*]]: i32):
// CHECK:             %[[INDEX_1:.*]] = linalg.index 0 : index
// CHECK:             %[[INDEX_CAST_1:.*]] = arith.index_cast %[[INDEX_1]] : index to i32
// CHECK:             linalg.yield %[[INDEX_CAST_1]] : i32
// CHECK:           } -> tensor<8xi32>
// CHECK:           %[[EXPAND_SHAPE_2:.*]] = tensor.expand_shape %[[GENERIC_6]] {{\[\[}}0, 1]] output_shape [8, 1] : tensor<8xi32> into tensor<8x1xi32>
// CHECK:           %[[GENERIC_7:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[EXPAND_SHAPE_2]], %[[FILL_0]] : tensor<8x1xi32>, tensor<8x1xi32>) outs(%[[EXPAND_SHAPE_2]] : tensor<8x1xi32>) {
// CHECK:           ^bb0(%[[VAL_15:.*]]: i32, %[[VAL_16:.*]]: i32, %[[VAL_17:.*]]: i32):
// CHECK:             %[[MULI_1:.*]] = arith.muli %[[VAL_15]], %[[VAL_16]] : i32
// CHECK:             linalg.yield %[[MULI_1]] : i32
// CHECK:           } -> tensor<8x1xi32>
// CHECK:           %[[EMPTY_6:.*]] = tensor.empty() : tensor<8x4xi32>
// CHECK:           %[[GENERIC_8:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_2]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[GENERIC_7]] : tensor<8x1xi32>) outs(%[[EMPTY_6]] : tensor<8x4xi32>) attrs =  {broadcastDims = array<i64: 1>} {
// CHECK:           ^bb0(%[[VAL_18:.*]]: i32, %[[VAL_19:.*]]: i32):
// CHECK:             linalg.yield %[[VAL_18]] : i32
// CHECK:           } -> tensor<8x4xi32>
// CHECK:           %[[EMPTY_7:.*]] = tensor.empty() : tensor<8x4xi32>
// CHECK:           %[[GENERIC_9:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_3]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[EXPAND_SHAPE_1]] : tensor<1x4xi32>) outs(%[[EMPTY_7]] : tensor<8x4xi32>) attrs =  {broadcastDims = array<i64: 0>} {
// CHECK:           ^bb0(%[[VAL_20:.*]]: i32, %[[VAL_21:.*]]: i32):
// CHECK:             linalg.yield %[[VAL_20]] : i32
// CHECK:           } -> tensor<8x4xi32>
// CHECK:           %[[GENERIC_10:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[GENERIC_8]], %[[GENERIC_9]] : tensor<8x4xi32>, tensor<8x4xi32>) outs(%[[GENERIC_8]] : tensor<8x4xi32>) {
// CHECK:           ^bb0(%[[VAL_22:.*]]: i32, %[[VAL_23:.*]]: i32, %[[VAL_24:.*]]: i32):
// CHECK:             %[[ADDI_1:.*]] = arith.addi %[[VAL_22]], %[[VAL_23]] : i32
// CHECK:             linalg.yield %[[ADDI_1]] : i32
// CHECK:           } -> tensor<8x4xi32>
// CHECK:           %[[SPLAT_1:.*]] = tensor.splat %[[ARG1]] : tensor<8x4x!tt.ptr<i64>>
// CHECK:           %[[GENERIC_11:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[SPLAT_1]], %[[GENERIC_10]] : tensor<8x4x!tt.ptr<i64>>, tensor<8x4xi32>) outs(%[[SPLAT_1]] : tensor<8x4x!tt.ptr<i64>>) {
// CHECK:           ^bb0(%[[VAL_25:.*]]: !tt.ptr<i64>, %[[VAL_26:.*]]: i32, %[[VAL_27:.*]]: !tt.ptr<i64>):
// CHECK:             %[[ADDPTR_1:.*]] = tt.addptr %[[VAL_25]], %[[VAL_26]] : !tt.ptr<i64>, i32
// CHECK:             linalg.yield %[[ADDPTR_1]] : !tt.ptr<i64>
// CHECK:           } -> tensor<8x4x!tt.ptr<i64>>
// CHECK:           %[[LOAD_1:.*]] = tt.load %[[GENERIC_11]] : tensor<8x4x!tt.ptr<i64>>
// CHECK:           %[[EMPTY_8:.*]] = tensor.empty() : tensor<8x4xf32>
// CHECK:           %[[GENERIC_12:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[LOAD_1]] : tensor<8x4xi64>) outs(%[[EMPTY_8]] : tensor<8x4xf32>) {
// CHECK:           ^bb0(%[[VAL_28:.*]]: i64, %[[VAL_29:.*]]: f32):
// CHECK:             %[[INDEX_CAST_2:.*]] = arith.index_cast %[[VAL_28]] : i64 to index
// CHECK:             %[[INDEX_2:.*]] = linalg.index 1 : index
// CHECK:             %[[INDEX_CAST_3:.*]] = arith.index_cast %[[VAL_28]] : i64 to index
// CHECK:             %[[CMPI_0:.*]] = arith.cmpi slt, %[[INDEX_CAST_3]], %[[CONSTANT_0]] : index
// CHECK:             cf.assert %[[CMPI_0]], "index must be smaller than axis size"
// CHECK:             %[[CMPI_1:.*]] = arith.cmpi sge, %[[VAL_28]], %[[CONSTANT_1]] : i64
// CHECK:             cf.assert %[[CMPI_1]], "index must be larger or equal to 0"
// CHECK:             %[[EXTRACT_0:.*]] = tensor.extract %[[LOAD_0]]{{\[}}%[[INDEX_CAST_2]], %[[INDEX_2]]] : tensor<4x4xf32>
// CHECK:             linalg.yield %[[EXTRACT_0]] : f32
// CHECK:           } -> tensor<8x4xf32>
// CHECK:           %[[SPLAT_2:.*]] = tensor.splat %[[ARG2]] : tensor<8x4x!tt.ptr<f32>>
// CHECK:           %[[GENERIC_13:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[SPLAT_2]], %[[GENERIC_10]] : tensor<8x4x!tt.ptr<f32>>, tensor<8x4xi32>) outs(%[[SPLAT_2]] : tensor<8x4x!tt.ptr<f32>>) {
// CHECK:           ^bb0(%[[VAL_30:.*]]: !tt.ptr<f32>, %[[VAL_31:.*]]: i32, %[[VAL_32:.*]]: !tt.ptr<f32>):
// CHECK:             %[[ADDPTR_2:.*]] = tt.addptr %[[VAL_30]], %[[VAL_31]] : !tt.ptr<f32>, i32
// CHECK:             linalg.yield %[[ADDPTR_2]] : !tt.ptr<f32>
// CHECK:           } -> tensor<8x4x!tt.ptr<f32>>
// CHECK:           tt.store %[[GENERIC_13]], %[[GENERIC_12]] : tensor<8x4x!tt.ptr<f32>>
// CHECK:           return
// CHECK:         }