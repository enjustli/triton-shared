// RUN: triton-shared-opt --split-input-file --linalg-to-vector %s | FileCheck %s



// CHECK-LABEL:   func.func @add() {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_2:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_3:.*]] = arith.constant 128 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 4096 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_6:.*]] = memref.alloc() : memref<4096xi32>
// CHECK:           %[[VAL_7:.*]] = memref.alloc() : memref<4096xf32>
// CHECK:           %[[VAL_8:.*]] = memref.alloc() : memref<f32>
// CHECK:           %[[VAL_9:.*]] = memref.alloc() : memref<i32>
// CHECK:           %[[VAL_10:.*]] = memref.alloc() {alignment = 512 : i64} : memref<128xf32>
// CHECK:           %[[VAL_11:.*]] = memref.load %[[VAL_8]][] : memref<f32>
// CHECK:           %[[VAL_12:.*]] = vector.broadcast %[[VAL_11]] : f32 to vector<128xf32>
// CHECK:           vector.transfer_write %[[VAL_12]], %[[VAL_10]]{{\[}}%[[VAL_5]]] {in_bounds = [true]} : vector<128xf32>, memref<128xf32>
// CHECK:           %[[VAL_13:.*]] = memref.alloc() {alignment = 512 : i64} : memref<128xi32>
// CHECK:           %[[VAL_14:.*]] = memref.load %[[VAL_9]][] : memref<i32>
// CHECK:           %[[VAL_15:.*]] = vector.broadcast %[[VAL_14]] : i32 to vector<128xi32>
// CHECK:           vector.transfer_write %[[VAL_15]], %[[VAL_13]]{{\[}}%[[VAL_5]]] {in_bounds = [true]} : vector<128xi32>, memref<128xi32>
// CHECK:           scf.for %[[VAL_16:.*]] = %[[VAL_5]] to %[[VAL_4]] step %[[VAL_3]] {
// CHECK:             %[[VAL_17:.*]] = vector.transfer_read %[[VAL_7]]{{\[}}%[[VAL_16]]], %[[VAL_2]] {in_bounds = [true]} : memref<4096xf32>, vector<128xf32>
// CHECK:             %[[VAL_18:.*]] = vector.transfer_read %[[VAL_6]]{{\[}}%[[VAL_16]]], %[[VAL_1]] {in_bounds = [true]} : memref<4096xi32>, vector<128xi32>
// CHECK:             %[[VAL_19:.*]] = vector.transfer_read %[[VAL_10]]{{\[}}%[[VAL_5]]], %[[VAL_2]] {in_bounds = [true]} : memref<128xf32>, vector<128xf32>
// CHECK:             %[[VAL_20:.*]] = vector.transfer_read %[[VAL_13]]{{\[}}%[[VAL_5]]], %[[VAL_1]] {in_bounds = [true]} : memref<128xi32>, vector<128xi32>
// CHECK:             %[[VAL_21:.*]] = arith.cmpf oeq, %[[VAL_17]], %[[VAL_19]] : vector<128xf32>
// CHECK:             %[[VAL_22:.*]] = arith.cmpi slt, %[[VAL_18]], %[[VAL_20]] : vector<128xi32>
// CHECK:             %[[VAL_23:.*]] = arith.andi %[[VAL_21]], %[[VAL_22]] : vector<128xi1>
// CHECK:             %[[VAL_24:.*]] = arith.cmpf ogt, %[[VAL_17]], %[[VAL_19]] : vector<128xf32>
// CHECK:             %[[VAL_25:.*]] = arith.ori %[[VAL_24]], %[[VAL_23]] : vector<128xi1>
// CHECK:             %[[VAL_26:.*]] = arith.select %[[VAL_25]], %[[VAL_17]], %[[VAL_19]] : vector<128xi1>, vector<128xf32>
// CHECK:             %[[VAL_27:.*]] = arith.select %[[VAL_25]], %[[VAL_18]], %[[VAL_20]] : vector<128xi1>, vector<128xi32>
// CHECK:             vector.transfer_write %[[VAL_26]], %[[VAL_10]]{{\[}}%[[VAL_5]]] {in_bounds = [true]} : vector<128xf32>, memref<128xf32>
// CHECK:             vector.transfer_write %[[VAL_27]], %[[VAL_13]]{{\[}}%[[VAL_5]]] {in_bounds = [true]} : vector<128xi32>, memref<128xi32>
// CHECK:           }
// CHECK:           scf.for %[[VAL_28:.*]] = %[[VAL_5]] to %[[VAL_3]] step %[[VAL_0]] {
// CHECK:             %[[VAL_29:.*]] = memref.load %[[VAL_10]]{{\[}}%[[VAL_28]]] : memref<128xf32>
// CHECK:             %[[VAL_30:.*]] = memref.load %[[VAL_13]]{{\[}}%[[VAL_28]]] : memref<128xi32>
// CHECK:             %[[VAL_31:.*]] = memref.load %[[VAL_8]][] : memref<f32>
// CHECK:             %[[VAL_32:.*]] = memref.load %[[VAL_9]][] : memref<i32>
// CHECK:             %[[VAL_33:.*]] = arith.cmpf oeq, %[[VAL_29]], %[[VAL_31]] : f32
// CHECK:             %[[VAL_34:.*]] = arith.cmpi slt, %[[VAL_30]], %[[VAL_32]] : i32
// CHECK:             %[[VAL_35:.*]] = arith.andi %[[VAL_33]], %[[VAL_34]] : i1
// CHECK:             %[[VAL_36:.*]] = arith.cmpf ogt, %[[VAL_29]], %[[VAL_31]] : f32
// CHECK:             %[[VAL_37:.*]] = arith.ori %[[VAL_36]], %[[VAL_35]] : i1
// CHECK:             %[[VAL_38:.*]] = arith.select %[[VAL_37]], %[[VAL_29]], %[[VAL_31]] : f32
// CHECK:             %[[VAL_39:.*]] = arith.select %[[VAL_37]], %[[VAL_30]], %[[VAL_32]] : i32
// CHECK:             memref.store %[[VAL_38]], %[[VAL_8]][] : memref<f32>
// CHECK:             memref.store %[[VAL_39]], %[[VAL_9]][] : memref<i32>
// CHECK:           }
// CHECK:           return
// CHECK:         }
module {
  func.func @argmax(%arg0: tensor<1024xi32>, %arg1: tensor<1024xf32>, %arg2: tensor<f32>, %arg3: tensor<i32>) -> (tensor<f32>, tensor<i32>) {
    %reduced:2 = linalg.reduce ins(%arg1, %arg0 : tensor<1024xf32>, tensor<1024xi32>) outs(%arg2, %arg3 : tensor<f32>, tensor<i32>) dimensions = [0]
      (%in: f32, %in_0: i32, %init: f32, %init_1: i32) {
        %0 = arith.cmpf oeq, %in, %init : f32
        %1 = arith.cmpi slt, %in_0, %init_1 : i32
        %2 = arith.andi %0, %1 : i1
        %3 = arith.cmpf ogt, %in, %init : f32
        %4 = arith.ori %3, %2 : i1
        %5 = arith.select %4, %in, %init : f32
        %6 = arith.select %4, %in_0, %init_1 : i32
        linalg.yield %5, %6 : f32, i32
      }
    return %reduced#0, %reduced#1 : tensor<f32>, tensor<i32>
  }
}


// -----

// CHECK-LABEL:   func.func @argmax_012() {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_2:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_3:.*]] = arith.constant 131 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 3 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 128 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 259 : index
// CHECK:           %[[VAL_7:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_8:.*]] = memref.alloc() : memref<259xi32>
// CHECK:           %[[VAL_9:.*]] = memref.alloc() : memref<259xf32>
// CHECK:           %[[VAL_10:.*]] = memref.alloc() : memref<f32>
// CHECK:           %[[VAL_11:.*]] = memref.alloc() : memref<i32>
// CHECK:           %[[VAL_12:.*]] = memref.alloc() {alignment = 512 : i64} : memref<128xf32>
// CHECK:           %[[VAL_13:.*]] = memref.load %[[VAL_10]][] : memref<f32>
// CHECK:           %[[VAL_14:.*]] = vector.broadcast %[[VAL_13]] : f32 to vector<128xf32>
// CHECK:           vector.transfer_write %[[VAL_14]], %[[VAL_12]]{{\[}}%[[VAL_7]]] {in_bounds = [true]} : vector<128xf32>, memref<128xf32>
// CHECK:           %[[VAL_15:.*]] = memref.alloc() {alignment = 512 : i64} : memref<128xi32>
// CHECK:           %[[VAL_16:.*]] = memref.load %[[VAL_11]][] : memref<i32>
// CHECK:           %[[VAL_17:.*]] = vector.broadcast %[[VAL_16]] : i32 to vector<128xi32>
// CHECK:           vector.transfer_write %[[VAL_17]], %[[VAL_15]]{{\[}}%[[VAL_7]]] {in_bounds = [true]} : vector<128xi32>, memref<128xi32>
// CHECK:           scf.for %[[VAL_18:.*]] = %[[VAL_7]] to %[[VAL_6]] step %[[VAL_5]] {
// CHECK:             %[[VAL_19:.*]] = vector.create_mask %[[VAL_4]] : vector<128xi1>
// CHECK:             %[[VAL_20:.*]] = arith.cmpi sle, %[[VAL_18]], %[[VAL_3]] : index
// CHECK:             %[[VAL_21:.*]] = scf.if %[[VAL_20]] -> (vector<128xf32>) {
// CHECK:               %[[VAL_22:.*]] = vector.transfer_read %[[VAL_9]]{{\[}}%[[VAL_18]]], %[[VAL_2]] {in_bounds = [true]} : memref<259xf32>, vector<128xf32>
// CHECK:               scf.yield %[[VAL_22]] : vector<128xf32>
// CHECK:             } else {
// CHECK:               %[[VAL_23:.*]] = vector.mask %[[VAL_19]] { vector.transfer_read %[[VAL_9]]{{\[}}%[[VAL_18]]], %[[VAL_2]] {in_bounds = [true]} : memref<259xf32>, vector<128xf32> } : vector<128xi1> -> vector<128xf32>
// CHECK:               scf.yield %[[VAL_23]] : vector<128xf32>
// CHECK:             }
// CHECK:             %[[VAL_24:.*]] = vector.create_mask %[[VAL_4]] : vector<128xi1>
// CHECK:             %[[VAL_25:.*]] = arith.cmpi sle, %[[VAL_18]], %[[VAL_3]] : index
// CHECK:             %[[VAL_26:.*]] = scf.if %[[VAL_25]] -> (vector<128xi32>) {
// CHECK:               %[[VAL_27:.*]] = vector.transfer_read %[[VAL_8]]{{\[}}%[[VAL_18]]], %[[VAL_1]] {in_bounds = [true]} : memref<259xi32>, vector<128xi32>
// CHECK:               scf.yield %[[VAL_27]] : vector<128xi32>
// CHECK:             } else {
// CHECK:               %[[VAL_28:.*]] = vector.mask %[[VAL_24]] { vector.transfer_read %[[VAL_8]]{{\[}}%[[VAL_18]]], %[[VAL_1]] {in_bounds = [true]} : memref<259xi32>, vector<128xi32> } : vector<128xi1> -> vector<128xi32>
// CHECK:               scf.yield %[[VAL_28]] : vector<128xi32>
// CHECK:             }
// CHECK:             %[[VAL_29:.*]] = vector.create_mask %[[VAL_4]] : vector<128xi1>
// CHECK:             %[[VAL_30:.*]] = arith.cmpi sle, %[[VAL_18]], %[[VAL_3]] : index
// CHECK:             %[[VAL_31:.*]] = scf.if %[[VAL_30]] -> (vector<128xf32>) {
// CHECK:               %[[VAL_32:.*]] = vector.transfer_read %[[VAL_12]]{{\[}}%[[VAL_7]]], %[[VAL_2]] {in_bounds = [true]} : memref<128xf32>, vector<128xf32>
// CHECK:               scf.yield %[[VAL_32]] : vector<128xf32>
// CHECK:             } else {
// CHECK:               %[[VAL_33:.*]] = vector.mask %[[VAL_29]] { vector.transfer_read %[[VAL_12]]{{\[}}%[[VAL_7]]], %[[VAL_2]] {in_bounds = [true]} : memref<128xf32>, vector<128xf32> } : vector<128xi1> -> vector<128xf32>
// CHECK:               scf.yield %[[VAL_33]] : vector<128xf32>
// CHECK:             }
// CHECK:             %[[VAL_34:.*]] = vector.create_mask %[[VAL_4]] : vector<128xi1>
// CHECK:             %[[VAL_35:.*]] = arith.cmpi sle, %[[VAL_18]], %[[VAL_3]] : index
// CHECK:             %[[VAL_36:.*]] = scf.if %[[VAL_35]] -> (vector<128xi32>) {
// CHECK:               %[[VAL_37:.*]] = vector.transfer_read %[[VAL_15]]{{\[}}%[[VAL_7]]], %[[VAL_1]] {in_bounds = [true]} : memref<128xi32>, vector<128xi32>
// CHECK:               scf.yield %[[VAL_37]] : vector<128xi32>
// CHECK:             } else {
// CHECK:               %[[VAL_38:.*]] = vector.mask %[[VAL_34]] { vector.transfer_read %[[VAL_15]]{{\[}}%[[VAL_7]]], %[[VAL_1]] {in_bounds = [true]} : memref<128xi32>, vector<128xi32> } : vector<128xi1> -> vector<128xi32>
// CHECK:               scf.yield %[[VAL_38]] : vector<128xi32>
// CHECK:             }
// CHECK:             %[[VAL_39:.*]] = arith.cmpf oeq, %[[VAL_21]], %[[VAL_31]] : vector<128xf32>
// CHECK:             %[[VAL_40:.*]] = arith.cmpi slt, %[[VAL_26]], %[[VAL_36]] : vector<128xi32>
// CHECK:             %[[VAL_41:.*]] = arith.andi %[[VAL_39]], %[[VAL_40]] : vector<128xi1>
// CHECK:             %[[VAL_42:.*]] = arith.cmpf ogt, %[[VAL_21]], %[[VAL_31]] : vector<128xf32>
// CHECK:             %[[VAL_43:.*]] = arith.ori %[[VAL_42]], %[[VAL_41]] : vector<128xi1>
// CHECK:             %[[VAL_44:.*]] = arith.select %[[VAL_43]], %[[VAL_21]], %[[VAL_31]] : vector<128xi1>, vector<128xf32>
// CHECK:             %[[VAL_45:.*]] = arith.select %[[VAL_43]], %[[VAL_26]], %[[VAL_36]] : vector<128xi1>, vector<128xi32>
// CHECK:             %[[VAL_46:.*]] = vector.create_mask %[[VAL_4]] : vector<128xi1>
// CHECK:             %[[VAL_47:.*]] = arith.cmpi sle, %[[VAL_18]], %[[VAL_3]] : index
// CHECK:             scf.if %[[VAL_47]] {
// CHECK:               vector.transfer_write %[[VAL_44]], %[[VAL_12]]{{\[}}%[[VAL_7]]] {in_bounds = [true]} : vector<128xf32>, memref<128xf32>
// CHECK:             } else {
// CHECK:               vector.mask %[[VAL_46]] { vector.transfer_write %[[VAL_44]], %[[VAL_12]]{{\[}}%[[VAL_7]]] {in_bounds = [true]} : vector<128xf32>, memref<128xf32> } : vector<128xi1>
// CHECK:             }
// CHECK:             %[[VAL_48:.*]] = vector.create_mask %[[VAL_4]] : vector<128xi1>
// CHECK:             %[[VAL_49:.*]] = arith.cmpi sle, %[[VAL_18]], %[[VAL_3]] : index
// CHECK:             scf.if %[[VAL_49]] {
// CHECK:               vector.transfer_write %[[VAL_45]], %[[VAL_15]]{{\[}}%[[VAL_7]]] {in_bounds = [true]} : vector<128xi32>, memref<128xi32>
// CHECK:             } else {
// CHECK:               vector.mask %[[VAL_48]] { vector.transfer_write %[[VAL_45]], %[[VAL_15]]{{\[}}%[[VAL_7]]] {in_bounds = [true]} : vector<128xi32>, memref<128xi32> } : vector<128xi1>
// CHECK:             }
// CHECK:           }
// CHECK:           scf.for %[[VAL_50:.*]] = %[[VAL_7]] to %[[VAL_5]] step %[[VAL_0]] {
// CHECK:             %[[VAL_51:.*]] = memref.load %[[VAL_12]]{{\[}}%[[VAL_50]]] : memref<128xf32>
// CHECK:             %[[VAL_52:.*]] = memref.load %[[VAL_15]]{{\[}}%[[VAL_50]]] : memref<128xi32>
// CHECK:             %[[VAL_53:.*]] = memref.load %[[VAL_10]][] : memref<f32>
// CHECK:             %[[VAL_54:.*]] = memref.load %[[VAL_11]][] : memref<i32>
// CHECK:             %[[VAL_55:.*]] = arith.cmpf oeq, %[[VAL_51]], %[[VAL_53]] : f32
// CHECK:             %[[VAL_56:.*]] = arith.cmpi slt, %[[VAL_52]], %[[VAL_54]] : i32
// CHECK:             %[[VAL_57:.*]] = arith.andi %[[VAL_55]], %[[VAL_56]] : i1
// CHECK:             %[[VAL_58:.*]] = arith.cmpf ogt, %[[VAL_51]], %[[VAL_53]] : f32
// CHECK:             %[[VAL_59:.*]] = arith.ori %[[VAL_58]], %[[VAL_57]] : i1
// CHECK:             %[[VAL_60:.*]] = arith.select %[[VAL_59]], %[[VAL_51]], %[[VAL_53]] : f32
// CHECK:             %[[VAL_61:.*]] = arith.select %[[VAL_59]], %[[VAL_52]], %[[VAL_54]] : i32
// CHECK:             memref.store %[[VAL_60]], %[[VAL_10]][] : memref<f32>
// CHECK:             memref.store %[[VAL_61]], %[[VAL_11]][] : memref<i32>
// CHECK:           }
// CHECK:           return
// CHECK:         }
func.func @argmax(%arg0: tensor<?xi32>, %arg1: tensor<?xf32>, %arg2: tensor<f32>, %arg3: tensor<i32>) -> (tensor<f32>, tensor<i32>) {
  %reduced:2 = linalg.reduce ins(%arg1, %arg0 : tensor<?xf32>, tensor<?xi32>) outs(%arg2, %arg3 : tensor<f32>, tensor<i32>) dimensions = [0]
    (%in: f32, %in_0: i32, %init: f32, %init_1: i32) {
      %0 = arith.cmpf oeq, %in, %init : f32
      %1 = arith.cmpi slt, %in_0, %init_1 : i32
      %2 = arith.andi %0, %1 : i1
      %3 = arith.cmpf ogt, %in, %init : f32
      %4 = arith.ori %3, %2 : i1
      %5 = arith.select %4, %in, %init : f32
      %6 = arith.select %4, %in_0, %init_1 : i32
      linalg.yield %5, %6 : f32, i32
    }
  return %reduced#0, %reduced#1 : tensor<f32>, tensor<i32>
}