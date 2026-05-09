// RUN: triton-shared-opt --triton-ptr-to-memref %s | FileCheck %s

module {
  func.func private @identity(%arg0: !tt.ptr<f32>) -> !tt.ptr<f32> {
    return %arg0 : !tt.ptr<f32>
  }

  func.func @kernel(%arg0: !tt.ptr<f32>) -> !tt.ptr<f32> {
    %0 = call @identity(%arg0) : (!tt.ptr<f32>) -> !tt.ptr<f32>
    return %0 : !tt.ptr<f32>
  }
}

// CHECK: func.func private @identity(%arg0: memref<*xf32>) -> memref<*xf32> {
// CHECK:   return %arg0 : memref<*xf32>
// CHECK: }
// CHECK: func.func @kernel(%arg0: memref<*xf32>) -> memref<*xf32> {
// CHECK:   %[[RET:.*]] = call @identity(%arg0) : (memref<*xf32>) -> memref<*xf32>
// CHECK:   return %[[RET]] : memref<*xf32>
// CHECK: }
// CHECK-NOT: unrealized_conversion_cast
