// RUN: triton-shared-opt --triton-ptr-to-memref %s | FileCheck %s

module {
  tt.func private @identity(%arg0: !tt.ptr<f32>) -> !tt.ptr<f32> {
    tt.return %arg0 : !tt.ptr<f32>
  }

  tt.func @kernel(%arg0: !tt.ptr<f32>) -> !tt.ptr<f32> {
    %0 = tt.call @identity(%arg0) : (!tt.ptr<f32>) -> !tt.ptr<f32>
    tt.return %0 : !tt.ptr<f32>
  }
}

// CHECK: tt.func private @identity(%arg0: memref<*xf32>) -> memref<*xf32> {
// CHECK:   tt.return %arg0 : memref<*xf32>
// CHECK: }
// CHECK: tt.func @kernel(%arg0: memref<*xf32>) -> memref<*xf32> {
// CHECK:   %[[RET:.*]] = tt.call @identity(%arg0) : (memref<*xf32>) -> memref<*xf32>
// CHECK:   tt.return %[[RET]] : memref<*xf32>
// CHECK: }
// CHECK-NOT: unrealized_conversion_cast
