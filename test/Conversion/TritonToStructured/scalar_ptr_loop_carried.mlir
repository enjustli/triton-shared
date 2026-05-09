// RUN: triton-shared-opt --triton-to-structured %s | FileCheck %s

module {
  tt.func @scalar_ptr_loop_carried(%arg0: !tt.ptr<f32>, %arg1: i32) {
    %cst = arith.constant 0.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = scf.for %i = %c0_i32 to %arg1 step %c1_i32
        iter_args(%ptr = %arg0) -> (!tt.ptr<f32>) : i32 {
      %cond = arith.cmpi eq, %i, %c0_i32 : i32
      %1 = scf.if %cond -> (!tt.ptr<f32>) {
        scf.yield %arg0 : !tt.ptr<f32>
      } else {
        scf.yield %ptr : !tt.ptr<f32>
      }
      scf.yield %1 : !tt.ptr<f32>
    }
    tt.store %0, %cst : !tt.ptr<f32>
    tt.return
  }
}

// CHECK-LABEL: tt.func @scalar_ptr_loop_carried
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: scf.for
// CHECK-SAME: iter_args(%{{.*}} = %arg0, %{{.*}} = %[[C0]])
// CHECK: tt.store
// CHECK-NOT: tts.get_structured_state
