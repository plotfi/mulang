// RUN: muc --emit=mlir %s 2>&1 | FileCheck %s

// CHECK: module {
// CHECK-NEXT:   mu.func private @f(%arg0: i32, %arg1: i32) -> i32 {
// CHECK-NEXT:     %0 = mu.mlir.constant(42 : i32) : i32
// CHECK-NEXT:     %1 = mu.mlir.constant(1337 : i32) : i32
// CHECK-NEXT:     %2 = mu.add %arg0, %0 : i32
// CHECK-NEXT:     %3 = mu.add %2, %arg1 : i32
// CHECK-NEXT:     %4 = mu.add %3, %1 : i32
// CHECK-NEXT:     mu.return %4 : i32
// CHECK-NEXT:   }
// CHECK-NEXT: }

fn f(a: int32, c: int32) -> int32 {
  var b: int32 = 42;
  var d: int32 = 1337;
  return a + b + c + d;
}
