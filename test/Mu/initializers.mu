// RUN: muc --emit=mlir %s 2>&1 | FileCheck %s

// CHECK: module {
// CHECK-NEXT:   mu.func private @f(%arg0: i32) -> i32 {
// CHECK-NEXT:     %0 = mu.mlir.constant(42 : i32) : i32
// CHECK-NEXT:     %1 = mu.add %arg0, %0 : i32
// CHECK-NEXT:     mu.return %1 : i32
// CHECK-NEXT:   }
// CHECK-NEXT: }

fn f(a: int32) -> int32 {
  var b: int32 = 42;
  return a + b;
}
