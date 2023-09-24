// RUN: muc --emit=mlir %s 2>&1 | FileCheck %s

// CHECK: module {
// CHECK-NEXT:   mu.func private @bitwise(%arg0: i32) -> i32 {
// CHECK-NEXT:     %0 = mu.mlir.constant(42 : i32) : i32
// CHECK-NEXT:     %1 = mu.mlir.constant(2 : i32) : i32
// CHECK-NEXT:     %2 = mu.and %0, %1 : i32
// CHECK-NEXT:     %3 = mu.or %arg0, %2 : i32
// CHECK-NEXT:     mu.return %3 : i32
// CHECK-NEXT:   }
// CHECK-NEXT: }

fn bitwise(a: int32) -> int32 {
  var b: int32 = 42;
  var d: int32 = 2;
  return a | b & d;
}