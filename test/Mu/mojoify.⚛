// RUN: muc --emit=mlir %s 2>&1 | FileCheck %s

// CHECK: module {
// CHECK-NEXT:   mu.func private @mod(%arg0: i32) -> i32 {
// CHECK-NEXT:     %0 = mu.mlir.constant(42 : i32) : i32
// CHECK-NEXT:     %1 = mu.mod %0, %arg0 : i32
// CHECK-NEXT:     mu.return %1 : i32
// CHECK-NEXT:   }
// CHECK-NEXT: }

fn mod(a: int32) -> int32 {
  var b: int32 = 42;
  return b % a;
}