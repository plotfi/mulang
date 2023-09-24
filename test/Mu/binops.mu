// RUN: muc --emit=mlir %s 2>&1 | FileCheck %s

// CHECK: module

// CHECK: mu.func private @add() -> i32
// CHECK-NEXT: mu.mlir.constant(23 : i32) : i32
// CHECK-NEXT: mu.mlir.constant(41 : i32) : i32
// CHECK-NEXT: mu.add
// CHECK-NEXT: mu.return

// CHECK: mu.func private @mul() -> i32
// CHECK-NEXT: mu.mlir.constant(2 : i32) : i32
// CHECK-NEXT: mu.mlir.constant(4 : i32) : i32
// CHECK-NEXT: mu.mul

fn add() -> int32 {
  return 23 + 41;
}

fn mul() -> int32 {
  return 2 * 4;
}
