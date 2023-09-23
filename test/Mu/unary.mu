// RUN: muc --emit=mlir %s 2>&1 | FileCheck %s

// CHECK: module

// CHECK: mu.func private @answer1() -> i32
// CHECK-NEXT: mu.mlir.constant(42 : i32) : i32
// CHECK-NEXT: mu.neg
// CHECK-NEXT: mu.return

// CHECK: mu.func private @answer2() -> i32
// CHECK-NEXT: mu.mlir.constant(42 : i32) : i32
// CHECK-NEXT: mu.invert

// CHECK: mu.func private @answer3() -> i32
// CHECK-NEXT: mu.mlir.constant(42 : i32) : i32
// CHECK-NEXT: mu.not

fn answer1() -> int32 {
  return -42;
}

fn answer2() -> int32 {
  return ~42;
}

fn answer3() -> int32 {
  return !42;
}
