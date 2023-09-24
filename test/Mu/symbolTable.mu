// RUN: muc --emit=mlir %s 2>&1 | FileCheck %s

// CHECK: module

// CHECK: mu.func private @f(%arg0: i32, %arg1: i32) -> i32
// CHECK-NEXT: mu.add %arg0, %arg1 : i32
// CHECK-NEXT: mu.return

// CHECK: mu.func private @g(%arg0: i32) -> i32 {
// CHECK-NEXT: mu.return %arg0 : i32

fn f(a: int32, b: int32) -> int32 {
  return a + b;
}

fn g(c: int32) -> int32 {
  return c;
}
