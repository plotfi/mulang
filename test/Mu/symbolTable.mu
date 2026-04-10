// RUN: muc --emit=mlir %s 2>&1 | FileCheck %s

// CHECK: module

// CHECK: mu.func private @f(%arg0: i32, %arg1: i32) -> i32
// CHECK:     %alloca = memref.alloca() : memref<i32>
// CHECK-NEXT:     memref.store %arg0, %alloca[] : memref<i32>
// CHECK-NEXT:     %alloca_0 = memref.alloca() : memref<i32>
// CHECK-NEXT:     memref.store %arg1, %alloca_0[] : memref<i32>
// CHECK-NEXT:     %0 = memref.load %alloca[] : memref<i32>
// CHECK-NEXT:     %1 = memref.load %alloca_0[] : memref<i32>
// CHECK-NEXT:     %2 = mu.add %0, %1 : i32
// CHECK-NEXT:     mu.return %2 : i32

// CHECK: mu.func private @g(%arg0: i32) -> i32 {
// CHECK-NEXT:     %alloca = memref.alloca() : memref<i32>
// CHECK-NEXT:     memref.store %arg0, %alloca[] : memref<i32>
// CHECK-NEXT:     %0 = memref.load %alloca[] : memref<i32>
// CHECK-NEXT:     mu.return %0 : i32

fn f(a: int32, b: int32) -> int32 {
  return a + b;
}

fn g(c: int32) -> int32 {
  return c;
}
