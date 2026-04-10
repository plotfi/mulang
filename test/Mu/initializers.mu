// RUN: muc --emit=mlir %s 2>&1 | FileCheck %s

// CHECK: module {
// CHECK-NEXT:   mu.func private @f(%arg0: i32) -> i32 {
// CHECK-NEXT:     %alloca = memref.alloca() : memref<i32>
// CHECK-NEXT:     memref.store %arg0, %alloca[] : memref<i32>
// CHECK-NEXT:     %0 = mu.mlir.constant(42 : i32) : i32
// CHECK-NEXT:     %alloca_0 = memref.alloca() : memref<i32>
// CHECK-NEXT:     memref.store %0, %alloca_0[] : memref<i32>
// CHECK-NEXT:     %1 = memref.load %alloca[] : memref<i32>
// CHECK-NEXT:     %2 = memref.load %alloca_0[] : memref<i32>
// CHECK-NEXT:     %3 = mu.add %1, %2 : i32
// CHECK-NEXT:     mu.return %3 : i32
// CHECK-NEXT:   }
// CHECK-NEXT: }

fn f(a: int32) -> int32 {
  var b: int32 = 42;
  return a + b;
}
