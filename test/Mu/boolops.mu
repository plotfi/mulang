// RUN: muc --emit=mlir %s 2>&1 | FileCheck %s

// CHECK: error: unknown variable 'd'
// CHECK: module {
// CHECK:   mu.func private @andf(%arg0: i32, %arg1: i32) -> i1 {
// CHECK-NEXT:     %alloca = memref.alloca() : memref<i32>
// CHECK-NEXT:     memref.store %arg0, %alloca[] : memref<i32>
// CHECK-NEXT:     %alloca_0 = memref.alloca() : memref<i32>
// CHECK-NEXT:     memref.store %arg1, %alloca_0[] : memref<i32>
// CHECK-NEXT:     %0 = memref.load %alloca[] : memref<i32>
// CHECK-NEXT:     %1 = memref.load %alloca_0[] : memref<i32>
// CHECK-NEXT:     %2 = mu.andb %0, %1 : (i32, i32) -> i1
// CHECK-NEXT:     mu.return %2 : i1
// CHECK-NEXT:   }
// CHECK:   mu.func private @orf(%arg0: i32, %arg1: i32) -> i1 {
// CHECK-NEXT:     %alloca = memref.alloca() : memref<i32>
// CHECK-NEXT:     memref.store %arg0, %alloca[] : memref<i32>
// CHECK-NEXT:     %alloca_0 = memref.alloca() : memref<i32>
// CHECK-NEXT:     memref.store %arg1, %alloca_0[] : memref<i32>
// CHECK-NEXT:     %0 = memref.load %alloca[] : memref<i32>
// CHECK-NEXT:     %1 = memref.load %alloca_0[] : memref<i32>
// CHECK-NEXT:     %2 = mu.orb %0, %1 : (i32, i32) -> i1
// CHECK-NEXT:     mu.return %2 : i1
// CHECK-NEXT:   }
// CHECK:   mu.func private @andf2(%arg0: i32, %arg1: i1) -> i1 {
// CHECK-NEXT:     %alloca = memref.alloca() : memref<i32>
// CHECK-NEXT:     memref.store %arg0, %alloca[] : memref<i32>
// CHECK-NEXT:     %alloca_0 = memref.alloca() : memref<i1>
// CHECK-NEXT:     memref.store %arg1, %alloca_0[] : memref<i1>
// CHECK-NEXT:     %0 = memref.load %alloca[] : memref<i32>
// CHECK-NEXT:     %1 = memref.load %alloca_0[] : memref<i1>
// CHECK-NEXT:     %2 = mu.andb %0, %1 : (i32, i1) -> i1
// CHECK-NEXT:     mu.return %2 : i1
// CHECK-NEXT:   }
// CHECK:   mu.func private @orf2(%arg0: i32, %arg1: i1) -> i1 {
// CHECK-NEXT:     %alloca = memref.alloca() : memref<i32>
// CHECK-NEXT:     memref.store %arg0, %alloca[] : memref<i32>
// CHECK-NEXT:     %alloca_0 = memref.alloca() : memref<i1>
// CHECK-NEXT:     memref.store %arg1, %alloca_0[] : memref<i1>
// CHECK-NEXT:     %0 = memref.load %alloca[] : memref<i32>
// CHECK-NEXT:     %1 = memref.load %alloca_0[] : memref<i1>
// CHECK-NEXT:     %2 = mu.orb %0, %1 : (i32, i1) -> i1
// CHECK-NEXT:     mu.return %2 : i1
// CHECK-NEXT:   }
// CHECK:   mu.func private @andf3(%arg0: i1, %arg1: i32) -> i1 {
// CHECK-NEXT:     %alloca = memref.alloca() : memref<i1>
// CHECK-NEXT:     memref.store %arg0, %alloca[] : memref<i1>
// CHECK-NEXT:     %alloca_0 = memref.alloca() : memref<i32>
// CHECK-NEXT:     memref.store %arg1, %alloca_0[] : memref<i32>
// CHECK-NEXT:     %0 = memref.load %alloca[] : memref<i1>
// CHECK-NEXT:     %1 = memref.load %alloca_0[] : memref<i32>
// CHECK-NEXT:     %2 = mu.andb %0, %1 : (i1, i32) -> i1
// CHECK-NEXT:     mu.return %2 : i1
// CHECK-NEXT:   }
// CHECK:   mu.func private @orf3(%arg0: i1, %arg1: i32) -> i1 {
// CHECK-NEXT:     %alloca = memref.alloca() : memref<i1>
// CHECK-NEXT:     memref.store %arg0, %alloca[] : memref<i1>
// CHECK-NEXT:     %alloca_0 = memref.alloca() : memref<i32>
// CHECK-NEXT:     memref.store %arg1, %alloca_0[] : memref<i32>
// CHECK-NEXT:     %0 = memref.load %alloca[] : memref<i1>
// CHECK-NEXT:     %1 = memref.load %alloca_0[] : memref<i32>
// CHECK-NEXT:     %2 = mu.orb %0, %1 : (i1, i32) -> i1
// CHECK-NEXT:     mu.return %2 : i1
// CHECK-NEXT:   }
// CHECK:   mu.func private @orAndf(%arg0: i32, %arg1: i32) -> i1 {
// CHECK-NEXT:     %alloca = memref.alloca() : memref<i32>
// CHECK-NEXT:     memref.store %arg0, %alloca[] : memref<i32>
// CHECK-NEXT:     %alloca_0 = memref.alloca() : memref<i32>
// CHECK-NEXT:     memref.store %arg1, %alloca_0[] : memref<i32>
// CHECK-NEXT:     %0 = memref.load %alloca[] : memref<i32>
// CHECK-NEXT:     %1 = memref.load %alloca_0[] : memref<i32>
// CHECK-NEXT:     %2 = mu.orb %0, %1 : (i32, i32) -> i1
// CHECK-NEXT:     %alloca_1 = memref.alloca() : memref<i1>
// CHECK-NEXT:     memref.store %2, %alloca_1[] : memref<i1>
// CHECK-NEXT:     %3 = mu.mlir.constant(42 : i32) : i32
// CHECK-NEXT:     %alloca_2 = memref.alloca() : memref<i32>
// CHECK-NEXT:     memref.store %3, %alloca_2[] : memref<i32>
// CHECK-NEXT:     %4 = memref.load %alloca_1[] : memref<i1>
// CHECK-NEXT:     %5 = memref.load %alloca_2[] : memref<i32>
// CHECK-NEXT:     %6 = mu.andb %4, %5 : (i1, i32) -> i1
// CHECK-NEXT:     mu.return %6 : i1
// CHECK-NEXT:   }
// CHECK-NEXT: }

fn andf(a: int32, b: int32) -> bool {
  return a && b;
}

fn orf(a: int32, b: int32) -> bool {
  return a || b;
}

fn andf2(a: int32, b: bool) -> bool {
  return a && b;
}

fn orf2(a: int32, b: bool) -> bool {
  return a || b;
}

fn andf3(a: bool, b: int32) -> bool {
  return a && b;
}

fn orf3(a: bool, b: int32) -> bool {
  return a || b;
}

fn orAndf(a: int32, b: int32) -> bool {
  var c: bool = a || b ;
  var d: int32 = 42;
  return c && d;
}

fn orAndf2(a: int32, b: int32) -> bool {
  return a || b && d;
}
