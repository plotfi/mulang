// RUN: muc --emit=mlir %s 2>&1 | FileCheck %s

// CHECK: mu.func private @answer() -> i32
// CHECK-NEXT: mu.mlir.constant(42 : i32) : i32
// CHECK-NEXT: mu.return

fn answer() -> int32 {
  return 42;
}
