// RUN: muc --emit=mlir %s 2>&1 | FileCheck %s

// CHECK: mu.func @main() -> i32
// CHECK-NEXT: mu.mlir.constant(0 : i32) : i32
// CHECK-NEXT: mu.return

fn main() -> int32 {
  return 0;
}
