// RUN: muc --emit=mlir %s 2>&1 | FileCheck %s

// CHECK: module
fn answer() -> int32 {
  return -42;
}
