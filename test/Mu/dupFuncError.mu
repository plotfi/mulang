// RUN: not muc --emit=mlir %s 2>&1 | FileCheck %s
//
// CHECK: error: 'mu.return' op does not return the same number of values (0) as the enclosing function (1)
// CHECK: error: module verification error
fn f(a: bool) -> int32 {}
fn f(a: bool) -> int32 {}
