// RUN: not muc --emit=mlir %s 2>&1 | FileCheck %s
//
// CHECK: loc("main":5:0): error: redefinition of symbol named 'f'
fn f(a: bool) -> int32 {}
fn f(a: bool) -> int32 {}
