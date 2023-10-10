// RUN: muc --emit=mlir %s 2>&1 | FileCheck %s

// CHECK:  mu.func private @g(%arg0: i1) -> i32 {
// CHECK-NEXT:    "mu.if"(%arg0) ({
// CHECK-NEXT:      "mu.break"() : () -> ()
// CHECK-NEXT:    }) : (i1) -> ()
// CHECK-NEXT:    %0 = mu.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    mu.return %0 : i32
// CHECK-NEXT:  }

// CHECK:  mu.func private @f(%arg0: i1) -> i32 {
// CHECK-NEXT:    "mu.if"(%arg0) ({
// CHECK-NEXT:      %1 = mu.mlir.constant(1 : i32) : i32
// CHECK-NEXT:      mu.return %1 : i32
// CHECK-NEXT:    }) : (i1) -> ()
// CHECK-NEXT:    %0 = mu.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    mu.return %0 : i32
// CHECK-NEXT:  }


fn g(a: bool) -> int32 {
  if a {
  }
  return 0;
}

fn f(a: bool) -> int32 {
  if a {
    return 1;
  }
  return 0;
}
