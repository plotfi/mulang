// RUN: muc --emit=llvm %s 2>&1 | FileCheck %s

// CHECK: define i32 @answer()
// CHECK-NEXT: ret i32 42

// CHECK: define i32 @add(i32 %0, i32 %1)
// CHECK: add i32

// CHECK: define i32 @sub(i32 %0, i32 %1)
// CHECK: sub i32

// CHECK: define i32 @mul(i32 %0, i32 %1)
// CHECK: mul i32

// CHECK: define i32 @div(i32 %0, i32 %1)
// CHECK: sdiv i32

// CHECK: define i32 @mod(i32 %0, i32 %1)
// CHECK: srem i32

// CHECK: define i32 @neg(i32 %0)
// CHECK: sub i32 0,

// CHECK: define i32 @bitinvert(i32 %0)
// CHECK: xor i32

// CHECK: define i32 @boolinvert(i32 %0)
// CHECK: icmp eq i32
// CHECK: zext i1

// CHECK: define i32 @bitand(i32 %0, i32 %1)
// CHECK: and i32

// CHECK: define i32 @bitor(i32 %0, i32 %1)
// CHECK: or i32

// CHECK: define i1 @booland(i32 %0, i32 %1)
// CHECK: icmp ne i32
// CHECK: icmp ne i32
// CHECK: and i1

// CHECK: define i1 @boolor(i32 %0, i32 %1)
// CHECK: icmp ne i32
// CHECK: icmp ne i32
// CHECK: or i1

// CHECK: define i32 @iftest(i1 %0)
// CHECK: br i1

fn answer() -> int32 {
  return 42;
}

fn add(a: int32, b: int32) -> int32 {
  return a + b;
}

fn sub(a: int32, b: int32) -> int32 {
  return a - b;
}

fn mul(a: int32, b: int32) -> int32 {
  return a * b;
}

fn div(a: int32, b: int32) -> int32 {
  return a / b;
}

fn mod(a: int32, b: int32) -> int32 {
  return a % b;
}

fn neg(a: int32) -> int32 {
  return -a;
}

fn bitinvert(a: int32) -> int32 {
  return ~a;
}

fn boolinvert(a: int32) -> int32 {
  return !a;
}

fn bitand(a: int32, b: int32) -> int32 {
  return a & b;
}

fn bitor(a: int32, b: int32) -> int32 {
  return a | b;
}

fn booland(a: int32, b: int32) -> bool {
  return a && b;
}

fn boolor(a: int32, b: int32) -> bool {
  return a || b;
}

fn iftest(a: bool) -> int32 {
  if a {
    return 1;
  }
  return 0;
}
