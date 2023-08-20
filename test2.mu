
fn bar(a: int32) -> int32 {
  return 42 + a;
}

fn foo(b: int64, z: int64) -> int64 {
  var d: int64 = 21;
  if b < 2 {
    d = 24;
  }
  /* var arr1: int8[24] = int8[24](2) */
  // var arr2: int8[] = int8[]()
  // var arr2: int8[] = int8[]()
  var a: int64 = 42;
  var c: int64 = bar();
  return a + b + c - d * z;
}

