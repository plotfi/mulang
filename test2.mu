
fn bar() -> int32 {
  return 42;
}

fn foo(b: int64, z: int64) -> int64 {
  var d: int64 = 21;
  if b < 2 {
    d = 24;
  }
  /* var arr: [arr] = [int8](16); */
  var a: int64 = 42;
  var c: int64 = bar();
  return a + b + c - d * z;
}

