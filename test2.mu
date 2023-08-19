
fn bar () -> int {
  return 42;
}

fn foo (b: int) -> int {
  var d: int = 21;
  if b < 2 {
    d = 24;
  }
  var a: int = 42;
  var c: int = bar();
  return a + b + c - d;
}

