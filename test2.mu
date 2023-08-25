
// type int = int32;

fn bar(a: int32) -> int32 {
  return 42 + a;
}

fn foo(b: int32, z: int32) -> int32 {
  var d: int32 = 21;
  if b < 2 {
    d = 24;
  }
  /* var arr1: int8[24] = int8[24](2) */
  // var arr2: int8[] = int8[]()
  // var arr2: int8[] = int8[]()
  // Potential Syntax:
  // define char int8; // typealias
  // define foo() -> int8 {} // function definition
  // define S { a: int32, b: int16 } // struct definition
  var a: int32 = 42;
  var c: int32 = bar(12);
  c = 32 + c;
  return a + b + c - d * z;
}

fn fibonacci(n: int32) -> int32 {
  
  if n < 1 {
    return 0;
  } else  {
    if n == 1 {
      return 1;
    }
  }

  var prev: int32 = 0;
  var current: int32 = 1;
  var i: int32 = 2;

  while i <= n {
    var next: int32 = prev + current;
    prev = current;
    current = next;
    i = i + 1;
  }
  
  return current;
}

fn main() -> int32 {

  var a: int32 = bar(13) + foo(1, 2);
  _ = printf("%d\n", a);

  var fib: int32 = fibonacci(5);
  _ = printf("fibonacci of 5 is: %d\n", fib);
  // asdf();

  return 0;
}
