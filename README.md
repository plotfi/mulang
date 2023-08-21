# µ-lang

A language just for fun.

The grammar was orignally taken from a C grammer at https://www.lysator.liu.se/c but has since been heavily modified.

The syntax is as follows:

```mu

fn fibonacci(n: int) -> int {

  if n < 2 {
    return n;
  }

  var prev: int = 0;
  var current: int = 1;
  var i: int = 2;

  while i <= n {
    var next: int = prev + current;
    prev = current;
    current = next;
    i = i + 1;
  }

  return current;
}

fn main() -> int {
  var fib: int = fibonacci(5);
  _ = printf("fibonacci of 5 is: %d\n", fib);

  return 0;
}
```

I intend to add structs and arrays eventually, and my intention is to use [MLIR](https://mlir.llvm.org).



