# µ-lang

A language just for fun.

The grammar was orignally taken from a C grammer at https://www.lysator.liu.se/c but has since been heavily modified.
The only C parts left are mostly from the lexer and from the binary expression parsing.

The Mu (muc) compiler uses [MLIR](https://mlir.llvm.org).

## Syntax

```mu

typealias int = int32;

// interative fib function. No arrays so no memoization yet.
fn fibonacci(n: int) -> int {

  if n < 2 {
    return n;
  }

  var prev: int = 0;
  var current: int = 1;
  var i: int = 2;

  /*
     To Keep the language simple, while loops are the only kind of loops.
     Theres also no increment operator since using x = x + 1 does the same job.
  */ 
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

  // Still have no idea how I will make strings or printing work.
  _ = printf("fibonacci of 5 is: %d\n", fib);

  return 0;
}
```


## Build Steps

First, build LLVM like so:

```bash
mkdir ~/opt/dev   # I like to develop projects here
git clone https://github.com/llvm/llvm-project
cd llvm-project

cmake -GNinja -DCMAKE_BUILD_TYPE=Release \
              -DLLVM_ENABLE_ASSERTIONS=ON \
              -DLLVM_BUILD_EXAMPLES=ON \
              -DLLVM_TARGETS_TO_BUILD="Native" \
              -DLLVM_ENABLE_PROJECTS="clang;mlir" \
              -DLLVM_INSTALL_UTILS=ON \
              -B./build \
              -DCMAKE_INSTALL_PREFIX=./destdir \
              ./llvm

ninja -C ./build
ninja -C ./build install
```

Next, checkout mulang and build like so:

```bash
cd ~/opt/dev
git clone https://github.com/plotfi/mulang

export LLVM_DIR=$HOME/opt/dev/llvm-project
make
```
