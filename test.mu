
/// Syntax Version A:
define foo = lambda (a: int32) -> int {
  return a + 42
}

type int = i32

/// Syntax Version B:
function foo (b: int) -> int {
  let a : int = 42;
  return a + b;
}

/// Structs and typedefs?
define _S = struct {
  a: int32
  b: float32
}

define S = type _S

/// Note: Considering dropping structs for now

