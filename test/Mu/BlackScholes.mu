// RUN: muc --emit=ast %s

// CHECK: (defun name: BlackScholes, type: float32_mut
// CHECK: (parameter ParamDecl name: S, varType: float32_mut )
// CHECK: (parameter ParamDecl name: X, varType: float32_mut )
// CHECK: (parameter ParamDecl name: T, varType: float32_mut )
// CHECK: (parameter ParamDecl name: r, varType: float32_mut )
// CHECK: (parameter ParamDecl name: v, varType: float32_mut )

fn BlackScholes(S: float32, X: float32, T: float32, r: float32, v: float32) -> float32 {
  var d1: float32 = (log(S / X) + (r + v * v / 2) * T) / (v * sqrt(T));
  var d2: float32 = d1 - v * sqrt(T);
  return S * cnd(d1) - X * exp(-r * T) * cnd(d2);
}

fn cnd(X: float32) -> float32 {
  var L: float32 = 0.0;
  var K: float32 = 0.0;
  var dCND: float32 = 0.0;
  var a1: float32 = 0.31938153;
  var a2: float32 = -0.356563782;
  var a3: float32 = 1.781477937;
  var a4: float32 = -1.821255978;
  var a5: float32 = 1.330274429;
  var RSQRT2PI: float32 = 0.39894228040143267793994605993438;
  var absX: float32 = abs(X);
  var t: float32 = 1.0 / (1.0 + 0.2316419 * absX);
  var y: float32 = 1.0 - RSQRT2PI * exp(-0.5 * X * X) * t
    * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))));
  if (X > 0.0) {
    dCND = y;
  } else {
    dCND = 1.0 - y;
  }
  return dCND;
}

fn fact(n: int32) -> float32 {
  var result: float32 = 1.0;
  var i: int32 = 1;
  while i <= n {
    result = result * i;
    i = i + 1;
  }
  return result;
}

fn pow(x: float32, n: int32) -> float32 {
  var result: float32 = 1.0;
  var i: int32 = 1;
  while i <= n {
    result = result * x;
    i = i + 1;
  }
  return result;
}

fn qnorm(x: float32) -> float32 {
  var result: float32 = 0.0;
  var term: float32 = x;
  var termSquared: float32 = term * term;
  var i: int32 = 1;
  while term != result {
    result = term;
    term = term + pow(x, i) / fact(i);
    i = i + 1;
  }
  return result;
}

fn phi(x: float32) -> float32 {
  var result: float32 = 0.0;
  var term: float32 = x;
  var termSquared: float32 = term * term;
  var i: int32 = 1;
  while term != result {
    result = term;
    term = term + pow(x, i) / fact(i);
    i = i + 1;
  }
  return result;
}

fn main() -> int32 {
  var S: float32 = 100.0;
  var X: float32 = 100.0;
  var T: float32 = 0.25;
  var r: float32 = 0.02;
  var v: float32 = 0.30;
  var call: float32 = BlackScholes(S, X, T, r, v);
  return 0;
}

fn abs(x: float32) -> float32 {
  if (x < 0.0) {
    return -x;
  } else {
    return x;
  }
}

fn log(x: float32) -> float32 {
  var result: float32 = 0.0;
  var term: float32 = (x - 1) / (x + 1);
  var termSquared: float32 = term * term;
  var denom: float32 = 1.0;
  var frac: float32 = term;
  var i: int32 = 1;
  while (frac != 0.0) {
    result = result + frac / denom;
    denom = denom + 2.0;
    frac = frac * termSquared;
    i = i + 1;
  }
  return 2.0 * result;
}

fn sqrt(x: float32) -> float32 {
  var result: float32 = 0.0;
  var term: float32 = x;
  var termSquared: float32 = term * term;
  var i: int32 = 1;
  while term != result {
    result = term;
    term = (termSquared + x) / (2.0 * term);
    termSquared = term * term;
    i = i + 1;
  }
  return result;
}

fn exp(x: float32) -> float32 {
  var result: float32 = 0.0;
  var term: float32 = 1.0;
  var i: int32 = 1;
  while term != result {
    result = term;
    term = term + pow(x, i) / fact(i);
    i = i + 1;
  }
  return result;
}


