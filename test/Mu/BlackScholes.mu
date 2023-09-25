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