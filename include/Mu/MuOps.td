//===- MuOps.td - Mu dialect ops ---------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MU_OPS
#define MU_OPS

include "Mu/MuTypes.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

//===----------------------------------------------------------------------===//
// Mu Operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

def FuncOp : Mu_Op<"func", [
    FunctionOpInterface, IsolatedFromAbove
  ]> {
  let summary = "user defined function operation";
  let description = [{
    The "mu.func" operation represents a user defined function. These are
    callable SSA-region operations that contain mu computations.

    Example:

    ```mlir
    mu.func @main() {
      %0 = mu.op1 to mutype1
      %1 = mu.op2 to mutype2
      mu.print %1 : mutype2
      mu.return
    }
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );
  let regions = (region AnyRegion:$body);

  let builders = [OpBuilder<(ins
    "StringRef":$name, "FunctionType":$type,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)
  >];
  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    Region *getCallableRegion() { return &getBody(); }
  }];
  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders = 1;
}

def ReturnOp : Mu_Op<"return", [Pure, ParentOneOf<["FuncOp", "IfOp"]>, Terminator]> {
  let summary = "return operation";
  let description = [{
    The "return" operation represents a return operation within a function.
    The operation takes an optional operand and produces no results.
    The operand type must match the signature of the function that contains
    the operation. For example:

    ```mlir
      mu.func @foo() -> sometype {
        ...
        mu.return %0 : sometype
      }
    ```
  }];

  // The return operation takes an optional input operand to return. This
  // value must match the return type of the enclosing function.
  let arguments = (ins Variadic<Mu_Any>:$input);

  // The return operation only emits the input in the format if it is present.
  let assemblyFormat = "($input^ `:` type($input))? attr-dict ";

  // Allow building a ReturnOp with no return operand.
  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, std::nullopt); }]>
  ];

  // Provide extra utility definitions on the c++ operation class definition.
  let extraClassDeclaration = [{
    bool hasOperand() { return getNumOperands() != 0; }
  }];

  // Indicate that additional verification for this operation is necessary.
  let hasVerifier = 1;
}

def BreakOp : Mu_Op<"break", [Pure, ParentOneOf<["IfOp"]>, Terminator]> {
  let summary = "break operation";
  let description = [{
    The "break" operation represents a break operation within a function.
    The operation takes an optional operand and produces no results.
    The operand type must match the signature of the function that contains
    the operation.The break op is internal to MLIR Mu to satisfy the requirement
    of the IfOp and WhileOp terminator.
  }];

  // Allow building a BreakOp with no return operand.
  let builders = [
  ];

  // Indicate that additional verification for this operation is necessary.
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

// We define a mu operation by inheriting from our base 'Mu_Op' class above.
// Here we provide the mnemonic and a list of traits for the operation. The
// constant operation is marked as 'Pure' as it is a pure operation
// and may be removed if dead.
def ConstantOp
    : Mu_Op<"mlir.constant", [ConstantLike, Pure]>
{
  // Provide a summary and description for this operation. This can be used to
  // auto-generate documentation of the operations within our dialect.
  let summary = "constant";
  let description = [{
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

    ```mlir
      %0 = mu.constant 42 + int32
    ```
  }];

  let arguments = (ins AnyAttr:$value);
  let results = (outs Mu_Type:$res);

  let assemblyFormat = "`(` $value `)` attr-dict `:` type($res)";
  // Indicate that the operation has a custom parser and printer method.
  // let hasCustomAssemblyFormat = 1;

  let builders = [
    OpBuilder<(ins "Type":$type, "int32_t":$value), [{
      build($_builder, $_state, type, $_builder.getIntegerAttr(type, value));
    }]>,
    OpBuilder<(ins "Type":$type, "int64_t":$value), [{
      build($_builder, $_state, type, $_builder.getIntegerAttr(type, value));
    }]>,
    OpBuilder<(ins "Type":$type, "const APInt &":$value), [{
      build($_builder, $_state, type, $_builder.getIntegerAttr(type, value));
    }]>,
    OpBuilder<(ins "Type":$type, "const APFloat &":$value), [{
      build($_builder, $_state, type, $_builder.getFloatAttr(type, value));
    }]>,
    OpBuilder<(ins "TypedAttr":$value), [{
      build($_builder, $_state, value.getType(), value);
    }]>
  ];

  let extraClassDeclaration = [{
    /// Whether the constant op can be constructed with a particular value and
    /// type.
    static bool isBuildableWith(Attribute value, Type type);

    /// Build the constant op with `value` and `type` if possible, otherwise
    /// returns null.
    static ConstantOp materialize(OpBuilder &builder, Attribute value,
                                  Type type, Location loc);
  }];

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// NegOp
//===----------------------------------------------------------------------===//

def NegOp : Mu_Op<"neg", [Pure]> {
  let summary = "unary negation operation";
  let description = [{
    The "neg" operation makes a numeric value negative.
  }];

  let arguments = (ins Mu_Type:$value);
  let results = (outs Mu_Type:$res);

  // Indicate that the operation has a custom parser and printer method.
  let hasCustomAssemblyFormat = 1;

  // Allow building an NegOp from one input expr
  let builders = [
    OpBuilder<(ins "Value":$innerExpr)>
  ];
}

//===----------------------------------------------------------------------===//
// NotOp
//===----------------------------------------------------------------------===//

def NotOp : Mu_Op<"not", [Pure]> {
  let summary = "unary not operation";
  let description = [{
    The "not" operation makes a numeric value boolean inverted.
    If value is not 0 it becomes 0, if it is 0 it becomes 1.
  }];

  let arguments = (ins Mu_Type:$value);
  let results = (outs Mu_Type:$res);

  // Indicate that the operation has a custom parser and printer method.
  let hasCustomAssemblyFormat = 1;

  // Allow building an NotOp from one input expr
  let builders = [
    OpBuilder<(ins "Value":$innerExpr)>
  ];
}

//===----------------------------------------------------------------------===//
// InvertOp
//===----------------------------------------------------------------------===//

def InvertOp : Mu_Op<"invert", [Pure]> {
  let summary = "unary invert operation";
  let description = [{
    The "invert" operation makes a numeric value bitwise inverted.
  }];

  let arguments = (ins Mu_Type:$value);
  let results = (outs Mu_Type:$res);

  // Indicate that the operation has a custom parser and printer method.
  let hasCustomAssemblyFormat = 1;

  // Allow building an InvertOp from one input expr
  let builders = [
    OpBuilder<(ins "Value":$innerExpr)>
  ];
}

//===----------------------------------------------------------------------===//
// ParenOp
//===----------------------------------------------------------------------===//

def ParenOp : Mu_Op<"paren", [Pure]> {
  let summary = "paren operation";
  let description = [{
    The "paren" operation passes any value through, it just helps set operator
    precedents.
  }];

  let arguments = (ins Mu_Type:$value);
  let results = (outs Mu_Type:$res);

  // Indicate that the operation has a custom parser and printer method.
  let hasCustomAssemblyFormat = 1;

  // Allow building an ParenOp from one input expr
  let builders = [
    OpBuilder<(ins "Value":$innerExpr)>
  ];
}

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

def AddOp : Mu_Op<"add", [Pure, Commutative]> {
  let summary = "Add operation for two values.";
  let description = [{
    The "add" operation performs addition between two Mu types.
  }];

  let arguments = (ins Mu_Type:$lhs, Mu_Type:$rhs);
  let results = (outs Mu_Type:$res);

  // Indicate that the operation has a custom parser and printer method.
  let hasCustomAssemblyFormat = 1;

  // Allow building an AddOp with from the two input operands.
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];
}

//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//

def MulOp : Mu_Op<"mul", [Pure, Commutative]> {
  let summary = "Multiply operation for two values.";
  let description = [{
    The "mul" operation performs multiplication between two Mu types.
  }];

  let arguments = (ins Mu_Type:$lhs, Mu_Type:$rhs);
  let results = (outs Mu_Type:$res);

  // Indicate that the operation has a custom parser and printer method.
  let hasCustomAssemblyFormat = 1;

  // Allow building a MulOp with from the two input operands.
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];
}

def DivOp : Mu_Op<"div", [Pure]> {
  let summary = "Divide operation for two values.";
  let description = [{
    The "div" operation performs division between two Mu types.
  }];

  let arguments = (ins Mu_Type:$lhs, Mu_Type:$rhs);
  let results = (outs Mu_Type:$res);

  // Indicate that the operation has a custom parser and printer method.
  let hasCustomAssemblyFormat = 1;

  // Allow building a DivOp with from the two input operands.
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];
}

def ModOp : Mu_Op<"mod", [Pure]> {
  let summary = "Modulo operation for two values.";
  let description = [{
    The "mod" operation performs modulo between two Mu types.
  }];

  let arguments = (ins Mu_Type:$lhs, Mu_Type:$rhs);
  let results = (outs Mu_Type:$res);
  // Indicate that the operation has a custom parser and printer method.
  let hasCustomAssemblyFormat = 1;
  // Allow building a ModOp with from the two input operands.
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];
}

def AndOp : Mu_Op<"and", [Pure, Commutative]> {
  let summary = "And operation for two values.";
  let description = [{
    The "and" operation performs bitwise and between two Mu types.
  }];

  let arguments = (ins Mu_Type:$lhs, Mu_Type:$rhs);
  let results = (outs Mu_Type:$res);

  // Indicate that the operation has a custom parser and printer method.
  let hasCustomAssemblyFormat = 1;
  // Allow building an AndOp with from the two input operands.
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];
}

def OrOp : Mu_Op<"or", [Pure, Commutative]> {
  let summary = "Or operation for two values.";
  let description = [{
    The "or" operation performs bitwise or between two Mu types.
  }];

  let arguments = (ins Mu_Type:$lhs, Mu_Type:$rhs);
  let results = (outs Mu_Type:$res);

  // Indicate that the operation has a custom parser and printer method.
  let hasCustomAssemblyFormat = 1;
  // Allow building an OrOp with from the two input operands.
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];
}

// TODO: Need to add a bool type to this language.
def OrBoolOp : Mu_Op<"orb", [Pure, Commutative]> {
  let summary = "Boolean or operation for two values.";
  let description = [{
    The "orb" operation performs boolean or between two Mu types.
  }];

  let arguments = (ins Mu_Any:$lhs, Mu_Any:$rhs);
  let results = (outs Mu_Bool:$res);
  // Indicate that the operation has a custom parser and printer method.
  let hasCustomAssemblyFormat = 1;
  // Allow building an OrBoolOp with from the two input operands.
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];
}

// TODO: Need to add a bool type to this language.
def AndBoolOp : Mu_Op<"andb" , [Pure, Commutative]> {
  let summary = "Boolean and operation for two values.";
  let description = [{
    The "andb" operation performs boolean and between two Mu types.
  }];

  let arguments = (ins Mu_Any:$lhs, Mu_Any:$rhs);
  let results = (outs Mu_Bool:$res);
  // Indicate that the operation has a custom parser and printer method.
  let hasCustomAssemblyFormat = 1;
  // Allow building an AndBoolOp with from the two input operands.
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];
}

/*
def XorOp : Mu_Op<"xor", [Pure, Commutative]> {
  let summary = "Xor operation for two values.";
  let description = [{
    The "xor" operation performs bitwise xor between two Mu types.
  }];

  let arguments = (ins Mu_Type:$lhs, Mu_Type:$rhs);
  let results = (outs Mu_Type:$res);

  // Indicate that the operation has a custom parser and printer method.
  let hasCustomAssemblyFormat = 1;
  // Allow building an XorOp with from the two input operands.
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];
}

def ShlOp : Mu_Op<"shl", [Pure]> {
  let summary = "Shift left operation for two values.";
  let description = [{
    The "shl" operation performs bitwise shift left between two Mu types.
  }];

  let arguments = (ins Mu_Type:$lhs, Mu_Type:$rhs);
  let results = (outs Mu_Type:$res);
  // Indicate that the operation has a custom parser and printer method.
  let hasCustomAssemblyFormat = 1;
  // Allow building a ShlOp with from the two input operands.
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];
}

def ShrOp : Mu_Op<"shr", [Pure]> {
  let summary = "Shift right operation for two values.";
  let description = [{
    The "shr" operation performs bitwise shift right between two Mu types.
  }];

  let arguments = (ins Mu_Type:$lhs, Mu_Type:$rhs);
  let results = (outs Mu_Type:$res);
  // Indicate that the operation has a custom parser and printer method.
  let hasCustomAssemblyFormat = 1;
  // Allow building a ShrOp with from the two input operands.
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];
}

def UShrOp : Mu_Op<"ushr", [Pure]> {
  let summary = "Unsigned shift right operation for two values.";
  let description = [{
    The "ushr" operation performs bitwise unsigned shift right between two Mu
    types.
  }];

  let arguments = (ins Mu_Type:$lhs, Mu_Type:$rhs);
  let results = (outs Mu_Type:$res);
  // Indicate that the operation has a custom parser and printer method.
  let hasCustomAssemblyFormat = 1;
  // Allow building a UShrOp with from the two input operands.
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];
}

*/

def SubOp : Mu_Op<"sub", [Pure]> {
  let summary = "Subtract operation for two values.";
  let description = [{
    The "sub" operation performs subtraction between two Mu types.
  }];

  let arguments = (ins Mu_Type:$lhs, Mu_Type:$rhs);
  let results = (outs Mu_Type:$res);
  // Indicate that the operation has a custom parser and printer method.
  let hasCustomAssemblyFormat = 1;
  // Allow building a SubOp with from the two input operands.
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];
}

def IfOp : Mu_Op<"if", [  DeclareOpInterfaceMethods<RegionBranchOpInterface> ]> {
  let summary = "if operation";
  let description = [{
    The "if" operation is a conditional branch operation. It takes a condition
    and one regions. The region is taken as the true branch. The condition must
    be a boolean type.
  }];

  let arguments = (ins Mu_Bool:$cond);
  let regions = (region AnyRegion:$true_branch);
  let skipDefaultBuilders = 1;

  let builders = [
    OpBuilder<(ins "Value":$cond)>
  ];
}

def IfElseOp : Mu_Op<"ifelse", [  DeclareOpInterfaceMethods<RegionBranchOpInterface> ]> {
  let summary = "ifelse operation";
  let description = [{
    The "ifelse" operation is a conditional branch operation. It takes a condition
    and two regions. The first region is taken as the true branch, the second
    region is taken as the false branch. The condition must be a boolean type.
  }];

  let arguments = (ins Mu_Bool:$cond);
  let regions = (region AnyRegion:$true_branch, AnyRegion:$false_branch);
  let skipDefaultBuilders = 1;

  let builders = [
    OpBuilder<(ins "Value":$cond)>
  ];
}

#endif // MU_OPS
