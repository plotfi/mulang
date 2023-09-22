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

def ReturnOp : Mu_Op<"return", [Pure, HasParent<"FuncOp">,
                                 Terminator]> {
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
  let arguments = (ins Variadic<Mu_Type>:$input);

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

def NegOp : Mu_Op<"neg", [Pure, SameTypeOperands, Elementwise, Scalarizable]> {
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

#endif // MU_OPS
