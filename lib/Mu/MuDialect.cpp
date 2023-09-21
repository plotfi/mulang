//===- MuDialect.cpp - Mu dialect -------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Mu/MuDialect.h"
#include "Mu/MuOps.h"
#include "Mu/MuTypes.h"

using namespace mlir;
using namespace mlir::mu;

#include "Mu/MuOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Mu dialect.
//===----------------------------------------------------------------------===//

void MuDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Mu/MuOps.cpp.inc"
      >();
  registerTypes();
}

namespace mlir {
namespace mu {

mlir::LogicalResult ReturnOp::verify() {
  // We know that the parent operation is a function, because of the 'HasParent'
  // trait attached to the operation definition.
  auto function = cast<FuncOp>((*this)->getParentOp());

  /// ReturnOps can only have a single optional operand.
  if (getNumOperands() > 1)
    return emitOpError() << "expects at most 1 return operand";

  // The operand number and types must match the function signature.
  const auto &results = function.getFunctionType().getResults();
  if (getNumOperands() != results.size())
    return emitOpError() << "does not return the same number of values ("
                         << getNumOperands() << ") as the enclosing function ("
                         << results.size() << ")";

  // If the operation does not have an input, we are done.
  if (!hasOperand())
    return mlir::success();

  auto inputType = *operand_type_begin();
  auto resultType = results.front();

  // Check that the result type of the function matches the operand type.
  if (inputType == resultType)
    return mlir::success();

  return emitError() << "type of return operand (" << inputType
                     << ") doesn't match function result type (" << resultType
                     << ")";
}

} // namespace  mu
} // namespace  mlir
