//===- MLIRGen.h - MLIR Generation from a Mu AST --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares a simple interface to perform IR generation targeting MLIR
// from a TranslationUnit AST (aka Module AST) for the Mu language.
//
//===----------------------------------------------------------------------===//

#ifndef MU_MLIRGEN_H
#define MU_MLIRGEN_H

#include "Mu/Parser/ast.h"
#include <memory>

namespace mlir {
class MLIRContext;
template <typename OpTy>
class OwningOpRef;
class ModuleOp;
} // namespace mlir

namespace mu {
/// Emit IR for the given Mu TranslationUnit AST, returns a newly created MLIR
/// module or nullptr on failure.
mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &context,
                                          mu::ast::TranslationUnit &moduleAST);
} // namespace mu

#endif // MU_MLIRGEN_H
