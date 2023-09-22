//===- MLIRGen.cpp - MLIR Generation from a Mu AST ------00----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a simple IR generation targeting MLIR from a Module AST
// for the Mu language.
//
//===----------------------------------------------------------------------===//

#include "Mu/MuOps.h"
#include "Mu/Parser/astenums.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include <numeric>

#include "Mu/MuDialect.h"
#include "Mu/MuMLIRGen.h"
#include "Mu/Parser/ast.h"
#include "Mu/Parser/astenums.h"

using namespace mlir::mu;
using namespace mu;

using llvm::ArrayRef;
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;
using llvm::ScopedHashTableScope;
using llvm::SmallVector;
using llvm::StringRef;
using llvm::Twine;

namespace {

/// Implementation of a simple MLIR emission from the Mu AST.
///
/// This will emit operations that are specific to the Mu language, preserving
/// the semantics of the language and (hopefully) allow to perform accurate
/// analysis and transformation based on these high level semantics.
struct MLIRGenImpl {
  MLIRGenImpl(mlir::MLIRContext &context) : builder(&context) {}

  /// Public API: convert the AST for a Mu module (source file) to an MLIR
  /// Module operation.
  mlir::ModuleOp mlirGen(mu::ast::TranslationUnit &moduleAST) {
    // We create an empty MLIR module and codegen functions one at a time and
    // add them to the module.
    theModule = mlir::ModuleOp::create(builder.getUnknownLoc());

    for (const mu::ast::Defun *f : moduleAST) {
      mlirGen(*f);
    }

    // Verify the module after we have finished constructing it, this will check
    // the structural properties of the IR and invoke any specific verifiers we
    // have on the Mu operations.
    if (failed(mlir::verify(theModule))) {
      theModule.emitError("module verification error");
      return nullptr;
    }

    return theModule;
  }

private:
  /// A "module" matches a Mu source file: containing a list of functions.
  mlir::ModuleOp theModule;

  /// The builder is a helper class to create IR inside a function. The builder
  /// is stateful, in particular it keeps an "insertion point": this is where
  /// the next operations will be introduced.
  mlir::OpBuilder builder;

  /// The symbol table maps a variable name to a value in the current scope.
  /// Entering a function creates a new scope, and the function arguments are
  /// added to the mapping. When the processing of a function is terminated, the
  /// scope is destroyed and the mappings created in this scope are dropped.
  llvm::ScopedHashTable<StringRef, mlir::Value> symbolTable;

  /// Helper conversion for a Mu AST location to an MLIR location.
  mlir::Location loc(const mu::ast::Location &loc) {
    return mlir::FileLineColLoc::get(builder.getStringAttr(loc.file), loc.line,
                                     loc.col);
  }

  /// Create the prototype for an MLIR function with as many arguments as the
  /// provided Mu AST prototype.
  mlir::mu::FuncOp mlirGenFunctionProto(const mu::ast::Defun &funcAST) {
    auto location = loc(funcAST.getLocation());

    // This is a generic function, the return type will be inferred later.
    // Arguments type are uniformly unranked tensors.

    llvm::SmallVector<mlir::Type, 4> argTypes;
    if (funcAST.hasParams()) {
      for (const auto *param : funcAST.getParams()) {
        argTypes.push_back(getType(param->getType()));
      }
    }

    auto funcType = builder.getFunctionType(argTypes, std::nullopt);
    return builder.create<mlir::mu::FuncOp>(location, funcAST.getName(),
                                            funcType);
  }

  /// Emit a new function and add it to the MLIR module.
  mlir::mu::FuncOp mlirGen(const mu::ast::Defun &funcAST) {
    // Create a scope in the symbol table to hold variable declarations.
    ScopedHashTableScope<llvm::StringRef, mlir::Value> varScope(symbolTable);

    // Create an MLIR function for the given prototype.
    builder.setInsertionPointToEnd(theModule.getBody());
    mlir::mu::FuncOp function = mlirGenFunctionProto(funcAST);
    if (!function)
      return nullptr;

    // Let's start the body of the function now!
    mlir::Block &entryBlock = function.front();

    #if 0
    auto protoArgs = funcAST.getProto()->getArgs();
    // Declare all the function arguments in the symbol table.
    for (const auto nameValue :
         llvm::zip(protoArgs, entryBlock.getArguments())) {
      if (failed(declare(std::get<0>(nameValue)->getName(),
                         std::get<1>(nameValue))))
        return nullptr;
    }
    #endif

    // Set the insertion point in the builder to the beginning of the function
    // body, it will be used throughout the codegen to create operations in this
    // function.
    builder.setInsertionPointToStart(&entryBlock);

    // Emit the body of the function.
    if (mlir::failed(mlirGen(funcAST.getBody()))) {
      function.erase();
      return nullptr;
    }

    // Implicitly return void if no return statement was emitted.
    // FIXME: we may fix the parser instead to always return the last expression
    // (this would possibly help the REPL case later)
    ReturnOp returnOp;
    if (!entryBlock.empty())
      returnOp = dyn_cast<ReturnOp>(entryBlock.back());
    if (!returnOp) {
      builder.create<ReturnOp>(loc(funcAST.getLocation()));
    } else if (returnOp.hasOperand()) {
      // Otherwise, if this return operation has an operand then add a result to
      // the function.
      function.setType(
          builder.getFunctionType(function.getFunctionType().getInputs(),
                                  *returnOp.operand_type_begin()));
    }

    // If this function isn't main, then set the visibility to private.
    if (funcAST.getName() != "main")
      function.setPrivate();

    return function;
  }

  /// Codegen a list of expression, return failure if one of them hit an error.
  mlir::LogicalResult mlirGen(const mu::ast::CompoundStatement &body) {
    for (const auto stmt : body) {
      if (const auto ret = dyn_cast<mu::ast::JumpReturnStatement>(stmt)) {
        return mlirGen(*ret);
      }
    }
    return mlir::success();
  }

  /// Emit a return operation. This will return failure if any generation fails.
  mlir::LogicalResult mlirGen(const mu::ast::JumpReturnStatement &ret) {
    auto location = loc(ret.getLocation());

    // 'return' takes an optional expression, handle that case here.
    mlir::Value expr = nullptr;
    if (ret.hasExpression()) {
      if (!(expr = mlirGen(*ret.getExpression())))
        return mlir::failure();
    }

    // Otherwise, this return operation has zero operands.
    builder.create<ReturnOp>(location,
                             expr ? ArrayRef(expr) : ArrayRef<mlir::Value>());
    return mlir::success();
  }


  /// Dispatch codegen for the right expression subclass using RTTI.
  mlir::Value mlirGen(const mu::ast::Expression &expr) {
    switch (expr.getExpressionKind()) {
    case mu::ast::enums::ExpressionType::Unary:
      return mlirGen(cast<mu::ast::UnaryExpression>(expr));
    // case mu::ast::enums::ExpressionType::Binary:
    //   return mlirGen(cast<mu::ast::BinaryExpression>(expr));
    case mu::ast::enums::ExpressionType::Constant:
      return mlirGen(cast<mu::ast::ConstantExpression>(expr));
    default:
      std::string str;
      llvm::raw_string_ostream sstr(str);
      sstr << expr.getExpressionKind();
      emitError(loc(expr.getLocation()))
          << "MLIR codegen encountered an unhandled expr kind '"
          << str << "' expression.";
      return nullptr;
    }
  }

  /// Emit a unary operation
  mlir::Value mlirGen(const mu::ast::UnaryExpression &unaryOp) {
    mlir::Value innerValue = mlirGen(unaryOp.getInternalExpression());
    auto location = loc(unaryOp.getLocation());
    if (!innerValue)
      return nullptr;

    // Derive the operation name from the binary operator. At the moment we only
    // support '+' and '*'.
    switch (unaryOp.getOp()) {
      case mu::ast::enums::UnaryOp::negOp: {
        return builder.create<NegOp>(location, innerValue);
      }
      // case mu::ast::enums::UnaryOp::notOp:
      // case mu::ast::enums::UnaryOp::invertOp:
      default: {
        std::string str;
        llvm::raw_string_ostream sstr(str);
        sstr <<  unaryOp.getOp();
        emitError(location, "invalid unary operator '") << str << "'";
        return nullptr;
      }
    }
  }

  /// Emit a constant for a single number (FIXME: semantic? broadcast?)
  mlir::Value mlirGen(const mu::ast::ConstantExpression &num) {
    return builder.create<ConstantOp>(loc(num.getLocation()),
                                      builder.getI32Type(),
                                      num.getValueAsInt()); // TODO: Fix
  }

  /// Build an MLIR type from a Mu AST variable type (forward to the generic
  /// getType above).
  [[nodiscard]] auto getType(const mu::ast::enums::Type &type) -> mlir::Type {
    return builder.getI32Type();
  }
};

} // namespace

namespace mu {

// The public API for codegen.
mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &context,
                                          mu::ast::TranslationUnit &moduleAST) {
  return MLIRGenImpl(context).mlirGen(moduleAST);
}

} // namespace mu
