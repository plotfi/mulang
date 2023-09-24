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
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <numeric>

#include "Mu/MuDialect.h"
#include "Mu/MuMLIRGen.h"
#include "Mu/Parser/ast.h"
#include "Mu/Parser/astenums.h"
#include "mlir/Support/LogicalResult.h"

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
  llvm::ScopedHashTable<StringRef,
                        std::pair<mlir::Value, const mu::ast::NamedDecl *>>
      symbolTable;
  using SymbolTableScopeT = llvm::ScopedHashTableScope<
      StringRef, std::pair<mlir::Value, const mu::ast::NamedDecl *>>;

  /// Helper conversion for a Mu AST location to an MLIR location.
  mlir::Location loc(const mu::ast::Location &loc) {
    return mlir::FileLineColLoc::get(builder.getStringAttr(loc.file), loc.line,
                                     loc.col);
  }

  /// Declare a variable in the current scope, return success if the variable
  /// wasn't declared yet.
  mlir::LogicalResult declare(const mu::ast::NamedDecl &param,
                              mlir::Value value) {
    if (symbolTable.count(param.getName()))
      return mlir::failure();
    symbolTable.insert(param.getName(), {value, &param});
    return mlir::success();
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
    SymbolTableScopeT varScope(symbolTable);

    // Create an MLIR function for the given prototype.
    builder.setInsertionPointToEnd(theModule.getBody());
    mlir::mu::FuncOp function = mlirGenFunctionProto(funcAST);
    if (!function) {
      return nullptr;
    }

    // Let's start the body of the function now!
    mlir::Block &entryBlock = function.front();

    if (funcAST.hasParams()) {
      auto &protoArgs = funcAST.getParams();

      // Declare all the function arguments in the symbol table.
      for (const auto nameValue :
      llvm::zip(protoArgs, entryBlock.getArguments())) {
        if (failed(declare(*std::get<0>(nameValue), std::get<1>(nameValue))))
          return nullptr;
      }
    }

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

                                  getType(funcAST.getReturnType())));
      /// *returnOp.operand_type_begin()));
    }

    // If this function isn't main, then set the visibility to private.
    if (funcAST.getName() != "main")
      function.setPrivate();

    return function;
  }

  /// Codegen a list of expression, return failure if one of them hit an error.
  mlir::LogicalResult mlirGen(const mu::ast::CompoundStatement &body) {
    for (const auto stmt : body) {
      if (const auto s = dyn_cast<mu::ast::JumpReturnStatement>(stmt);
        s && mlirGen(*s).failed()) {
        return mlir::failure();
      } else if (const auto s = dyn_cast<ast::InitializationStatement>(stmt);
                 s && mlirGen(*s).failed()) {
        return mlir::failure();
      }
    }
    return mlir::success();
  }


  /// Handle a variable declaration, we'll codegen the expression that forms the
  /// initializer and record the value in the symbol table before returning it.
  /// Future expressions will be able to reference this variable through symbol
  /// table lookup.
  mlir::LogicalResult mlirGen(const mu::ast::InitializationStatement &initDecl) {
    const auto init = initDecl.getExpression();

    mlir::Value value = mlirGen(*init);
    if (!value) {
      return mlir::failure();
    }

    // Handle the case where we are initializing a struct value.
    auto initType = initDecl.getType();

    // Check that the initializer type is the same as the variable
    // declaration.
    mlir::Type type = getType(initType);
    if (!type) {
      return mlir::failure();
    }

    if (type != value.getType()) {
      emitError(loc(initDecl.getLocation()))
        << "type of initializer is different than the value given. Got "
        << value.getType() << ", but expected " << type;
      return mlir::failure();
    }

    // Register the value in the symbol table.
    if (failed(declare(initDecl, value))) {
      return mlir::failure();
    }

    return mlir::success();
  }

  /// Emit a return operation. This will return failure if any generation fails.
  mlir::LogicalResult mlirGen(const mu::ast::JumpReturnStatement &ret) {
    auto location = loc(ret.getLocation());

    // 'return' takes an optional expression, handle that case here.
    mlir::Value expr = nullptr;
    if (ret.hasExpression()) {
      if (!(expr = mlirGen(*ret.getExpression()))) {
        return mlir::failure();
      }
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
    case mu::ast::enums::ExpressionType::Binary:
      return mlirGen(cast<mu::ast::BinaryExpression>(expr));
    case mu::ast::enums::ExpressionType::Constant:
      return mlirGen(cast<mu::ast::ConstantExpression>(expr));
    case mu::ast::enums::ExpressionType::Identifier:
      return mlirGen(cast<mu::ast::IdentifierExpression>(expr));
    case mu::ast::enums::ExpressionType::Parenthesis:
      return mlirGen(cast<mu::ast::ParenthesisExpression>(expr));
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

  /// Emit a paren operation
  mlir::Value mlirGen(const mu::ast::ParenthesisExpression &parenExpr) {
    mlir::Value innerValue = mlirGen(parenExpr.getInternalExpression());
    auto location = loc(parenExpr.getLocation());
    if (!innerValue) {
      return nullptr;
    }
    return builder.create<ParenOp>(location, innerValue);
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
      case mu::ast::enums::UnaryOp::notOp: {
        return builder.create<NotOp>(location, innerValue);
      }
      case mu::ast::enums::UnaryOp::invertOp: {
        return builder.create<InvertOp>(location, innerValue);
      }
    }

    std::string str;
    llvm::raw_string_ostream sstr(str);
    sstr << "invalid unary operator " << unaryOp.getOp();
    llvm_unreachable(str.c_str());
  }

  /// Emit a binary operation
  mlir::Value mlirGen(const mu::ast::BinaryExpression &binaryOp) {
    // First emit the operations for each side of the operation before emitting
    // the operation itself. For example if the expression is `a + foo(a)`
    // 1) First it will visiting the LHS, which will return a reference to the
    //    value holding `a`. This value should have been emitted at declaration
    //    time and registered in the symbol table, so nothing would be
    //    codegen'd. If the value is not in the symbol table, an error has been
    //    emitted and nullptr is returned.
    // 2) Then the RHS is visited (recursively) and a call to `foo` is emitted
    //    and the result value is returned. If an error occurs we get a nullptr
    //    and propagate.
    //
    mlir::Value lhs = mlirGen(binaryOp.getLHS());
    if (!lhs)
      return nullptr;
    auto location = loc(binaryOp.getLocation());

    // Otherwise, this is a normal binary op.
    mlir::Value rhs = mlirGen(binaryOp.getRHS());
    if (!rhs)
      return nullptr;

    // Derive the operation name from the binary operator. At the moment we only
    // support '+' and '*'.
    switch (binaryOp.getBinaryOp()) {
    case  mu::ast::enums::BinaryOp::addOp:
      return builder.create<AddOp>(location, lhs, rhs);
    case  mu::ast::enums::BinaryOp::mulOp:
      return builder.create<MulOp>(location, lhs, rhs);
    case mu::ast::enums::BinaryOp::subOp:
      return builder.create<SubOp>(location, lhs, rhs);
    case mu::ast::enums::BinaryOp::divOp:
      return builder.create<DivOp>(location, lhs, rhs);
    case mu::ast::enums::BinaryOp::modOp:
      return builder.create<ModOp>(location, lhs, rhs);
    case mu::ast::enums::BinaryOp::andOp:
      return builder.create<AndOp>(location, lhs, rhs);
    case mu::ast::enums::BinaryOp::orOp:
      return builder.create<OrOp>(location, lhs, rhs);
    case mu::ast::enums::BinaryOp::orbOp:
      return builder.create<OrBoolOp>(location, lhs, rhs);
    case mu::ast::enums::BinaryOp::andbOp:
      return builder.create<AndBoolOp>(location, lhs, rhs);
    default:
      break;
    }

    std::string str;
    llvm::raw_string_ostream sstr(str);
    sstr << binaryOp.getBinaryOp();
    emitError(location, "invalid binary operator '") << str << "'";

    llvm_unreachable(str.c_str());
  }

  /// This is a reference to a variable in an expression. The variable is
  /// expected to have been declared and so should have a value in the symbol
  /// table, otherwise emit an error and return nullptr.
  mlir::Value mlirGen(const mu::ast::IdentifierExpression &expr) {
    if (auto variable = symbolTable.lookup(expr.getName()).first)
      return variable;

    emitError(loc(expr.getLocation()), "error: unknown variable '")
        << expr.getName() << "'";
    return nullptr;
  }

  /// Emit a constant for a single number (FIXME: semantic? broadcast?)
  mlir::Value mlirGen(const mu::ast::ConstantExpression &num) {
    return builder.create<ConstantOp>(loc(num.getLocation()),
                                      getType(num.getType()),
                                      num.getValueAsInt()); // TODO: Fix
  }

  /// Build an MLIR type from a Mu AST variable type (forward to the generic
  /// getType above).
  [[nodiscard]] auto getType(const mu::ast::enums::Type type) -> mlir::Type {
    switch (type) {
    case mu::ast::enums::Type::char_mut:
      break;
    case mu::ast::enums::Type::uint8_mut:
      break;
    case mu::ast::enums::Type::sint8_mut:
      break;
    case mu::ast::enums::Type::uint16_mut:
      break;
    case mu::ast::enums::Type::sint16_mut:
      break;
    case mu::ast::enums::Type::uint32_mut:
      break;
    case mu::ast::enums::Type::sint32_mut:
      return builder.getI32Type();
    case mu::ast::enums::Type::uint64_mut:
      break;
    case mu::ast::enums::Type::sint64_mut:
      break;
    case mu::ast::enums::Type::float32_mut:
      break;
    case mu::ast::enums::Type::float64_mut:
      break;
    case mu::ast::enums::Type::bool_mut:
      return builder.getI1Type();
    }

    llvm_unreachable("unexpected type given to getType()");
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
