//===- MuOps.cpp - Mu dialect ops -------------------------------*- C++ -*-===//

// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Mu/MuOps.h"
#include "Mu/MuDialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/IR/OpDefinition.h"

#define GET_OP_CLASSES
#include "Mu/MuOps.cpp.inc"

/// A generalized printer for binary operations. It prints in two different
/// forms depending on if all of the types match.
static void printBinaryOp(mlir::OpAsmPrinter &printer, mlir::Operation *op) {
  printer << " " << op->getOperands();
  printer.printOptionalAttrDict(op->getAttrs());
  printer << " : ";

  // If all of the types are the same, print the type directly.
  mlir::Type resultType = *op->result_type_begin();
  if (llvm::all_of(op->getOperandTypes(),
                   [=](mlir::Type type) { return type == resultType; })) {
    printer << resultType;
    return;
  }

  // Otherwise, print a functional type.
  printer.printFunctionalType(op->getOperandTypes(), op->getResultTypes());
}

/// A generalized parser for binary operations. This parses the different forms
/// of 'printBinaryOp' below.
static mlir::ParseResult parseBinaryOp(mlir::OpAsmParser &parser,
                                       mlir::OperationState &result) {
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 2> operands;
  llvm::SMLoc operandsLoc = parser.getCurrentLocation();
  mlir::Type type;
  if (parser.parseOperandList(operands, /*requiredOperandCount=*/2) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(type))
    return mlir::failure();

  // If the type is a function type, it contains the input and result types of
  // this operation.
  if (mlir::FunctionType funcType = llvm::dyn_cast<mlir::FunctionType>(type)) {
    if (parser.resolveOperands(operands, funcType.getInputs(), operandsLoc,
                               result.operands))
      return mlir::failure();
    result.addTypes(funcType.getResults());
    return mlir::success();
  }

  // Otherwise, the parsed type is the type of both operands and results.
  if (parser.resolveOperands(operands, type, result.operands))
    return mlir::failure();
  result.addTypes(type);
  return mlir::success();
}

namespace mlir {
namespace mu {

mlir::ParseResult FuncOp::parse(mlir::OpAsmParser &parser,
                                mlir::OperationState &result) {
  // Dispatch to the FunctionOpInterface provided utility method that parses the
  // function operation.
  auto buildFuncType =
      [](mlir::Builder &builder, llvm::ArrayRef<mlir::Type> argTypes,
         llvm::ArrayRef<mlir::Type> results,
         mlir::function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };

  return mlir::function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false,
      getFunctionTypeAttrName(result.name), buildFuncType,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void FuncOp::print(mlir::OpAsmPrinter &p) {
  // Dispatch to the FunctionOpInterface provided utility method that prints the
  // function operation.
  mlir::function_interface_impl::printFunctionOp(
      p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
      getArgAttrsAttrName(), getResAttrsAttrName());
}

void FuncOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                   llvm::StringRef name, mlir::FunctionType type,
                   llvm::ArrayRef<mlir::NamedAttribute> attrs) {
  // FunctionOpInterface provides a convenient `build` method that will populate
  // the state of our FuncOp, and create an entry block.
  buildWithEntryBlock(builder, state, name, type, attrs, type.getInputs());
}

//===----------------------------------------------------------------------===//
// NegOp
//===----------------------------------------------------------------------===//

void NegOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value innerValue) {
  state.addTypes(innerValue.getType());
  state.addOperands({innerValue});
}

mlir::ParseResult NegOp::parse(mlir::OpAsmParser &parser,
                               mlir::OperationState &result) {
  return parseBinaryOp(parser, result);
}

void NegOp::print(mlir::OpAsmPrinter &p) { printBinaryOp(p, *this); }

//===----------------------------------------------------------------------===//
// NotOp
//===----------------------------------------------------------------------===//

void NotOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value innerValue) {
  state.addTypes(innerValue.getType());
  state.addOperands({innerValue});
}

mlir::ParseResult NotOp::parse(mlir::OpAsmParser &parser,
                               mlir::OperationState &result) {
  return parseBinaryOp(parser, result);
}

void NotOp::print(mlir::OpAsmPrinter &p) { printBinaryOp(p, *this); }

//===----------------------------------------------------------------------===//
// InvertOp
//===----------------------------------------------------------------------===//

void InvertOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value innerValue) {
  state.addTypes(innerValue.getType());
  state.addOperands({innerValue});
}

mlir::ParseResult InvertOp::parse(mlir::OpAsmParser &parser,
                               mlir::OperationState &result) {
  return parseBinaryOp(parser, result);
}

void InvertOp::print(mlir::OpAsmPrinter &p) { printBinaryOp(p, *this); }

//===----------------------------------------------------------------------===//
// ParenOp
//===----------------------------------------------------------------------===//

void ParenOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value innerValue) {
  state.addTypes(innerValue.getType());
  state.addOperands({innerValue});
}

mlir::ParseResult ParenOp::parse(mlir::OpAsmParser &parser,
                               mlir::OperationState &result) {
  return parseBinaryOp(parser, result);
}

void ParenOp::print(mlir::OpAsmPrinter &p) { printBinaryOp(p, *this); }

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

void AddOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs, mlir::Value rhs) {
  state.addTypes(lhs.getType());
  state.addOperands({lhs, rhs});
}

mlir::ParseResult AddOp::parse(mlir::OpAsmParser &parser,
                               mlir::OperationState &result) {
  return parseBinaryOp(parser, result);
}

void AddOp::print(mlir::OpAsmPrinter &p) { printBinaryOp(p, *this); }


//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//

void MulOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs, mlir::Value rhs) {
  state.addTypes(lhs.getType());
  state.addOperands({lhs, rhs});
}

mlir::ParseResult MulOp::parse(mlir::OpAsmParser &parser,
                               mlir::OperationState &result) {
  return parseBinaryOp(parser, result);
}

void MulOp::print(mlir::OpAsmPrinter &p) { printBinaryOp(p, *this); }

//===----------------------------------------------------------------------===//
// SubOp
//===----------------------------------------------------------------------===//

void SubOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs, mlir::Value rhs) {
  state.addTypes(lhs.getType());
  state.addOperands({lhs, rhs});
}

mlir::ParseResult SubOp::parse(mlir::OpAsmParser &parser,
                               mlir::OperationState &result) {
  return parseBinaryOp(parser, result);
}

void SubOp::print(mlir::OpAsmPrinter &p) { printBinaryOp(p, *this); }

//===----------------------------------------------------------------------===//
// DivOp
//===----------------------------------------------------------------------===//

void DivOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs, mlir::Value rhs) {
  state.addTypes(lhs.getType());
  state.addOperands({lhs, rhs});
}

mlir::ParseResult DivOp::parse(mlir::OpAsmParser &parser,
                               mlir::OperationState &result) {
  return parseBinaryOp(parser, result);
}

void DivOp::print(mlir::OpAsmPrinter &p) { printBinaryOp(p, *this); }

///===---------------------------------------------------------------------===//
// ModOp
///===---------------------------------------------------------------------===//

void ModOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs, mlir::Value rhs) {
  state.addTypes(lhs.getType());
  state.addOperands({lhs, rhs});
}

mlir::ParseResult ModOp::parse(mlir::OpAsmParser &parser,
                               mlir::OperationState &result) {
  return parseBinaryOp(parser, result);
}

void ModOp::print(mlir::OpAsmPrinter &p) { printBinaryOp(p, *this); }

///===---------------------------------------------------------------------===//
// AndOp
///===---------------------------------------------------------------------===//

void AndOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs, mlir::Value rhs) {
  state.addTypes(lhs.getType());
  state.addOperands({lhs, rhs});
}

mlir::ParseResult AndOp::parse(mlir::OpAsmParser &parser,
                               mlir::OperationState &result) {
  return parseBinaryOp(parser, result);
}

void AndOp::print(mlir::OpAsmPrinter &p) { printBinaryOp(p, *this); }

///===---------------------------------------------------------------------===//
// OrOp
///===---------------------------------------------------------------------===//

void OrOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs, mlir::Value rhs) {
  state.addTypes(lhs.getType());
  state.addOperands({lhs, rhs});
}

mlir::ParseResult OrOp::parse(mlir::OpAsmParser &parser,
                               mlir::OperationState &result) {
  return parseBinaryOp(parser, result);
}

void OrOp::print(mlir::OpAsmPrinter &p) { printBinaryOp(p, *this); }

///===---------------------------------------------------------------------===//
// OrBoolOp
///===---------------------------------------------------------------------===//

void OrBoolOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs, mlir::Value rhs) {
  state.addTypes(builder.getI1Type());
  state.addOperands({lhs, rhs});
}

mlir::ParseResult OrBoolOp::parse(mlir::OpAsmParser &parser,
                               mlir::OperationState &result) {
  return parseBinaryOp(parser, result);
}

void OrBoolOp::print(mlir::OpAsmPrinter &p) { printBinaryOp(p, *this); }

///===---------------------------------------------------------------------===//
// AndBoolOp
///===---------------------------------------------------------------------===//

void AndBoolOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs, mlir::Value rhs) {
  state.addTypes(builder.getI1Type());
  state.addOperands({lhs, rhs});
}

mlir::ParseResult AndBoolOp::parse(mlir::OpAsmParser &parser,
                               mlir::OperationState &result) {
  return parseBinaryOp(parser, result);
}

void AndBoolOp::print(mlir::OpAsmPrinter &p) { printBinaryOp(p, *this); }


///===---------------------------------------------------------------------===//
/// IfOp
/// ===--------------------------------------------------------------------===//



void IfOp::getSuccessorRegions(RegionBranchPoint point,
                               SmallVectorImpl<RegionSuccessor> &regions) {
  // assert((point.isParent() || point == getRegion()) && "expected loop region");
  // // The loop may typically branch back to its body or to the parent operation.
  // // If the predecessor is the parent op and the trip count is known to be at
  // // least one, branch into the body using the iterator arguments. And in cases
  // // we know the trip count is zero, it can only branch back to its parent.
  // std::optional<uint64_t> tripCount = getTrivialConstantTripCount(*this);
  // if (point.isParent() && tripCount.has_value()) {
  //   if (tripCount.value() > 0) {
  //     regions.push_back(RegionSuccessor(&getRegion(), getRegionIterArgs()));
  //     return;
  //   }
  //   if (tripCount.value() == 0) {
  //     regions.push_back(RegionSuccessor(getResults()));
  //     return;
  //   }
  // }

  // // From the loop body, if the trip count is one, we can only branch back to
  // // the parent.
  // if (!point.isParent() && tripCount && *tripCount == 1) {
  //   regions.push_back(RegionSuccessor(getResults()));
  //   return;
  // }

  // // In all other cases, the loop may branch back to itself or the parent
  // // operation.
  // regions.push_back(RegionSuccessor(&getRegion(), getRegionIterArgs()));
  // regions.push_back(RegionSuccessor(getResults()));
}


/// Builds a terminator operation without relying on OpBuilder APIs to avoid
/// cyclic header inclusion.

void IfOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                 // TypeRange resultTypes,
                 mlir::Value cond) {
  state.addOperands(cond);
  OpBuilder::InsertionGuard guard(builder);
  Region *region = state.addRegion();
  builder.createBlock(region);
}

void IfElseOp::getSuccessorRegions(RegionBranchPoint point,
                                   SmallVectorImpl<RegionSuccessor> &regions) {
}


/// Builds a terminator operation without relying on OpBuilder APIs to avoid
/// cyclic header inclusion.
void IfElseOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                     mlir::Value cond) {
  state.addOperands(cond);
  OpBuilder::InsertionGuard guard(builder);
  Region *region = state.addRegion();
  builder.createBlock(region);
}

} // namespace mu
} // namespace mlir
