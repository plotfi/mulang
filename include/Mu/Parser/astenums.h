//===- astenums.h - AST Support for Mu ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements AST support for Mu.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/raw_ostream.h"

namespace muast {
namespace enums {

enum class ASTNodeType {
  ASTNodeList,
  UnaryExpr,
  BinaryExpr,
  IdentifierExpr,
  ConstantExpr,
  StringLiteralExpr,
  CallExpr,
  ParenthesisExpr,
  CompoundStat,
  SelectIfStat,
  IterationWhileStat,
  JumpReturnStat,
  AssignmentStat,
  InitializationStat,
  ParamDecl,
  DefunDecl,
  TranslationUnit
};

enum class Type {
  char_mut,
  uint8_mut,
  sint8_mut,
  uint16_mut,
  sint16_mut,
  uint32_mut,
  sint32_mut,
  uint64_mut,
  sint64_mut,
  float32_mut,
  float64_mut
};

enum class ExpressionType {
  Unary,
  Binary,
  Identifier,
  Constant,
  StringLiteral,
  Call,
  Parenthesis,
};

enum class UnaryOp { invertOp, notOp, negOp };

enum class BinaryOp {
  mulOp,
  divOp,
  modOp,
  addOp,
  subOp,
  lshOp,
  rshOp,
  ltOp,
  gtOp,
  leOp,
  geOp,
  eqOp,
  neOp,
  andOp,
  xorOp,
  orOp,
  andbOp,
  orbOp
};

enum class StatementType {
  Compound,
  SelectionIf,
  IterationWhile,
  JumpReturn,
  Assignment,
  Initialization
};

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os, ASTNodeType v) {
  using enum ASTNodeType;
  switch (v) {
  case ASTNodeList:
    os << "ASTNodeList";
    break;
  case UnaryExpr:
    os << "UnaryExpr";
    break;
  case BinaryExpr:
    os << "BinaryExpr";
    break;
  case IdentifierExpr:
    os << "IdentifierExpr";
    break;
  case ConstantExpr:
    os << "ConstantExpr";
    break;
  case StringLiteralExpr:
    os << "StringLiteralExpr";
    break;
  case CallExpr:
    os << "CallExpr";
    break;
  case ParenthesisExpr:
    os << "ParenthesisExpr";
    break;
  case CompoundStat:
    os << "CompoundStat";
    break;
  case SelectIfStat:
    os << "SelectIfStat";
    break;
  case IterationWhileStat:
    os << "IterationWhileStat";
    break;
  case JumpReturnStat:
    os << "JumpReturnStat";
    break;
  case AssignmentStat:
    os << "AssignmentStat";
    break;
  case InitializationStat:
    os << "InitializationStat";
    break;
  case ParamDecl:
    os << "ParamDecl";
    break;
  case DefunDecl:
    os << "DefunDecl";
    break;
  case TranslationUnit:
    os << "TranslationUnit";
    break;
  }
  return os;
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os, Type v) {
  using enum Type;
  switch (v) {
  case char_mut:
    os << "char_mut";
    break;
  case uint8_mut:
    os << "uint8_mut";
    break;
  case sint8_mut:
    os << "sint8_mut";
    break;
  case uint16_mut:
    os << "uint16_mut";
    break;
  case sint16_mut:
    os << "sint16_mut";
    break;
  case uint32_mut:
    os << "uint32_mut";
    break;
  case sint32_mut:
    os << "sint32_mut";
    break;
  case uint64_mut:
    os << "uint64_mut";
    break;
  case sint64_mut:
    os << "sint64_mut";
    break;
  case float32_mut:
    os << "float32_mut";
    break;
  case float64_mut:
    os << "float64_mut";
    break;
  }
  return os;
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os, ExpressionType v) {
  using enum ExpressionType;
  switch (v) {
  case Unary:
    os << " unary ";
    break;
  case Binary:
    os << " binary ";
    break;
  case Identifier:
    os << " identifier ";
    break;
  case Constant:
    os << " constant ";
    break;
  case StringLiteral:
    os << " string_literal ";
    break;
  case Call:
    os << " call ";
    break;
  case Parenthesis:
    os << " paren ";
    break;
  }
  return os;
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os, UnaryOp v) {
  using enum UnaryOp;
  switch (v) {
  case invertOp:
    os << "op: invert ";
    break;
  case notOp:
    os << "op: not ";
    break;
  case negOp:
    os << "op: neg ";
    break;
  }
  return os;
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os, BinaryOp v) {
  switch (v) {
  case BinaryOp::mulOp:
    os << "op: mul ";
    break;
  case BinaryOp::divOp:
    os << "op: div ";
    break;
  case BinaryOp::modOp:
    os << "op: mod ";
    break;
  case BinaryOp::addOp:
    os << "op: add ";
    break;
  case BinaryOp::subOp:
    os << "op: sub ";
    break;
  case BinaryOp::lshOp:
    os << "op: lsh ";
    break;
  case BinaryOp::rshOp:
    os << "op: rsh ";
    break;
  case BinaryOp::ltOp:
    os << "op: lt ";
    break;
  case BinaryOp::gtOp:
    os << "op: gt ";
    break;
  case BinaryOp::leOp:
    os << "op: le ";
    break;
  case BinaryOp::geOp:
    os << "op: ge ";
    break;
  case BinaryOp::eqOp:
    os << "op: eq ";
    break;
  case BinaryOp::neOp:
    os << "op: ne ";
    break;
  case BinaryOp::andOp:
    os << "op: and ";
    break;
  case BinaryOp::xorOp:
    os << "op: xor ";
    break;
  case BinaryOp::orOp:
    os << "op: or ";
    break;
  case BinaryOp::andbOp:
    os << "op: andb ";
    break;
  case BinaryOp::orbOp:
    os << "op: orb ";
    break;
  }
  return os;
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os, StatementType v) {
  switch (v) {
  case StatementType::Compound:
    os << "compound";
    break;
  case StatementType::SelectionIf:
    os << "if";
    break;
  case StatementType::IterationWhile:
    os << "while";
    break;
  case StatementType::JumpReturn:
    os << "return";
    break;
  case StatementType::Assignment:
    os << "assign";
    break;
  case StatementType::Initialization:
    os << "initialize";
    break;
  }
  return os;
}

} // namespace enums
} // namespace muast
