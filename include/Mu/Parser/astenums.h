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

#ifndef _ASTENUMS_H_
#define _ASTENUMS_H_

#include "llvm/Support/raw_ostream.h"

namespace mu {
namespace ast {
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

enum class ConstantType {
  IntKindHex,
  IntKind1,
  IntKind2,
  Char,
  FloatKind1,
  FloaKind2,
  FloatKind3
};

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os, ASTNodeType v) {
  switch (v) {
  case ASTNodeType::ASTNodeList:
    os << "ASTNodeList";
    break;
  case ASTNodeType::UnaryExpr:
    os << "UnaryExpr";
    break;
  case ASTNodeType::BinaryExpr:
    os << "BinaryExpr";
    break;
  case ASTNodeType::IdentifierExpr:
    os << "IdentifierExpr";
    break;
  case ASTNodeType::ConstantExpr:
    os << "ConstantExpr";
    break;
  case ASTNodeType::StringLiteralExpr:
    os << "StringLiteralExpr";
    break;
  case ASTNodeType::CallExpr:
    os << "CallExpr";
    break;
  case ASTNodeType::ParenthesisExpr:
    os << "ParenthesisExpr";
    break;
  case ASTNodeType::CompoundStat:
    os << "CompoundStat";
    break;
  case ASTNodeType::SelectIfStat:
    os << "SelectIfStat";
    break;
  case ASTNodeType::IterationWhileStat:
    os << "IterationWhileStat";
    break;
  case ASTNodeType::JumpReturnStat:
    os << "JumpReturnStat";
    break;
  case ASTNodeType::AssignmentStat:
    os << "AssignmentStat";
    break;
  case ASTNodeType::InitializationStat:
    os << "InitializationStat";
    break;
  case ASTNodeType::ParamDecl:
    os << "ParamDecl";
    break;
  case ASTNodeType::DefunDecl:
    os << "DefunDecl";
    break;
  case ASTNodeType::TranslationUnit:
    os << "TranslationUnit";
    break;
  }
  return os;
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os, Type v) {
  switch (v) {
  case Type::char_mut:
    os << "char_mut";
    break;
  case Type::uint8_mut:
    os << "uint8_mut";
    break;
  case Type::sint8_mut:
    os << "sint8_mut";
    break;
  case Type::uint16_mut:
    os << "uint16_mut";
    break;
  case Type::sint16_mut:
    os << "sint16_mut";
    break;
  case Type::uint32_mut:
    os << "uint32_mut";
    break;
  case Type::sint32_mut:
    os << "sint32_mut";
    break;
  case Type::uint64_mut:
    os << "uint64_mut";
    break;
  case Type::sint64_mut:
    os << "sint64_mut";
    break;
  case Type::float32_mut:
    os << "float32_mut";
    break;
  case Type::float64_mut:
    os << "float64_mut";
    break;
  }
  return os;
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os, ExpressionType v) {
  switch (v) {
  case ExpressionType::Unary:
    os << " unary ";
    break;
  case ExpressionType::Binary:
    os << " binary ";
    break;
  case ExpressionType::Identifier:
    os << " identifier ";
    break;
  case ExpressionType::Constant:
    os << " constant ";
    break;
  case ExpressionType::StringLiteral:
    os << " string_literal ";
    break;
  case ExpressionType::Call:
    os << " call ";
    break;
  case ExpressionType::Parenthesis:
    os << " paren ";
    break;
  }
  return os;
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os, UnaryOp v) {
  switch (v) {
  case UnaryOp::invertOp:
    os << "op: invert ";
    break;
  case UnaryOp::notOp:
    os << "op: not ";
    break;
  case UnaryOp::negOp:
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

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os, ConstantType v) {
  switch (v) {
    case ConstantType::IntKindHex:
      os << "inthex";
      break;
    case ConstantType::IntKind1:
      os << "intkind1";
      break;
    case ConstantType::IntKind2:
      os << "intkind2";
      break;
    case ConstantType::Char:
      os << "char";
      break;
    case ConstantType::FloatKind1:
      os << "floatkind1";
      break;
    case ConstantType::FloaKind2:
      os << "floatkind2";
      break;
    case ConstantType::FloatKind3:
      os << "floatkind3";
      break;
  }
  return os;
}

} // namespace enums
} // namespace ast
} // namespace mu

#endif // _ASTENUMS_H_
