//===- ast.h - AST Support for Mu -----------------------------------------===//
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

#ifndef _AST_H_
#define _AST_H_

#include <cstdlib>
#include <ios>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <vector>

#include "llvm/Support/raw_ostream.h"

#include "Mu/Support/µt8.h"
#include "Mu/Parser/astenums.h"

#define indentStr "  "

namespace mu {
namespace ast {

// These are kinda gross, move them to another Source of Header:
struct ASTNode;
fn inline getASTNodeID(Ref<ASTNode> node) -> unsigned;
fv inline dumpASTNode(Ref<ASTNode> node);
fv inline dumpASTNodeType(Ref<ASTNode> node);
fv clearYYValStorage();

using namespace mu::ast::enums;

struct ASTNodeTracker {
  // explicit ASTNodeTracker(VectorRef<ASTNode> tracked)
  //     : tracked(std::move(tracked)) {}
  ASTNodeTracker(const ASTNodeTracker &) = default;
  ASTNodeTracker(ASTNodeTracker &&) = delete;
  ASTNodeTracker &operator=(const ASTNodeTracker &) = default;
  ASTNodeTracker &operator=(ASTNodeTracker &&) = delete;
  virtual ~ASTNodeTracker() {}

  fn static get() -> const ASTNodeTracker& {
    if (!instance.has_value()) {
      instance = new ASTNodeTracker();
    }
    return *instance.value();
  }

  fv static destroy() {
    #ifndef NDEBUG
    if (instance.has_value() && instance.value()->hasTrackedNodes()) {
      llvm::errs() << "Dumping Tracked Nodes:\n";
      for (unsigned i = 0; i < instance.value()->tracked.size(); i++) {
        if (instance.value()->tracked[i]) {
          llvm::errs() << "[" << i << "] = " << instance.value()->tracked[i] << " ";
          dumpASTNodeType(instance.value()->tracked[i]);
          llvm::errs() << "\n";
          dumpASTNode(instance.value()->tracked[i]);
          llvm::errs() << "\n";
        }
      }
    }
    #endif
    assert(instance && !instance.value()->hasTrackedNodes() &&
           "Expected all nodes to be untracked by dtors");
    delete instance.value();
    instance.reset();
    clearYYValStorage();
  }

  fn size() const -> size_t { return tracked.size(); }
  fv track(Ref<ASTNode> node) const { tracked.push_back(node); }

  fv untrack(Ref<ASTNode> node) const {
    let trackedNode = tracked[getASTNodeID(node)];
    assert(trackedNode == node && "tracked node mismatch!");
    tracked[getASTNodeID(node)] = nullptr;
  }

  fn hasTrackedNodes() const -> bool {
    return !std::all_of(tracked.begin(), tracked.end(),
                        [](Ref<ASTNode> node) { return nullptr == node; });
  }

  fv virtual dump(unsigned indent = 0) const {
    for (let node : tracked) {
      dumpASTNode(node);
    }
  }

  fv virtual dumpNodeByID(unsigned id) const { dumpASTNode(tracked[id]); }

private:
  mutable VectorRef<ASTNode> tracked;
  ASTNodeTracker() {}
  static OptionalRef<ASTNodeTracker> instance;
};

template <typename T>
fn checked_ptr_cast(MutableRef<void> ptr) -> MutableRef<T> {
  var casted = static_cast<MutableRef<T>>(ptr);
  assert(casted->check() && "checked_ptr_cast failed");
  return casted;
}

struct Location {
  std::string file = "main"; // Everything in one file for now: "main"
  unsigned line = 0;         // line number
  unsigned col = 0;          // column number
};

struct ASTNode {
  static const unsigned static_magic_number;
  ASTNode(): tracker(ASTNodeTracker::get()) {
    this->id = tracker.size();
    tracker.track(this);
    // llvm::errs() << "Previous Node: ";
    // if (id) {
    //   tracker.dumpNodeByID(id - 1);
    // }
    // llvm::errs() << "\n";
    // assert(id != 26 && "blow up ASTNode for debug");
    // assert(id != 102 && "blow up ASTNode for debug");
    // llvm::errs() << "create base ASTNode with id: " << id << " !!!\n";
  }
  // explicit ASTNode(const ASTNodeTracker &tracker) : tracker(tracker) {}
  ASTNode(const ASTNode &) = default;
  ASTNode(ASTNode &&) = delete;
  ASTNode &operator=(const ASTNode &) = delete;
  ASTNode &operator=(ASTNode &&) = delete;
  virtual ~ASTNode() {
    #ifndef NDEBUG
    llvm::errs() << "Untracking ID: " << getID() << "\n";
    #endif
    tracker.untrack(this);
  }

  fv dumpNodeInfo() const {
    llvm::errs() << " (ASTNode id: " << id;
    if (location.line > 0) {
      llvm::errs() << ", lineNumber: " << location.line;
    }
    llvm::errs() << ") ";
  }

  fv virtual dump(unsigned indent = 0) const = 0;
  fn getID() const -> unsigned { return id; }
  fn getLocation() const -> CxxRef<Location> { return location; }
  fv setLineNumber(unsigned lineNumber) { location.line = lineNumber; }
  fn virtual getKind() const -> ASTNodeType = 0;
  fn check() const -> bool {
    return magic_number == ASTNode::static_magic_number;
  }

private:
  const ASTNodeTracker &tracker;
  uint id = 0;
  mu::ast::Location location;
  const unsigned magic_number = ASTNode::static_magic_number;
};

fv inline dumpASTNode(Ref<ASTNode> node) { node->dump(); }
fn inline getASTNodeID(Ref<ASTNode> node) -> unsigned { return node->getID(); }
fv inline dumpASTNodeType(Ref<ASTNode> node) {
  llvm::errs() << "ASTNode dump with ID "
               << node->getID()
               << ": "
               << node->getKind();
}

template <typename T, unsigned newlines = 0, bool printAsList = true>
struct ASTList : public ASTNode {
  ASTList() = delete;
  ASTList(const ASTList &) = delete;
  ASTList(ASTList &&) = delete;
  ASTList &operator=(const ASTList &) = delete;
  ASTList &operator=(ASTList &&) = delete;
  ASTList(Ref<T> t) { things.push_back(t); }
  virtual ~ASTList() {
    for (let thing : things) {
      delete thing;
    }
    things.clear();
  }

  using ConstIterator = typename VectorRef<T>::const_iterator;
  fn begin() const -> ConstIterator { return things.cbegin(); };
  fn   end() const -> ConstIterator { return   things.cend(); };

  fn append(Ref<T> t) -> decltype(this) {
    things.push_back(t);
    return this;
  }

  fn size() const -> size_t { return things.size(); }

  fv virtual dump(unsigned indent = 0) const override {
    if (printAsList) {
      llvm::errs() << "\n";
      for (unsigned i = 0; i < indent; i++)
        llvm::errs() << indentStr;
      llvm::errs() << "(" << getKind() << " ";
      this->dumpNodeInfo();
    }

    unsigned incr = printAsList ? 1 : 0;

    for (let thing : things) {
      llvm::errs() << "\n";
      for (unsigned i = 0; i < indent + incr; i++)
        llvm::errs() << indentStr;
      thing->dump(indent + incr);
      for (unsigned i = 0; i < newlines; i++)
        llvm::errs() << "\n";
    }

    if (printAsList) {
      llvm::errs() << "\n";
      for (unsigned i = 0; i < indent; i++)
        llvm::errs() << indentStr;
      llvm::errs() << ") ";
    }
  }

  fn virtual getKind() const -> ASTNodeType override {
    return ASTNodeType::ASTNodeList;
  }

  static bool classof(const ASTNode *node) {
    return node->getKind() == ASTNodeType::ASTNodeList;
  }

private:
  VectorRef<T> things;
};

struct Expression : public ASTNode {
  // Expression(const Expression &) = default;
  // Expression(Expression &&) = delete;
  // Expression &operator=(const Expression &) = delete;
  // Expression &operator=(Expression &&) = delete;
  virtual ~Expression() {}
  fn virtual getExpressionKind() const -> ExpressionType = 0;
  fv virtual dumpInternal(unsigned indent = 0) const = 0;
  fv virtual dump(unsigned indent = 0) const override {
    llvm::errs() << '\n';
    for (unsigned i = 0; i < indent; i++)
      llvm::errs() << indentStr;
    llvm::errs() << "("
           << "\033[31m"
           << "expression type: "
           << "\033[0m"
           << getExpressionKind();
    llvm::errs() << " " << getKind() << " ";
    this->dumpNodeInfo();
    this->dumpInternal(indent + 1);
    llvm::errs() << ")";
  }
};

struct UnaryExpression : public Expression {
  UnaryExpression(const UnaryExpression &) = delete;
  UnaryExpression(UnaryExpression &&) = delete;
  UnaryExpression &operator=(const UnaryExpression &) = delete;
  UnaryExpression &operator=(UnaryExpression &&) = delete;
  UnaryExpression(UnaryOp op, std::unique_ptr<Expression> innerExpr)
      : op(op), innerExpr(std::move(innerExpr)) {}
  virtual ~UnaryExpression() {}
  fn virtual getExpressionKind() const -> ExpressionType override {
    return ExpressionType::Unary;
  }

  fn getOp() const -> UnaryOp { return op; }
  fn getInternalExpression() const -> CxxRef<Expression> {
    return *innerExpr.get();
  }

  fv virtual dumpInternal(unsigned indent = 0) const override {
    llvm::errs() << "(" << getKind() << " " << op;
    innerExpr->dump(indent + 1);
    llvm::errs() << " )";
  }

  fn virtual getKind() const -> ASTNodeType override {
    return ASTNodeType::UnaryExpr;
  }

  static bool classof(const ASTNode *node) {
    return node->getKind() == ASTNodeType::UnaryExpr;
  }

private:
  UnaryOp op;
  std::unique_ptr<Expression> innerExpr;
};

struct BinaryExpression : public Expression {
  BinaryExpression(const BinaryExpression &) = default;
  BinaryExpression(BinaryExpression &&) = delete;
  BinaryExpression &operator=(const BinaryExpression &) = delete;
  BinaryExpression &operator=(BinaryExpression &&) = delete;
  BinaryExpression(BinaryOp op,
                   Ref<Expression> leftExpr, Ref<Expression> rightExpr):
    op(op), leftExpr(leftExpr), rightExpr(rightExpr) {}
  virtual ~BinaryExpression() {
    delete leftExpr;
    delete rightExpr;
  }
  fn virtual getExpressionKind() const -> ExpressionType override {
    return ExpressionType::Binary;
  }

  fv virtual dumpInternal(unsigned indent = 0) const override {
    llvm::errs() << "(" << getKind() << "op: " << op << " ";
    leftExpr->dump(indent + 1);
    rightExpr->dump(indent + 1);
    llvm::errs() << ")";
  }

  fn getBinaryOp() const -> mu::ast::enums::BinaryOp { return op; }
  fn getLHS() const -> CxxRef<Expression> { return *leftExpr; }
  fn getRHS() const -> CxxRef<Expression> { return *rightExpr; }

  fn virtual getKind() const -> ASTNodeType override {
    return ASTNodeType::BinaryExpr;
  }

  static bool classof(const ASTNode *node) {
    return node->getKind() == ASTNodeType::BinaryExpr;
  }

private:
  BinaryOp op;
  Ref<Expression> leftExpr;
  Ref<Expression> rightExpr;
};

struct IdentifierExpression : public Expression {
  IdentifierExpression() = delete;
  IdentifierExpression(const IdentifierExpression &) = default;
  IdentifierExpression(IdentifierExpression &&) = delete;
  IdentifierExpression &operator=(const IdentifierExpression &) = delete;
  IdentifierExpression &operator=(IdentifierExpression &&) = delete;
  IdentifierExpression(std::string name) : name(name) {}
  virtual ~IdentifierExpression() {}
  fn virtual getExpressionKind() const -> ExpressionType override {
    return ExpressionType::Identifier;
  }
  fv virtual dumpInternal(unsigned indent = 0) const override {
    llvm::errs() << "(" << getKind() << "name: " << name << " )";
  }

  fn virtual getKind() const -> ASTNodeType override {
    return ASTNodeType::IdentifierExpr;
  }

  static bool classof(const ASTNode *node) {
    return node->getKind() == ASTNodeType::IdentifierExpr;
  }

private:
  std::string name;
};

struct ConstantExpression : public Expression {
  ConstantExpression() = delete;
  ConstantExpression(const ConstantExpression &) = default;
  ConstantExpression(ConstantExpression &&) = delete;
  ConstantExpression &operator=(const ConstantExpression &) = delete;
  ConstantExpression &operator=(ConstantExpression &&) = delete;
  ConstantExpression(std::string constant, mu::ast::enums::ConstantType type):
    constant(constant), type(type) {}
  virtual ~ConstantExpression() {}
  fn virtual getExpressionKind() const -> ExpressionType override {
    return ExpressionType::Constant;
  }
  fv virtual dumpInternal(unsigned indent = 0) const override {
    llvm::errs() << "(" << getKind() << constant << " )";
  }
  fn virtual getKind() const -> ASTNodeType override {
    return ASTNodeType::ConstantExpr;
  }

  static bool classof(const ASTNode *node) {
    return node->getKind() == ASTNodeType::ConstantExpr;
  }

  fn getValueAsInt() const -> int32_t {
    assert((type == mu::ast::enums::ConstantType::Char ||
            type == mu::ast::enums::ConstantType::IntKind1 ||
            type == mu::ast::enums::ConstantType::IntKind2 ||
            type == mu::ast::enums::ConstantType::IntKindHex) &&
           "can not get value as int if it is not integral");
    if (type == mu::ast::enums::ConstantType::Char) {
      return constant.c_str()[0];
    }
    return std::stoi(constant);
  }

private:
  std::string constant;
  mu::ast::enums::ConstantType type;
};

struct StringLiteralExpression : public Expression {
  StringLiteralExpression(const StringLiteralExpression &) = default;
  StringLiteralExpression(StringLiteralExpression &&) = delete;
  StringLiteralExpression &operator=(const StringLiteralExpression &) = delete;
  StringLiteralExpression &operator=(StringLiteralExpression &&) = delete;
  virtual ~StringLiteralExpression() {}
  fn virtual getExpressionKind() const -> ExpressionType override {
    return ExpressionType::StringLiteral;
  }
  fn virtual getKind() const -> ASTNodeType override {
    return ASTNodeType::StringLiteralExpr;
  }

  static bool classof(const ASTNode *node) {
    return node->getKind() == ASTNodeType::StringLiteralExpr;
  }
};

typealias ExpressionList = ASTList<Expression>;
struct CallExpression : public Expression {
  CallExpression() = delete;
  CallExpression(const CallExpression &) = delete;
  CallExpression(CallExpression &&) = delete;
  CallExpression &operator=(const CallExpression &) = delete;
  CallExpression &operator=(CallExpression &&) = delete;
  CallExpression(std::string name) : name(name) {}
  CallExpression(std::string name,
                 OptionalOwnedRef<ExpressionList> exprList)
      : name(name), exprList(std::move(exprList)) {}
  virtual ~CallExpression() {}

  fn virtual getExpressionKind() const->ExpressionType override {
    return ExpressionType::Call;
  }
  fv virtual dumpInternal(unsigned indent = 0) const override {
    llvm::errs() << "(" << getKind() << " " << name << " )";
    if (exprList.has_value()) {
      exprList.value()->dump(indent);
    }
    llvm::errs() << ")";
  }
  fn virtual getKind() const->ASTNodeType override {
    return ASTNodeType::CallExpr;
  }

  static bool classof(const ASTNode *node) {
    return node->getKind() == ASTNodeType::CallExpr;
  }

private:
  std::string name = "";
  OptionalOwnedRef<ExpressionList> exprList;
};

struct ParenthesisExpression : public Expression {
  ParenthesisExpression(const ParenthesisExpression &) = delete;
  ParenthesisExpression(ParenthesisExpression &&) = delete;
  ParenthesisExpression &operator=(const ParenthesisExpression &) = delete;
  ParenthesisExpression &operator=(ParenthesisExpression &&) = delete;
  ParenthesisExpression(std::unique_ptr<Expression> innerExpr)
      : innerExpr(std::move(innerExpr)) {}
  virtual ~ParenthesisExpression() {}

  fn virtual getExpressionKind() const -> ExpressionType override {
    return ExpressionType::Parenthesis;
  }
  fv virtual dumpInternal(unsigned indent = 0) const override {
    llvm::errs() << "(" << getKind() << " ";
    innerExpr->dump(indent + 1);
    llvm::errs() << " )";
  }
  fn virtual getKind() const -> ASTNodeType override {
    return ASTNodeType::ParenthesisExpr;
  }

  static bool classof(const ASTNode *node) {
    return node->getKind() == ASTNodeType::ParenthesisExpr;
  }

private:
  std::unique_ptr<Expression> innerExpr;
};

struct Statement : public ASTNode {
  // Statement(const Statement &) = default;
  // Statement(Statement &&) = delete;
  // Statement &operator=(const Statement &) = delete;
  // Statement &operator=(Statement &&) = delete;
  virtual ~Statement() {}
  fn virtual hasExpression() const -> bool = 0;
  fn virtual getExpression() const -> Ref<Expression> = 0;
  fn virtual getStatementKind() const -> StatementType = 0;
  fv virtual dumpInternal(unsigned indent = 0) const = 0;
  fv virtual dump(unsigned indent = 0) const override {
    llvm::errs() << '\n';
    for (unsigned i = 0; i < indent; i++)
      llvm::errs() << indentStr;
    llvm::errs() << "(statement type: " << getStatementKind();
    llvm::errs() << " " << getKind() << " ";
    this->dumpNodeInfo();
    this->dumpInternal(indent + 1);
    fflush(stdout);
    if (hasExpression()) {
      getExpression()->dump(indent + 1);
    }
    llvm::errs() << ")";
  }
};

typealias StatementList = ASTList<Statement>;
struct CompoundStatement : public Statement {
  CompoundStatement() = default;
  CompoundStatement(const CompoundStatement &) = delete;
  CompoundStatement(CompoundStatement &&) = delete;
  CompoundStatement &operator=(const CompoundStatement &) = delete;
  CompoundStatement &operator=(CompoundStatement &&) = delete;
  CompoundStatement(std::unique_ptr<StatementList> statements)
      : statements(std::move(statements)) {}
  virtual ~CompoundStatement() {}

  fn virtual hasExpression() const -> bool override { return false; }
  fn virtual getExpression() const -> Ref<Expression> override {
    assert(false && "Statement can not have an expresson.");
    exit(EXIT_FAILURE);
  }
  fn virtual getStatementKind() const -> StatementType override {
    return StatementType::Compound;
  }
  fv virtual dumpInternal(unsigned indent = 0) const override {
    if (!statements.has_value())
      return;
    statements.value()->dump(indent + 1);
  }
  fn virtual getKind() const -> ASTNodeType override {
    return ASTNodeType::CompoundStat;
  }

  auto begin() const { return statements.value()->begin(); }
  auto end() const { return statements.value()->end(); }

  static bool classof(const ASTNode *node) {
    return node->getKind() == ASTNodeType::CompoundStat;
  }

private:
  OptionalOwnedRef<StatementList> statements;
};

struct SelectionIfStatement : public Statement {
  SelectionIfStatement() = delete;
  SelectionIfStatement(const SelectionIfStatement &) = default;
  SelectionIfStatement(SelectionIfStatement &&) = delete;
  SelectionIfStatement &operator=(const SelectionIfStatement &) = delete;
  SelectionIfStatement &operator=(SelectionIfStatement &&) = delete;
  SelectionIfStatement(Ref<Expression> expr, Ref<CompoundStatement> ifBranch,
                       Ref<CompoundStatement> elseBranch)
      : expr(expr), ifBranch(ifBranch), elseBranch(elseBranch) {}
  SelectionIfStatement(Ref<Expression> expr, Ref<CompoundStatement> ifBranch)
      : expr(expr), ifBranch(ifBranch) {}

  virtual ~SelectionIfStatement() {
    delete expr;
    delete ifBranch;
    if (elseBranch.has_value()) {
      delete elseBranch.value();
    }
  }

  fn virtual hasExpression() const -> bool override { return true; }
  fn virtual getExpression() const -> Ref<Expression> override { return expr; }
  fn virtual getStatementKind() const -> StatementType override {
    return StatementType::SelectionIf;
  }
  fv virtual dumpInternal(unsigned indent = 0) const override {
    if (ifBranch) {
      ifBranch->dump(indent + 1);
    }

    if (elseBranch.has_value()) {
      elseBranch.value()->dump(indent + 1);
    }
  }
  fn virtual getKind() const -> ASTNodeType override {
    return ASTNodeType::SelectIfStat;
  }

  static bool classof(const ASTNode *node) {
    return node->getKind() == ASTNodeType::SelectIfStat;
  }

private:
  Ref<Expression> expr;
  Ref<CompoundStatement> ifBranch;
  OptionalRef<CompoundStatement> elseBranch;
};

struct IterationWhileStatement : public Statement {
  IterationWhileStatement(const IterationWhileStatement &) = default;
  IterationWhileStatement(IterationWhileStatement &&) = delete;
  IterationWhileStatement &operator=(const IterationWhileStatement &) = delete;
  IterationWhileStatement &operator=(IterationWhileStatement &&) = delete;
  IterationWhileStatement(Ref<Expression> expr, Ref<CompoundStatement> body)
      : expr(expr), body(body) {}
  virtual ~IterationWhileStatement() {
    delete expr;
    delete body;
  }

  fn virtual hasExpression() const -> bool override { return true; }
  fn virtual getExpression() const -> Ref<Expression> override { return expr; }
  fn virtual getStatementKind() const -> StatementType override {
    return StatementType::IterationWhile;
  }
  fv virtual dumpInternal(unsigned indent = 0) const override {
    if (body) {
      body->dump(indent + 1);
    }
  }
  fn virtual getKind() const -> ASTNodeType override {
    return ASTNodeType::IterationWhileStat;
  }

  static bool classof(const ASTNode *node) {
    return node->getKind() == ASTNodeType::IterationWhileStat;
  }

private:
  Ref<Expression> expr;
  Ref<CompoundStatement> body;
};

struct JumpReturnStatement : public Statement {
  JumpReturnStatement() = delete;
  JumpReturnStatement(const JumpReturnStatement &) = default;
  JumpReturnStatement(JumpReturnStatement &&) = delete;
  JumpReturnStatement &operator=(const JumpReturnStatement &) = delete;
  JumpReturnStatement &operator=(JumpReturnStatement &&) = delete;
  JumpReturnStatement(Ref<Expression> expr) : expr(expr) {}
  virtual ~JumpReturnStatement() {
    delete expr;
  }

  fn virtual hasExpression() const -> bool override { return true; }
  fn virtual getExpression() const -> Ref<Expression> override { return expr; }
  fn virtual getStatementKind() const -> StatementType override {
    return StatementType::JumpReturn;
  }
  fv virtual dumpInternal(unsigned indent = 0) const override {}
  fn virtual getKind() const -> ASTNodeType override {
    return ASTNodeType::JumpReturnStat;
  }

  static bool classof(const ASTNode *node) {
    return node->getKind() == ASTNodeType::JumpReturnStat;
  }

private:
  Ref<Expression> expr;
};

struct AssignmentStatement : public Statement {
  AssignmentStatement(const AssignmentStatement &) = default;
  AssignmentStatement(AssignmentStatement &&) = delete;
  AssignmentStatement &operator=(const AssignmentStatement &) = delete;
  AssignmentStatement &operator=(AssignmentStatement &&) = delete;
  AssignmentStatement(Ref<Expression> expr, std::string name)
      : expr(expr), name(name) {}

  virtual ~AssignmentStatement() {
    delete expr;
  }

  fn virtual hasExpression() const -> bool override { return true; }
  fn virtual getExpression() const -> Ref<Expression> override { return expr; }
  fn virtual getStatementKind() const -> StatementType override {
    return StatementType::Assignment;
  }
  fv virtual dumpInternal(unsigned indent = 0) const override {
    llvm::errs() << " name: " << name;
  }
  fn virtual getKind() const -> ASTNodeType override {
    return ASTNodeType::AssignmentStat;
  }

  static bool classof(const ASTNode *node) {
    return node->getKind() == ASTNodeType::AssignmentStat;
  }

private:
  Ref<Expression> expr;
  std::string name;
};

struct InitializationStatement : public Statement {
  InitializationStatement(const InitializationStatement &) = default;
  InitializationStatement(InitializationStatement &&) = delete;
  InitializationStatement &operator=(const InitializationStatement &) = delete;
  InitializationStatement &operator=(InitializationStatement &&) = delete;
  InitializationStatement(Ref<Expression> expr, std::string name, Type varType)
      : expr(expr), name(name), varType(varType) {}
  virtual ~InitializationStatement() {
    delete expr;
  }

  fn virtual hasExpression() const -> bool override { return true; }
  fn virtual getExpression() const -> Ref<Expression> override { return expr; }
  fn virtual getStatementKind() const -> StatementType override {
    return StatementType::Initialization;
  }
  fv virtual dumpInternal(unsigned indent = 0) const override {
    llvm::errs() << " name: " << name << ", varType: " << varType;
  }
  fn virtual getKind() const -> ASTNodeType override {
    return ASTNodeType::InitializationStat;
  }

  static bool classof(const ASTNode *node) {
    return node->getKind() == ASTNodeType::InitializationStat;
  }

private:
  Ref<Expression> expr;
  std::string name;
  Type varType;
};

struct ParamDecl : public ASTNode {
  ParamDecl() = delete;
  ParamDecl(const ParamDecl &) = default;
  ParamDecl(ParamDecl &&) = delete;
  ParamDecl &operator=(const ParamDecl &) = delete;
  ParamDecl &operator=(ParamDecl &&) = delete;
  ParamDecl(std::string name, Type varType) : name(name), varType(varType) {}
  virtual ~ParamDecl(){};
  fv virtual dump(unsigned indent = 0) const override {
    llvm::errs() << "(parameter ";
    llvm::errs() << "" << getKind() << " ";
    llvm::errs() << "name: " << name << ", ";
    llvm::errs() << "varType: " << varType << " ),";
  }
  fn virtual getKind() const -> ASTNodeType override {
    return ASTNodeType::ParamDecl;
  }

  static bool classof(const ASTNode *node) {
    return node->getKind() == ASTNodeType::ParamDecl;
  }

  fn getName() const -> std::string { return name; }
  fn getType() const -> Type { return varType; }

private:
  std::string name;
  Type varType;
};

typealias ParamList = ASTList<ParamDecl>;

struct Defun : public ASTNode {
  Defun() = delete;
  Defun(const Defun &) = delete;
  Defun(Defun &&) = delete;
  Defun &operator=(const Defun &) = delete;
  Defun &operator=(Defun &&) = delete;

  Defun(std::string name, std::unique_ptr<ParamList> params, Type returnType,
        std::unique_ptr<CompoundStatement> body)
      : name(name), params(std::move(params)), returnType(returnType),
        body(std::move(body)) {}
  Defun(std::string name, Type returnType,
        std::unique_ptr<CompoundStatement> body)
      : name(name), returnType(returnType), body(std::move(body)) {}
  virtual ~Defun() {}

  fv virtual dump(unsigned indent = 0) const override {
    llvm::errs() << "(defun name: " << name << ", type: " << returnType;
    llvm::errs() << " " << getKind() << " ";
    this->dumpNodeInfo();
    if (params.has_value()) {
      params.value()->dump(indent + 1);
    }
    body->dump(indent + 1);
    llvm::errs() << ")";
  }
  fn virtual getKind() const -> ASTNodeType override {
    return ASTNodeType::DefunDecl;
  }

  static bool classof(const ASTNode *node) {
    return node->getKind() == ASTNodeType::DefunDecl;
  }

  fn getName() const -> std::string { return name; }
  fn getReturnType() const -> Type { return returnType; }
  fn getBody() const -> CxxRef<CompoundStatement> { return *body.get(); }
  fn hasParams() const -> bool { return params.has_value(); }
  fn getParams() const -> CxxRef<ParamList> {
    if (!params.has_value()) {
      assert(false && "expected has value");
      exit(EXIT_FAILURE);
    }
    return *params.value().get();
  }

private:
  std::string name = "";
  OptionalOwnedRef<ParamList> params;
  Type returnType = Type::sint32_mut;
  std::unique_ptr<CompoundStatement> body;
};

typealias DefunList = ASTList<Defun, 1, false>;

struct TranslationUnit : public ASTNode {
  TranslationUnit() = delete;
  TranslationUnit(std::unique_ptr<DefunList> functions) : functions(std::move(functions)) {}
  virtual ~TranslationUnit() { }

  fv virtual dump(unsigned indent = 0) const override {
    llvm::errs() << "TranslationUnit Node: ";
    this->dumpNodeInfo();
    llvm::errs() << "\n";

    functions->dump();
  };

  fn virtual getKind() const -> ASTNodeType override {
    return ASTNodeType::TranslationUnit;
  }

  auto begin() { return functions->begin(); }
  auto end() { return functions->end(); }

  static bool classof(const ASTNode *node) {
    return node->getKind() == ASTNodeType::TranslationUnit;
  }

private:
  std::string name = "main";
  std::unique_ptr<DefunList> functions;
};

} // namespace mu
} // namespace ast

#endif /* _AST_H_ */
