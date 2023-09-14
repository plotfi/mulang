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

#define astout llvm::errs()
#define indentStr "  "

namespace muast {
// These are kinda gross, move them to another Source of Header:
struct ASTNode;
fn inline getASTNodeID(Ref<ASTNode> node) -> unsigned;
fv inline dumpASTNode(Ref<ASTNode> node);
fv inline dumpASTNodeType(Ref<ASTNode> node);
fv clearYYValStorage();

using namespace muast::enums;

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
    if (instance.has_value() && instance.value()->hasTrackedNodes()) {
      astout << "Dumping Tracked Nodes:\n";
      for (unsigned i = 0; i < instance.value()->tracked.size(); i++) {
        if (instance.value()->tracked[i]) {
          astout << "[" << i << "] = " << instance.value()->tracked[i] << " ";
          dumpASTNodeType(instance.value()->tracked[i]);
          astout << "\n";
          dumpASTNode(instance.value()->tracked[i]);
          astout << "\n";
        }
      }
    }
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

struct ASTNode {
  static const unsigned static_magic_number;
  ASTNode(): tracker(ASTNodeTracker::get()) {
    id = tracker.size();
    astout << "Previous Node: ";
    if (id) {
      tracker.dumpNodeByID(id - 1);
    }
    astout << "\n";
    // assert(id != 26 && "blow up ASTNode for debug");
    // assert(id != 102 && "blow up ASTNode for debug");
    astout << "create base ASTNode with id: " << id << " !!!\n";
    tracker.track(this);
  }
  // explicit ASTNode(const ASTNodeTracker &tracker) : tracker(tracker) {}
  ASTNode(const ASTNode &) = default;
  ASTNode(ASTNode &&) = delete;
  ASTNode &operator=(const ASTNode &) = delete;
  ASTNode &operator=(ASTNode &&) = delete;
  virtual ~ASTNode() {
    astout << "Untracking ID: " << getID() << "\n";
    tracker.untrack(this);
  }

  fv dumpNodeInfo() const {
    astout << " (ASTNode id: " << id;
    if (lineNumber > 0) {
      astout << ", lineNumber: " << lineNumber;
    }
    astout << ") ";
  }

  fv virtual dump(unsigned indent = 0) const = 0;
  fn getID() const -> unsigned { return id; }
  fn getLineNumber() const -> unsigned { return lineNumber; }
  fv setLineNumber(uint lineNumber) { this->lineNumber = lineNumber; }
  fn virtual type() const -> ASTNodeType = 0;
  fn check() const -> bool {
    return magic_number == ASTNode::static_magic_number;
  }

private:
  const ASTNodeTracker &tracker;
  uint id = 0;
  uint lineNumber = 0;
  const unsigned magic_number = ASTNode::static_magic_number;
};

fv inline dumpASTNode(Ref<ASTNode> node) { node->dump(); }
fn inline getASTNodeID(Ref<ASTNode> node) -> unsigned { return node->getID(); }
fv inline dumpASTNodeType(Ref<ASTNode> node) {
  llvm::errs() << "ASTNode dump with ID "
               << node->getID()
               << ": "
               << node->type();
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

  fv virtual dump(unsigned indent = 0) const override {
    if (printAsList) {
      astout << "\n";
      for (unsigned i = 0; i < indent; i++)
        astout << indentStr;
      astout << "(" << type() << " ";
      this->dumpNodeInfo();
    }

    unsigned incr = printAsList ? 1 : 0;

    for (let thing : things) {
      astout << "\n";
      for (unsigned i = 0; i < indent + incr; i++)
        astout << indentStr;
      thing->dump(indent + incr);
      for (unsigned i = 0; i < newlines; i++)
        astout << "\n";
    }

    if (printAsList) {
      astout << "\n";
      for (unsigned i = 0; i < indent; i++)
        astout << indentStr;
      astout << ") ";
    }
  }

  fn virtual type() const -> ASTNodeType override {
    return ASTNodeType::ASTNodeList;
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
  fn virtual getExpressionType() const -> ExpressionType = 0;
  fv virtual dumpInternal(unsigned indent = 0) const = 0;
  fv virtual dump(unsigned indent = 0) const override {
    astout << '\n';
    for (unsigned i = 0; i < indent; i++)
      astout << indentStr;
    astout << "("
           << "\033[31m"
           << "expression type: "
           << "\033[0m"
           << getExpressionType();
    astout << " " << type() << " ";
    this->dumpNodeInfo();
    this->dumpInternal(indent + 1);
    astout << ")";
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
  fn virtual getExpressionType() const -> ExpressionType override {
    return ExpressionType::Unary;
  }

  fv virtual dumpInternal(unsigned indent = 0) const override {
    astout << "(" << type() << " " << op;
    innerExpr->dump(indent + 1);
    astout << " )";
  }

  fn virtual type() const -> ASTNodeType override {
    return ASTNodeType::UnaryExpr;
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
  fn virtual getExpressionType() const -> ExpressionType override {
    return ExpressionType::Binary;
  }

  fv virtual dumpInternal(unsigned indent = 0) const override {
    astout << "(" << type() << "op: " << op << " ";
    leftExpr->dump(indent + 1);
    rightExpr->dump(indent + 1);
    astout << ")";
  }

  fn virtual type() const -> ASTNodeType override {
    return ASTNodeType::BinaryExpr;
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
  fn virtual getExpressionType() const -> ExpressionType override {
    return ExpressionType::Identifier;
  }
  fv virtual dumpInternal(unsigned indent = 0) const override {
    astout << "(" << type() << "name: " << name << " )";
  }

  fn virtual type() const -> ASTNodeType override {
    return ASTNodeType::IdentifierExpr;
  }

private:
  std::string name;
};

struct ConstantExpression : public Expression {

  [[gnu::pure]] int wtf() const { return 42; }
  ConstantExpression() = delete;
  ConstantExpression(const ConstantExpression &) = default;
  ConstantExpression(ConstantExpression &&) = delete;
  ConstantExpression &operator=(const ConstantExpression &) = delete;
  ConstantExpression &operator=(ConstantExpression &&) = delete;
  ConstantExpression(std::string constant) : constant(constant) {}
  virtual ~ConstantExpression() {}
  fn virtual getExpressionType() const -> ExpressionType override {
    return ExpressionType::Constant;
  }
  fv virtual dumpInternal(unsigned indent = 0) const override {
    astout << "(" << type() << constant << " )";
  }
  fn virtual type() const -> ASTNodeType override {
    return ASTNodeType::IdentifierExpr;
  }

private:
  std::string constant;
};

struct StringLiteralExpression : public Expression {
  StringLiteralExpression(const StringLiteralExpression &) = default;
  StringLiteralExpression(StringLiteralExpression &&) = delete;
  StringLiteralExpression &operator=(const StringLiteralExpression &) = delete;
  StringLiteralExpression &operator=(StringLiteralExpression &&) = delete;
  virtual ~StringLiteralExpression() {}
  fn virtual getExpressionType() const -> ExpressionType override {
    return ExpressionType::StringLiteral;
  }
  fn virtual type() const -> ASTNodeType override {
    return ASTNodeType::StringLiteralExpr;
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

  fn virtual getExpressionType() const->ExpressionType override {
    return ExpressionType::Call;
  }
  fv virtual dumpInternal(unsigned indent = 0) const override {
    astout << "(" << type() << " " << name << " )";
    if (exprList.has_value()) {
      exprList.value()->dump(indent);
    }
    astout << ")";
  }
  fn virtual type() const->ASTNodeType override {
    return ASTNodeType::CallExpr;
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

  fn virtual getExpressionType() const -> ExpressionType override {
    return ExpressionType::Parenthesis;
  }
  fv virtual dumpInternal(unsigned indent = 0) const override {
    astout << "(" << type() << " ";
    innerExpr->dump(indent + 1);
    astout << " )";
  }
  fn virtual type() const -> ASTNodeType override {
    return ASTNodeType::ParenthesisExpr;
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
  fn virtual getStatementType() const -> StatementType = 0;
  fv virtual dumpInternal(unsigned indent = 0) const = 0;
  fv virtual dump(unsigned indent = 0) const override {
    astout << '\n';
    for (unsigned i = 0; i < indent; i++)
      astout << indentStr;
    astout << "(statement type: " << getStatementType();
    astout << " " << type() << " ";
    this->dumpNodeInfo();
    this->dumpInternal(indent + 1);
    fflush(stdout);
    if (hasExpression()) {
      getExpression()->dump(indent + 1);
    }
    astout << ")";
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
  fn virtual getStatementType() const -> StatementType override {
    return StatementType::Compound;
  }
  fv virtual dumpInternal(unsigned indent = 0) const override {
    if (!statements.has_value())
      return;
    statements.value()->dump(indent + 1);
  }
  fn virtual type() const -> ASTNodeType override {
    return ASTNodeType::CompoundStat;
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
  fn virtual getStatementType() const -> StatementType override {
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
  fn virtual type() const -> ASTNodeType override {
    return ASTNodeType::SelectIfStat;
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
  fn virtual getStatementType() const -> StatementType override {
    return StatementType::IterationWhile;
  }
  fv virtual dumpInternal(unsigned indent = 0) const override {
    if (body) {
      body->dump(indent + 1);
    }
  }
  fn virtual type() const -> ASTNodeType override {
    return ASTNodeType::IterationWhileStat;
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
  fn virtual getStatementType() const -> StatementType override {
    return StatementType::JumpReturn;
  }
  fv virtual dumpInternal(unsigned indent = 0) const override {}
  fn virtual type() const -> ASTNodeType override {
    return ASTNodeType::JumpReturnStat;
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
  fn virtual getStatementType() const -> StatementType override {
    return StatementType::Assignment;
  }
  fv virtual dumpInternal(unsigned indent = 0) const override {
    astout << " name: " << name;
  }
  fn virtual type() const -> ASTNodeType override {
    return ASTNodeType::AssignmentStat;
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
  fn virtual getStatementType() const -> StatementType override {
    return StatementType::Initialization;
  }
  fv virtual dumpInternal(unsigned indent = 0) const override {
    astout << " name: " << name << ", varType: " << varType;
  }
  fn virtual type() const -> ASTNodeType override {
    return ASTNodeType::InitializationStat;
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
    astout << "(parameter ";
    astout << "" << type() << " ";
    astout << "name: " << name << ", ";
    astout << "varType: " << varType << " ),";
  }
  fn virtual type() const -> ASTNodeType override {
    return ASTNodeType::ParamDecl;
  }

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
    astout << "(defun name: " << name << ", type: " << returnType;
    astout << " " << type() << " ";
    this->dumpNodeInfo();
    if (params.has_value()) {
      params.value()->dump(indent + 1);
    }
    body->dump(indent + 1);
    astout << ")";
  }
  fn virtual type() const -> ASTNodeType override {
    return ASTNodeType::DefunDecl;
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
    astout << "TranslationUnit Node: ";
    this->dumpNodeInfo();
    astout << "\n";

    functions->dump();
  };

  fn virtual type() const -> ASTNodeType override {
    return ASTNodeType::TranslationUnit;
  }

  auto begin() { return functions->begin(); }
  auto end() { return functions->end(); }

private:
  std::string name = "main";
  std::unique_ptr<DefunList> functions;
};

} // namespace muast

#endif /* _AST_H_ */
