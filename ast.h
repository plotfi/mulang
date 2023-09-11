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

#include "µt8.h"

// These are kinda gross, move them to another Source of Header:
struct ASTNode;
fn inline getASTNodeID(Ref<ASTNode> node) -> unsigned;
fv inline dumpASTNode(Ref<ASTNode> node);
fv inline dumpASTNodeType(Ref<ASTNode> node);
fv clearYYValStorage();

#define astout std::cout
#define indentStr "  "

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

inline std::ostream &operator<<(std::ostream &os, ASTNodeType v) {
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
  std::cout << "ASTNode dump with ID "
            << node->getID()
            << ": "
            << node->type();
}

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

inline std::ostream &operator<<(std::ostream &os, Type v) {
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

template <typename T, unsigned newlines = 0, bool printAsList = true>
class ASTList : public ASTNode {
  VectorRef<T> things;

public:
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

inline std::ostream &operator<<(std::ostream &os, ExpressionType v) {
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

enum class UnaryOp { invertOp, notOp, negOp };

inline std::ostream &operator<<(std::ostream &os, UnaryOp v) {
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

inline std::ostream &operator<<(std::ostream &os, BinaryOp v) {
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
  BinaryExpression(BinaryOp op, Ref<Expression> leftExpr,
                   Ref<Expression> rightExpr)
      : op(op), leftExpr(leftExpr), rightExpr(rightExpr) {
    assert(leftExpr && rightExpr &&
           "inner expressions on binary expression must not be null");
  }
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

enum class StatementType {
  Compound,
  SelectionIf,
  IterationWhile,
  JumpReturn,
  Assignment,
  Initialization
};

inline std::ostream &operator<<(std::ostream &os, StatementType v) {
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
  Ref<Expression> expr;
  Ref<CompoundStatement> body;
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
};

struct JumpReturnStatement : public Statement {
  Ref<Expression> expr;
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
};

struct AssignmentStatement : public Statement {
  Ref<Expression> expr;
  std::string name;

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
};

struct InitializationStatement : public Statement {
  Ref<Expression> expr;
  std::string name;
  Type varType;

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
  std::string name = "main";
  std::unique_ptr<DefunList> funcs;

  TranslationUnit() = delete;
  TranslationUnit(std::unique_ptr<DefunList> funcs) : funcs(std::move(funcs)) {}
  virtual ~TranslationUnit() { }

  fv virtual dump(unsigned indent = 0) const override {
    astout << "TranslationUnit Node: ";
    this->dumpNodeInfo();
    astout << "\n";

    funcs->dump();
  };

  fn virtual type() const -> ASTNodeType override {
    return ASTNodeType::TranslationUnit;
  }
};

#endif /* _AST_H_ */