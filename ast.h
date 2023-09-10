#include <cstdlib>
#include <ios>
#include <iostream>
#include <memory>
#include <string>
#include <system_error>
#include <vector>

#include "µt8.h"

// These are kinda gross, move them to another Source of Header:
struct ASTNode;
extern fv dumpASTNode(const ASTNode *node);
extern unsigned getASTNodeID(const ASTNode *node);
extern unsigned astNodeCreateCount;
extern unsigned astNodeDestroyCount;


fv extern dumpASTNodeType(const ASTNode *node);

#define astout std::cout
#define indentStr "  "

struct ASTNodeTracker {
  typealias ASTNodeTrackerRef = ASTNodeTracker &;

  virtual ~ASTNodeTracker() {}

  fn static get() -> ASTNodeTrackerRef {
    instance = instance ? instance : new ASTNodeTracker();
    return *instance;
  }

  fv static destroy() {
    if (instance && instance->hasTrackedNodes()) {
      astout << "Dumping Tracked Nodes:\n";
      for (unsigned i = 0; i < instance->tracked.size(); i++) {
        if (instance->tracked[i]) {
          astout << "[" << i << "] = " << instance->tracked[i] << " ";
          dumpASTNodeType(instance->tracked[i]);
          astout << "\n";
          dumpASTNode(instance->tracked[i]);
          astout << "\n";
        }
      }
    }
    assert(instance && !instance->hasTrackedNodes() &&
           "Expected all nodes to be untracked by dtors");
    delete instance;
    instance = nullptr;
  }

  fn size() const -> size_t { return tracked.size(); }
  fv track(Ref<ASTNode> node) { tracked.push_back(node); }

  fv untrack(const ASTNode *node) {
    let trackedNode = tracked[getASTNodeID(node)];
    assert(trackedNode == node && "tracked node mismatch!");
    tracked[getASTNodeID(node)] = nullptr;
  }

  fn hasTrackedNodes() const -> bool {
    return !std::all_of(tracked.begin(), tracked.end(),
                        [](const ASTNode *node) { return nullptr == node; });
  }

  fv virtual dump(unsigned indent = 0) const {
    for (auto *node : tracked) {
      dumpASTNode(node);
    }
  }

  fv virtual dumpNodeByID(unsigned id) const { dumpASTNode(tracked[id]); }

private:
  std::vector<const ASTNode *> tracked;
  ASTNodeTracker() {}
  static ASTNodeTracker *instance;
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
  switch (v) {
  case ASTNodeType::ASTNodeList:          os << "ASTNodeList"; break;
  case ASTNodeType::UnaryExpr:            os << "UnaryExpr"; break;
  case ASTNodeType::BinaryExpr:           os << "BinaryExpr"; break;
  case ASTNodeType::IdentifierExpr:       os << "IdentifierExpr"; break;
  case ASTNodeType::ConstantExpr:         os << "ConstantExpr"; break;
  case ASTNodeType::StringLiteralExpr:    os << "StringLiteralExpr"; break;
  case ASTNodeType::CallExpr:             os << "CallExpr"; break;
  case ASTNodeType::ParenthesisExpr:      os << "ParenthesisExpr"; break;
  case ASTNodeType::CompoundStat:         os << "CompoundStat"; break;
  case ASTNodeType::SelectIfStat:         os << "SelectIfStat"; break;
  case ASTNodeType::IterationWhileStat:   os << "IterationWhileStat"; break;
  case ASTNodeType::JumpReturnStat:       os << "JumpReturnStat"; break;
  case ASTNodeType::AssignmentStat:       os << "AssignmentStat"; break;
  case ASTNodeType::InitializationStat:   os << "InitializationStat"; break;
  case ASTNodeType::ParamDecl:            os << "ParamDecl"; break;
  case ASTNodeType::DefunDecl:            os << "DefunDecl"; break;
  case ASTNodeType::TranslationUnit:      os << "TranslationUnit"; break;
  }
  return os;
}

template <typename T>
T *checked_ptr_cast(void *ptr) {
  auto casted = static_cast<T*>(ptr);
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
  ASTNodeTracker &tracker;
  uint id = 0;
  uint lineNumber = 0;
  const unsigned magic_number = ASTNode::static_magic_number;
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

inline std::ostream &operator<<(std::ostream &os, Type v) {
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

template <typename T, unsigned newlines = 0, bool printAsList = true>
class ASTList : public ASTNode {
  std::vector<Ref<T>> things;
  typealias ASTListPtr = ASTList *;

public:
  ASTList() = delete;
  ASTList(Ref<T> t) {
    things.push_back(t);
  }
  virtual ~ASTList() {
    for (let thing : things) {
      delete thing;
    }
  }

  typename std::vector<Ref<T>>::const_iterator begin() const {
    return things.cbegin();
  };
  typename std::vector<Ref<T>>::const_iterator end() const {
    return things.cend();
  };

  fn append(Ref<T> t) -> ASTListPtr {
    assert(t && "Expected non-null thing");
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

struct Expression : public ASTNode {
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
  UnaryOp op;
  Ref<Expression> innerExpr;
  UnaryExpression(UnaryOp op, Ref<Expression> innerExpr)
      : op(op), innerExpr(innerExpr) {
    assert(innerExpr &&
           "inner expression on unary expression must not be null");
  }
  virtual ~UnaryExpression() {
    if (!innerExpr)
      return;
    delete innerExpr;
  }
  fn virtual getExpressionType() const -> ExpressionType override {
    return ExpressionType::Unary;
  }

  fv virtual dumpInternal(unsigned indent = 0) const override {
    astout << "(" << type() << " )";
  }

  fn virtual type() const -> ASTNodeType override {
    return ASTNodeType::UnaryExpr;
  }
};

struct BinaryExpression : public Expression {
  BinaryExpression(BinaryOp op, Ref<Expression> leftExpr,
                                Ref<Expression> rightExpr)
      : op(op), leftExpr(leftExpr), rightExpr(rightExpr) {

    astout << "Bianry Expr:" << op << "\n";
    astout << "root ID: " << this->getID() << "\n";
    astout << "left ID: " << leftExpr->getID() << "\n";
    astout << "right ID: " << rightExpr->getID() << "\n";
    leftExpr->dump();
    rightExpr->dump();
    astout << "\n";
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
  std::string name;
  IdentifierExpression() = delete;
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
};

struct ConstantExpression : public Expression {
  ConstantExpression() = delete;
  ConstantExpression(std::string constant): constant(constant) {}
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
  CallExpression(std::string name, ExpressionList *exprList = nullptr):
    name(name), exprList(exprList) {}
  virtual ~CallExpression() {
    if (exprList) {
      delete exprList;
    }
  }
  fn virtual getExpressionType() const -> ExpressionType override {
    return ExpressionType::Call;
  }
  fv virtual dumpInternal(unsigned indent = 0) const override {
    astout << "(" << type() << " " << name << " )";
    exprList->dump(indent);
    astout << ")";
  }
  fn virtual type() const -> ASTNodeType override {
    return ASTNodeType::CallExpr;
  }

private:
  std::string name = "";
  ExpressionList *exprList = nullptr;
};

struct ParenthesisExpression : public Expression {
  Expression *innerExpr = nullptr;
  ParenthesisExpression(Expression *innerExpr = nullptr)
      : innerExpr(innerExpr) {}
  virtual ~ParenthesisExpression() {
    if (!innerExpr)
      return;
    delete innerExpr;
  }
  fn virtual getExpressionType() const -> ExpressionType override {
    return ExpressionType::Parenthesis;
  }
  fv virtual dumpInternal(unsigned indent = 0) const override {
    astout << "(" << type() << " )";
  }
  fn virtual type() const -> ASTNodeType override {
    return ASTNodeType::ParenthesisExpr;
  }
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
  virtual ~Statement() { }
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
  StatementList *statements = nullptr;
  CompoundStatement(StatementList *statements = nullptr)
      : statements(statements) {}
  virtual ~CompoundStatement() {
    if (!statements)
      return;
    delete statements;
  }
  fn virtual hasExpression() const -> bool override { return false; }
  fn virtual getExpression() const -> Ref<Expression> override {
    assert(false && "Statement can not have an expresson.");
    exit(EXIT_FAILURE);
  }
  fn virtual getStatementType() const -> StatementType override {
    return StatementType::Compound;
  }
  fv virtual dumpInternal(unsigned indent = 0) const override {
    if (statements) {
      for (auto statement : *statements) {
        for (unsigned i = 0; i < indent; i++)
          astout << "\t";
        statement->dump(indent + 1);
      }
    }
  }
  fn virtual type() const -> ASTNodeType override {
    return ASTNodeType::CompoundStat;
  }
};

struct SelectionIfStatement : public Statement {
  Ref<Expression> expr;
  Ref<CompoundStatement> ifBranch;
  CompoundStatement *elseBranch = nullptr;

  SelectionIfStatement() = delete;
  SelectionIfStatement(Ref<Expression> expr, Ref<CompoundStatement> ifBranch,
                       CompoundStatement *elseBranch = nullptr)
      : expr(expr), ifBranch(ifBranch), elseBranch(elseBranch) {}
  virtual ~SelectionIfStatement() {
    delete expr;
    delete ifBranch;
    if (elseBranch) {
      delete elseBranch;
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

    if (elseBranch) {
      elseBranch->dump(indent + 1);
    }
  }
  fn virtual type() const -> ASTNodeType override {
    return ASTNodeType::SelectIfStat;
  }
};

struct IterationWhileStatement : public Statement {
  Ref<Expression> expr;
  Ref<CompoundStatement> body;
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
  std::string name = "";
  ParamList *params = nullptr;
  Type returnType = Type::sint32_mut;
  CompoundStatement *body;

  Defun() = default;
  Defun(const Defun &) = default;
  Defun(Defun &&) = delete;
  Defun &operator=(const Defun &) = delete;
  Defun &operator=(Defun &&) = delete;
  Defun(std::string name, ParamList *params, Type returnType,
        CompoundStatement *body)
      : name(name), params(params), returnType(returnType), body(body) {}

  virtual ~Defun() {
    if (params) {
      delete params;
    }

    if (body) {
      delete body;
    }
  }

  fv virtual dump(unsigned indent = 0) const override {
    astout << "(defun name: " << name << ", type: " << returnType;
    astout << " " << type() << " ";
    this->dumpNodeInfo();
    if (params)
      params->dump(indent + 1);
    if (body) {
      body->dump(indent + 1);
    }
    astout << ")";
  }
  fn virtual type() const -> ASTNodeType override {
    return ASTNodeType::DefunDecl;
  }
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
