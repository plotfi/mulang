#include <ios>
#include <iostream>
#include <string>
#include <vector>

struct ASTNode;

extern void dumpASTNode(const ASTNode *node);

struct ASTNodeTracker {
private:
  std::vector<const ASTNode*> tracked;
  std::vector<const ASTNode*> untracked;
  ASTNodeTracker() { }

  static ASTNodeTracker *instance;

public:
  static ASTNodeTracker *get() {
    if (!instance) instance = new ASTNodeTracker();
    return instance;
  }

  void track(const ASTNode *node) {
    tracked.push_back(node);
  }

  void untrack(const ASTNode *node) {
    untracked.push_back(node);
  }

  size_t size() const {
    return tracked.size();
  }

  virtual void dump(unsigned indent = 0) const {
    for (auto *node : tracked) {
      dumpASTNode(node);
    }
  }

  virtual void dumpTracked() const {
    for (auto *node : tracked) {
      std::cout << "Node: " << node << "\n";
    }
  }

  virtual void dumpUntracked() const {
    for (auto *node : tracked) {
      if (std::find(untracked.begin(), untracked.end(), node) == untracked.end()) {
        std::cout << "Node: " << node << "\n";
      }
    }
  }



};

extern unsigned astNodeCreateCount;
extern unsigned astNodeDestroyCount;

struct ASTNode {
  ASTNodeTracker *tracker = nullptr;
  ASTNode() {
    std::cout << "create base ASTNode " << astNodeCreateCount++ << " !!!\n";
    tracker = ASTNodeTracker::get();
    tracker->track(this);
  }
  virtual ~ASTNode() {
    std::cout << "destroy base ASTNode " << astNodeDestroyCount++ << " !!!\n";
  }
  virtual void dump(unsigned indent = 0) const = 0;
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

template <typename T> struct ASTList : public ASTNode {
  std::vector<T *> things;

  ASTList() = delete;
  ASTList(T *t) { things.push_back(t); }
  virtual ~ASTList() {
    for (const auto *thing : things) {
      if (thing) {
        delete thing;
      }
    }
  }

  ASTList *append(T *t) {
    things.push_back(t);
    return this;
  }

  virtual void dump(unsigned indent = 0) const override {
    for (auto *thing : things) {
      std::cout << "\n";
      for (unsigned i = 0; i < indent; i++)
        std::cout << "\t";
      thing->dump();
    }
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
    os << "expression_type: unary ";
    break;
  case ExpressionType::Binary:
    os << "expression_type: binary ";
    break;
  case ExpressionType::Identifier:
    os << "expression_type: identifier ";
    break;
  case ExpressionType::Constant:
    os << "expression_type: constant ";
    break;
  case ExpressionType::StringLiteral:
    os << "expression_type: string_literal ";
    break;
  case ExpressionType::Call:
    os << "expression_type: call ";
    break;
  case ExpressionType::Parenthesis:
    os << "expression_type: paren ";
    break;
  }
  return os;
}

struct Expression : public ASTNode {
  virtual ~Expression() {}
  virtual ExpressionType getExpressionType() const = 0;
};

enum class UnaryOp {
  invertOp,
  notOp,
  negOp
};

inline std::ostream &operator<<(std::ostream &os, UnaryOp v) {
  switch (v) {
  case UnaryOp::invertOp:  os << "op: invert "; break;
  case UnaryOp::notOp:  os << "op: not "; break;
  case UnaryOp::negOp:  os << "op: neg "; break;
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
  case BinaryOp::mulOp:  os << "op: mul "; break;
  case BinaryOp::divOp:  os << "op: div "; break;
  case BinaryOp::modOp:  os << "op: mod "; break;
  case BinaryOp::addOp:  os << "op: add "; break;
  case BinaryOp::subOp:  os << "op: sub "; break;
  case BinaryOp::lshOp:  os << "op: lsh "; break;
  case BinaryOp::rshOp:  os << "op: rsh "; break;
  case BinaryOp::ltOp:   os << "op: lt "; break;
  case BinaryOp::gtOp:   os << "op: gt "; break;
  case BinaryOp::leOp:   os << "op: le "; break;
  case BinaryOp::geOp:   os << "op: ge "; break;
  case BinaryOp::eqOp:   os << "op: eq "; break;
  case BinaryOp::neOp:   os << "op: ne "; break;
  case BinaryOp::andOp:  os << "op: and "; break;
  case BinaryOp::xorOp:  os << "op: xor "; break;
  case BinaryOp::orOp:   os << "op: or "; break;
  case BinaryOp::andbOp: os << "op: andb "; break;
  case BinaryOp::orbOp:  os << "op: orb "; break;
  }
  return os;
}

struct UnaryExpression : public Expression {
  UnaryOp op;
  Expression *innerExpr = nullptr;
  UnaryExpression(UnaryOp op, Expression *innerExpr)
      : op(op), innerExpr(innerExpr) {
    assert(innerExpr &&
           "inner expression on unary expression must not be null");
  }
  virtual ~UnaryExpression() {
    if (innerExpr) {
      delete innerExpr;
    }
  }
  virtual ExpressionType getExpressionType() const override {
    return ExpressionType::Unary;
  }

  virtual void dump(unsigned indent = 0) const override {
    std::cout << "(unary )";
  }
};

struct BinaryExpression : public Expression {
  BinaryOp op;
  Expression *leftExpr = nullptr;
  Expression *rightExpr = nullptr;
  BinaryExpression(BinaryOp op, Expression *leftExpr, Expression *rightExpr)
      : op(op), leftExpr(leftExpr), rightExpr(rightExpr) {
    assert(leftExpr && rightExpr &&
           "inner expressions on binary expression must not be null");
  }

  virtual ~BinaryExpression() {
    if (leftExpr) {
      delete leftExpr;
    }
    if (rightExpr) {
      delete rightExpr;
    }
  }
  virtual ExpressionType getExpressionType() const override {
    return ExpressionType::Binary;
  }

  virtual void dump(unsigned indent = 0) const override {
    std::cout << "(binary )";
  }
};

struct IdentifierExpression : public Expression {
  virtual ~IdentifierExpression() {}
  virtual ExpressionType getExpressionType() const override {
    return ExpressionType::Identifier;
  }
};

struct ConstantExpression : public Expression {
  virtual ~ConstantExpression() {}
  virtual ExpressionType getExpressionType() const override {
    return ExpressionType::Constant;
  }
};

struct StringLiteralExpression : public Expression {
  virtual ~StringLiteralExpression() {}
  virtual ExpressionType getExpressionType() const override {
    return ExpressionType::StringLiteral;
  }
};

struct CallExpression : public Expression {
  virtual ~CallExpression() {}
  virtual ExpressionType getExpressionType() const override {
    return ExpressionType::Call;
  }
};

struct ParenthesisExpression : public Expression {
  Expression *innerExpr = nullptr;
  ParenthesisExpression(Expression *innerExpr = nullptr)
      : innerExpr(innerExpr) {}
  virtual ~ParenthesisExpression() {
    if (innerExpr) {
      delete innerExpr;
    }
  }
  virtual ExpressionType getExpressionType() const override {
    return ExpressionType::Parenthesis;
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
  Expression *expr = nullptr;
  Statement(Expression *expr = nullptr) : expr(expr) {}
  virtual ~Statement() {
    if (expr) {
      delete expr;
    }
  }
  virtual bool hasExpression() const { return false; }
  virtual StatementType getStatementType() const = 0;
  virtual void dumpInternal(unsigned indent = 0) const = 0;
  virtual void dump(unsigned indent = 0) const override {
    std::cout << '\n';
    for (unsigned i = 0; i < indent; i++)
      std::cout << "\t";
    std::cout << "(statement type: " << getStatementType();
    this->dumpInternal(indent + 1);
    std::cout << ")";
  }
};

using StatementList = ASTList<Statement>;
struct CompoundStatement : public Statement {
  StatementList *statements = nullptr;
  CompoundStatement(StatementList *statements = nullptr)
      : Statement(nullptr), statements(statements) {}
  virtual ~CompoundStatement() {
    if (statements) {
      delete statements;
    }
  }
  virtual StatementType getStatementType() const {
    return StatementType::Compound;
  }
  virtual void dumpInternal(unsigned indent = 0) const {
    if (statements) {
      for (auto *statement : statements->things) {
        for (unsigned i = 0; i < indent; i++)
          std::cout << "\t";
        statement->dump(indent + 1);
      }
    }
  }
};

struct SelectionIfStatement : public Statement {
  CompoundStatement *ifBranch = nullptr;
  CompoundStatement *elseBranch = nullptr;

  SelectionIfStatement() = delete;
  SelectionIfStatement(Expression *expr, CompoundStatement *ifBranch,
                       CompoundStatement *elseBranch = nullptr)
      : Statement(expr), ifBranch(ifBranch), elseBranch(elseBranch) {}
  virtual ~SelectionIfStatement() {
    if (ifBranch) {
      delete ifBranch;
    }
    if (elseBranch) {
      delete elseBranch;
    }
  }

  virtual bool hasExpression() const override { return true; }
  virtual StatementType getStatementType() const override {
    return StatementType::SelectionIf;
  }
  virtual void dumpInternal(unsigned indent = 0) const override {
    if (ifBranch) {
      ifBranch->dump(indent + 1);
    }

    if (elseBranch) {
      elseBranch->dump(indent + 1);
    }
  }
};

struct IterationWhileStatement : public Statement {
  CompoundStatement *body = nullptr;
  IterationWhileStatement(Expression *expr, CompoundStatement *body)
      : Statement(expr), body(body) {}
  virtual ~IterationWhileStatement() {
    if (body) {
      delete body;
    }
  }
  virtual StatementType getStatementType() const override {
    return StatementType::IterationWhile;
  }
  virtual void dumpInternal(unsigned indent = 0) const override {
    if (body) {
      body->dump(indent + 1);
    }
  }
};

struct JumpReturnStatement : public Statement {
  virtual StatementType getStatementType() const override {
    return StatementType::JumpReturn;
  }
  virtual void dumpInternal(unsigned indent = 0) const override {}
};

struct AssignmentStatement : public Statement {
  std::string name;

  AssignmentStatement(Expression *expr, std::string name)
      : Statement(expr), name(name) {}

  virtual StatementType getStatementType() const override {
    return StatementType::Assignment;
  }
  virtual void dumpInternal(unsigned indent = 0) const override {
    std::cout << " name: " << name;
  }
};

struct InitializationStatement : public Statement {
  std::string name;
  Type type;

  InitializationStatement(Expression *expr, std::string name, Type type)
      : Statement(expr), name(name), type(type) {}
  virtual ~InitializationStatement() {}

  virtual StatementType getStatementType() const override {
    return StatementType::Initialization;
  }
  virtual void dumpInternal(unsigned indent = 0) const override {
    std::cout << " name: " << name << ", type: " << type;
  }
};

struct ParamDecl : public ASTNode {
  std::string name;
  Type type;

  ParamDecl() = delete;
  ParamDecl(std::string name, Type type) : name(name), type(type) {}
  virtual ~ParamDecl() {};

  virtual void dump(unsigned indent = 0) const override {
    for (unsigned i = 0; i < indent; i++)
      std::cout << "\t";

    std::cout << "(parameter ";
    std::cout << "name: " << name << ", ";
    std::cout << "type: " << type << " ),";
  }
};

using ParamList = ASTList<ParamDecl>;

struct Defun : public ASTNode {
  std::string name = "";
  ParamList *params = nullptr;
  Type returnType = Type::sint32_mut;
  CompoundStatement *body;

  Defun() = default;
  Defun(const char *name, ParamList *params = nullptr,
        Type returnType = Type::sint32_mut, CompoundStatement *body = nullptr)
      : name(name), params(params), returnType(returnType), body(body) {}
  virtual ~Defun() {
    if (params) {
      delete params;
    }

    if (body) {
      delete body;
    }
  }

  virtual void dump(unsigned indent = 0) const override {
    for (unsigned i = 0; i < indent; i++)
      std::cout << "\t";

    std::cout << "(defun name: " << name << ", type: " << returnType;
    if (params)
      params->dump(indent + 1);
    if (body) {
      body->dump(indent + 1);
    }
    std::cout << ")";
  }
};

using DefunList = ASTList<Defun>;

struct TranslationUnit : public ASTNode {
  std::string name = "main";
  DefunList *funcs = nullptr;

  TranslationUnit() = delete;
  TranslationUnit(DefunList *funcs) : funcs(funcs) {}
  virtual ~TranslationUnit() {
    delete funcs;
  }

  virtual void dump(unsigned indent = 0) const override { funcs->dump(); };
};
