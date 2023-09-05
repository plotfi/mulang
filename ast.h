#include <ios>
#include <iostream>
#include <string>
#include <vector>

struct ASTNode;

extern void dumpASTNode(const ASTNode *node);
extern void dumpASTNodeType(const ASTNode *node);
extern unsigned getASTNodeID(const ASTNode *node);

struct ASTNodeTracker {
private:
  std::vector<const ASTNode*> tracked;
  ASTNodeTracker() { }

  static ASTNodeTracker *instance;

public:
  static ASTNodeTracker *get() {
    if (!instance) instance = new ASTNodeTracker();
    return instance;
  }

  static void destroy() {
    if (instance && instance->hasTrackedNodes()) {
      std::cout << "Dumping Tracked Nodes:\n";
      for (unsigned i = 0; i < instance->tracked.size(); i++) {
        if (instance->tracked[i]) {
          std::cout << "[" << i << "] = " << instance->tracked[i] << " ";
          dumpASTNodeType(instance->tracked[i]);
        }
      }
    }
    assert(instance && !instance->hasTrackedNodes() && "Expected all nodes to be untracked by dtors");
    delete instance;
    instance = nullptr;
  }

  void track(const ASTNode *node) {
    tracked.push_back(node);
  }

  void untrack(const ASTNode *node) {
    const ASTNode *trackedNode = tracked[getASTNodeID(node)];
    assert(trackedNode == node && "tracked node mismatch!");
    tracked[getASTNodeID(node)] = nullptr;
  }

  [[nodiscard]] auto hasTrackedNodes() const -> bool {
    return !std::all_of(tracked.begin(), tracked.end(), [](const ASTNode *node) { return nullptr == node; });
  }

  size_t size() const {
    return tracked.size();
  }

  virtual void dump(unsigned indent = 0) const {
    for (auto *node : tracked) {
      dumpASTNode(node);
    }
  }
};

extern unsigned astNodeCreateCount;
extern unsigned astNodeDestroyCount;

struct ASTNode {
  ASTNodeTracker *tracker = nullptr;
  unsigned id = 0;
  ASTNode() {
    tracker = ASTNodeTracker::get();
    id = tracker->size();
    std::cout << "create base ASTNode with id: " << id << " !!!\n";
    tracker->track(this);
  }
  virtual ~ASTNode() {
    tracker->untrack(this);
  }
  virtual void dump(unsigned indent = 0) const = 0;
  virtual void dumpASTNodeType() const = 0;
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

template <typename T, unsigned newlines = 0> struct ASTList : public ASTNode {
  std::vector<T *> things;

  ASTList() = delete;
  ASTList(T *t) { things.push_back(t); }
  virtual ~ASTList() {
    for (const auto *thing : things) {
      if (!thing)
        continue;
      delete thing;
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
      for (unsigned i = 0; i < newlines; i++)
        std::cout << "\n";
    }
  }


  virtual void dumpASTNodeType() const override { std::cout << "ASTList\n"; }
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
  virtual ~Expression() { }
  virtual ExpressionType getExpressionType() const = 0;
  virtual void dumpInternal(unsigned indent = 0) const = 0;
  virtual void dump(unsigned indent = 0) const override {
    std::cout << '\n';
    for (unsigned i = 0; i < indent; i++)
      std::cout << "\t";
    std::cout << "(expression type: " << getExpressionType();
    this->dumpInternal(indent + 1);
    std::cout << ")";
  }
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
    if (!innerExpr)
      return;
    delete innerExpr;
  }
  virtual ExpressionType getExpressionType() const override {
    return ExpressionType::Unary;
  }

  virtual void dumpInternal(unsigned indent = 0) const override {
    std::cout << "(unary )";
  }
  virtual void dumpASTNodeType() const override { std::cout << "UnaryExpr\n"; }
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

  virtual void dumpInternal(unsigned indent = 0) const override {
    std::cout << "(binary )";
  }
  virtual void dumpASTNodeType() const override { std::cout << "BinaryExpr\n"; }
};

struct IdentifierExpression : public Expression {
  std::string name;
  IdentifierExpression() = delete;
  IdentifierExpression(std::string name): name(name) {}
  virtual ~IdentifierExpression() {}
  virtual ExpressionType getExpressionType() const override {
    return ExpressionType::Identifier;
  }
  virtual void dumpASTNodeType() const override { std::cout << "IdExpr\n"; }
  virtual void dumpInternal(unsigned indent = 0) const override {
    std::cout << "(IDENTIFIER )";
  }
};

struct ConstantExpression : public Expression {
  virtual ~ConstantExpression() {}
  virtual ExpressionType getExpressionType() const override {
    return ExpressionType::Constant;
  }
  virtual void dumpASTNodeType() const override { std::cout << "ConstExpr\n"; }
  virtual void dumpInternal(unsigned indent = 0) const override {
    std::cout << "(CONSTANT )";
  }
};

struct StringLiteralExpression : public Expression {
  virtual ~StringLiteralExpression() {}
  virtual ExpressionType getExpressionType() const override {
    return ExpressionType::StringLiteral;
  }
  virtual void dumpASTNodeType() const override { std::cout << "StirngLitExpr\n"; }
};

struct CallExpression : public Expression {
  virtual ~CallExpression() {}
  virtual ExpressionType getExpressionType() const override {
    return ExpressionType::Call;
  }
  virtual void dumpASTNodeType() const override { std::cout << "CallExpr\n"; }
  virtual void dumpInternal(unsigned indent = 0) const override {
    std::cout << "(call)";
  }
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
  virtual ExpressionType getExpressionType() const override {
    return ExpressionType::Parenthesis;
  }
  virtual void dumpASTNodeType() const override { std::cout << "ParenExpr\n"; }
  virtual void dumpInternal(unsigned indent = 0) const override {
    std::cout << "(paren)";
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
    if (!expr)
      return;
    delete expr;
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
    fflush(stdout);
    if (expr) {
      expr->dump(indent + 1);
    }
    std::cout << ")";
  }
};

using StatementList = ASTList<Statement>;
struct CompoundStatement : public Statement {
  StatementList *statements = nullptr;
  CompoundStatement(StatementList *statements = nullptr)
      : Statement(nullptr), statements(statements) {}
  virtual ~CompoundStatement() {
    if (!statements)
      return;
    delete statements;
  }
  virtual StatementType getStatementType() const override {
    return StatementType::Compound;
  }
  virtual void dumpInternal(unsigned indent = 0) const override {
    if (statements) {
      for (auto *statement : statements->things) {
        for (unsigned i = 0; i < indent; i++)
          std::cout << "\t";
        statement->dump(indent + 1);
      }
    }
  }
  virtual void dumpASTNodeType() const override { std::cout << "CompoundStatement\n"; }
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
  virtual void dumpASTNodeType() const override { std::cout << "SelectIfStatement\n"; }
};

struct IterationWhileStatement : public Statement {
  CompoundStatement *body = nullptr;
  IterationWhileStatement(Expression *expr, CompoundStatement *body)
      : Statement(expr), body(body) {}
  virtual ~IterationWhileStatement() {
    if (!body)
      return;
    delete body;
  }
  virtual StatementType getStatementType() const override {
    return StatementType::IterationWhile;
  }
  virtual void dumpInternal(unsigned indent = 0) const override {
    if (body) {
      body->dump(indent + 1);
    }
  }
  virtual void dumpASTNodeType() const override { std::cout << "IterWhileStatement\n"; }
};

struct JumpReturnStatement : public Statement {
  JumpReturnStatement() = delete;
  JumpReturnStatement(Expression *expr): Statement(expr) { }

  virtual StatementType getStatementType() const override {
    return StatementType::JumpReturn;
  }
  virtual void dumpInternal(unsigned indent = 0) const override {}
  virtual void dumpASTNodeType() const override { std::cout << "JumpRetStatement\n"; }
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
  virtual void dumpASTNodeType() const override { std::cout << "AssignmentStatement\n"; }
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
  virtual void dumpASTNodeType() const override { std::cout << "InitStatement\n"; }
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
  virtual void dumpASTNodeType() const override { std::cout << "ParamDecl\n"; }
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
  virtual void dumpASTNodeType() const override { std::cout << "DefunDecl\n"; }
};

using DefunList = ASTList<Defun, 1>;

struct TranslationUnit : public ASTNode {
  std::string name = "main";
  DefunList *funcs = nullptr;

  TranslationUnit() = delete;
  TranslationUnit(DefunList *funcs) : funcs(funcs) {}
  virtual ~TranslationUnit() {
    delete funcs;
  }

  virtual void dump(unsigned indent = 0) const override { funcs->dump(); };
  virtual void dumpASTNodeType() const override { std::cout << "TranslationUnit\n"; }
};
