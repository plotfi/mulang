%{
#include <cstdio>
#include <cstdlib>
#include <unordered_map>

#include "Mu/Parser/ast.h"
#include "Mu/Parser/yyvaltype.h"

#include "llvm/Support/raw_ostream.h"

using namespace mu::ast;

#if 1
#define debug_print(...) printf(__VA_ARGS__)
#else
#define debug_print(...)
#endif

int yylex();
void yyerror(const char *s);

#define YYSTYPE void*

mu::ast::TranslationUnit *topnode = nullptr;

// TODO: This is a gross hack for now to satisfy Bison
static
std::unordered_map<mu::ast::enums::Type, mu::ast::ASTTypeContainer*> gASTTypeMap = {
  { mu::ast::enums::Type::sint8_mut  , new mu::ast::ASTTypeContainer( mu::ast::enums::Type::sint8_mut  ) },
  { mu::ast::enums::Type::uint8_mut  , new mu::ast::ASTTypeContainer( mu::ast::enums::Type::uint8_mut  ) },
  { mu::ast::enums::Type::sint16_mut , new mu::ast::ASTTypeContainer( mu::ast::enums::Type::sint16_mut ) },
  { mu::ast::enums::Type::uint16_mut , new mu::ast::ASTTypeContainer( mu::ast::enums::Type::uint16_mut ) },
  { mu::ast::enums::Type::sint32_mut , new mu::ast::ASTTypeContainer( mu::ast::enums::Type::sint32_mut ) },
  { mu::ast::enums::Type::uint32_mut , new mu::ast::ASTTypeContainer( mu::ast::enums::Type::uint32_mut ) },
  { mu::ast::enums::Type::sint64_mut , new mu::ast::ASTTypeContainer( mu::ast::enums::Type::sint64_mut ) },
  { mu::ast::enums::Type::uint64_mut , new mu::ast::ASTTypeContainer( mu::ast::enums::Type::uint64_mut ) },
  { mu::ast::enums::Type::float32_mut, new mu::ast::ASTTypeContainer( mu::ast::enums::Type::float32_mut) },
  { mu::ast::enums::Type::float64_mut, new mu::ast::ASTTypeContainer( mu::ast::enums::Type::float64_mut) },
  { mu::ast::enums::Type::bool_mut   , new mu::ast::ASTTypeContainer( mu::ast::enums::Type::bool_mut   ) },
};

%}

%token VAR FUNCTION
%token PTR_OP
%token LEFT_OP RIGHT_OP
%token AND_OP OR_OP
%token LE_OP GE_OP EQ_OP NE_OP
%token IDENTIFIER CONSTANT STRING_LITERAL
%token TYPE_NAME
%token CHAR SHORT INT LONG FLOAT DOUBLE USHORT UINT ULONG INT8 UINT8 BOOL
%token IF ELSE WHILE RETURN

%start translation_unit
%%

/*** mu top-level constructs ***/
translation_unit
  : toplevel_declaration_list {
    var FL = std::unique_ptr<DefunList>(checked_ptr_cast<DefunList>($1));
    topnode = new TranslationUnit(std::move(FL));
  }
  ;

toplevel_declaration_list
  : toplevel_declaration {
    $$ = new DefunList(checked_ptr_cast<Defun>($1));
  }
  | toplevel_declaration_list toplevel_declaration {
    $$ = checked_ptr_cast<DefunList>($1)->append(checked_ptr_cast<Defun>($2));
  }
  ;

toplevel_declaration
  : mu_function_definition {
    $$ = $1;
  }
  ;

mu_function_definition
  : FUNCTION IDENTIFIER '(' parameter_list ')' PTR_OP
      type_specifier compound_statement {
    var P = std::unique_ptr<ParamList>(checked_ptr_cast<ParamList>($4));
    var B = std::unique_ptr<CompoundStatement>(
        checked_ptr_cast<CompoundStatement>($8));
    $$ = new Defun(static_cast<YYValType *>($2)->value, std::move(P),
                   static_cast<mu::ast::ASTTypeContainer*>($7)->getType(), std::move(B));
    checked_ptr_cast<ASTNode>($$)->setLineNumber(
        static_cast<YYValType *>($2)->linenum);
  }
  | FUNCTION IDENTIFIER '(' ')' PTR_OP  type_specifier compound_statement {
    var B = std::unique_ptr<CompoundStatement>(
        checked_ptr_cast<CompoundStatement>($7));
    $$ = new Defun(static_cast<YYValType *>($2)->value,
                   static_cast<mu::ast::ASTTypeContainer*>($6)->getType(),
                   std::move(B));
    checked_ptr_cast<ASTNode>($$)->setLineNumber(
        static_cast<YYValType *>($2)->linenum);
  }
  ;

parameter_list
  : parameter_declaration {
    $$ = new ParamList(checked_ptr_cast<ParamDecl>($1));
    checked_ptr_cast<ASTNode>($$)->setLineNumber(
        checked_ptr_cast<ParamDecl>($1)->getLocation().line);
  }
  | parameter_list ',' parameter_declaration {
    $$ = checked_ptr_cast<ParamList>($1)->append(
        checked_ptr_cast<ParamDecl>($3));
    checked_ptr_cast<ASTNode>($$)->setLineNumber(
        static_cast<YYValType *>($2)->linenum);
  }
  ;

parameter_declaration
  : IDENTIFIER ':' type_specifier {
    $$ = new ParamDecl(static_cast<YYValType *>($1)->value,
                       static_cast<mu::ast::ASTTypeContainer*>($3)->getType());
    checked_ptr_cast<ASTNode>($$)->setLineNumber(
        static_cast<YYValType *>($2)->linenum);
  }
  ;

/*** mu statements ***/
statement_list
  : statement {
    $$ = new StatementList(checked_ptr_cast<Statement>($1));
  }
  | statement_list statement {
    $$ = checked_ptr_cast<StatementList>($1)->append(
        checked_ptr_cast<Statement>($2));
  };

statement
  : compound_statement {
    $$ = $1;
  }
  | selection_statement {
    $$ = $1;
  }
  | iteration_statement {
    $$ = $1;
  }
  | jump_statement {
    $$ = $1;
  }
  | assignment_statement {
    $$ = $1;
  }
  | init_statement {
    $$ = $1;
  }
  ;

compound_statement
  : '{' '}'  {
    $$ = new CompoundStatement();
    checked_ptr_cast<ASTNode>($$)->setLineNumber(
        static_cast<YYValType *>($1)->linenum);
  }
  | '{' statement_list '}' {
    var B = std::unique_ptr<StatementList>(checked_ptr_cast<StatementList>($2));
    $$ = new CompoundStatement(std::move(B));
    checked_ptr_cast<ASTNode>($$)->setLineNumber(
        static_cast<YYValType *>($1)->linenum);
  }
  ;

selection_statement
  : IF expression compound_statement {
    $$ = new SelectionIfStatement(checked_ptr_cast<Expression>($2),
                                  checked_ptr_cast<CompoundStatement>($3));
    checked_ptr_cast<ASTNode>($$)
      ->setLineNumber(static_cast<YYValType *>($1)->linenum);
  }
  | IF expression compound_statement ELSE compound_statement {
    $$ = new SelectionIfStatement(checked_ptr_cast<Expression>($2),
                                  checked_ptr_cast<CompoundStatement>($3),
                                  checked_ptr_cast<CompoundStatement>($5));
    checked_ptr_cast<ASTNode>($$)
      ->setLineNumber(static_cast<YYValType *>($1)->linenum);
  }
  ;

iteration_statement
  : WHILE expression compound_statement {
    $$ = new IterationWhileStatement(checked_ptr_cast<Expression>($2),
                                     checked_ptr_cast<CompoundStatement>($3));
    checked_ptr_cast<ASTNode>($$)
      ->setLineNumber(static_cast<YYValType *>($1)->linenum);
  }
  ;

jump_statement
  : RETURN expression ';' {
    $$ = new JumpReturnStatement(checked_ptr_cast<Expression>($2));
    checked_ptr_cast<ASTNode>($$)
      ->setLineNumber(static_cast<YYValType *>($1)->linenum);
  }
  ;

assignment_statement
  : IDENTIFIER '=' expression ';' {
    $$ = new AssignmentStatement(checked_ptr_cast<Expression>($3),
                                 static_cast<YYValType *>($1)->value);
    checked_ptr_cast<ASTNode>($$)
      ->setLineNumber(static_cast<YYValType *>($1)->linenum);
  }
  ;

init_statement
  : VAR IDENTIFIER ':' type_specifier '=' expression ';' {
    $$ = new InitializationStatement(checked_ptr_cast<Expression>($6),
                                     static_cast<YYValType *>($2)->value,
                                     static_cast<mu::ast::ASTTypeContainer*>($4)->getType());
    checked_ptr_cast<ASTNode>($$)
      ->setLineNumber(static_cast<YYValType *>($2)->linenum);
  }
  ;

/*** mu specifiers ***/
type_specifier
  : CHAR      { $$ = gASTTypeMap[mu::ast::enums::Type::sint8_mut];   }
  | UINT8     { $$ = gASTTypeMap[mu::ast::enums::Type::uint8_mut];   }
  | USHORT    { $$ = gASTTypeMap[mu::ast::enums::Type::uint16_mut];  }
  | UINT      { $$ = gASTTypeMap[mu::ast::enums::Type::uint32_mut];  }
  | ULONG     { $$ = gASTTypeMap[mu::ast::enums::Type::uint64_mut];  }
  | INT8      { $$ = gASTTypeMap[mu::ast::enums::Type::sint8_mut];   }
  | SHORT     { $$ = gASTTypeMap[mu::ast::enums::Type::sint16_mut];  }
  | INT       { $$ = gASTTypeMap[mu::ast::enums::Type::sint32_mut];  }
  | LONG      { $$ = gASTTypeMap[mu::ast::enums::Type::sint64_mut];  }
  | FLOAT     { $$ = gASTTypeMap[mu::ast::enums::Type::float32_mut]; }
  | DOUBLE    { $$ = gASTTypeMap[mu::ast::enums::Type::float64_mut]; }
  | BOOL      { $$ = gASTTypeMap[mu::ast::enums::Type::bool_mut];    }
  | TYPE_NAME
  ;

/*** mu expressions ***/
primary_expression
  : IDENTIFIER {
    $$ = new IdentifierExpression(static_cast<YYValType *>($1)->value);
    checked_ptr_cast<ASTNode>($$)
      ->setLineNumber(static_cast<YYValType *>($1)->linenum);
  }
  | CONSTANT {
    auto value = static_cast<YYValType *>($1)->value;
    auto tokText = static_cast<YYValType *>($1)->tokText;

    mu::ast::enums::ConstantType constantType = mu::ast::enums::ConstantType::Char;
    if (tokText == "CONSTANT_INT_HEX") {
      constantType = mu::ast::enums::ConstantType::IntKindHex;
    } else if (tokText == "CONSTANT_INT_KIND1") {
      constantType = mu::ast::enums::ConstantType::IntKind1;
    } else if (tokText == "CONSTANT_INT_KIND2") {
      constantType = mu::ast::enums::ConstantType::IntKind2;
    } else if (tokText == "CONSTANT_CHAR") {
      constantType = mu::ast::enums::ConstantType::Char;
    } else if (tokText == "CONSTANT_FLOAT_KIND1") {
      constantType = mu::ast::enums::ConstantType::FloatKind1;
    } else if (tokText == "CONSTANT_FLOAT_KIND2") {
      constantType = mu::ast::enums::ConstantType::FloatKind2;
    } else if (tokText == "CONSTANT_FLOAT_KIND3") {
      constantType = mu::ast::enums::ConstantType::FloatKind3;
    } else {
      yyerror("invalid constant kind");
      assert(false && "EXIT_FAILURE");
      exit(EXIT_FAILURE);
    }

    $$ = new ConstantExpression(value, constantType);
    checked_ptr_cast<ASTNode>($$)
      ->setLineNumber(static_cast<YYValType *>($1)->linenum);
  }
  | STRING_LITERAL {
    $$ = nullptr;
  }
  | '(' expression ')' {
    var E = std::unique_ptr<Expression>(checked_ptr_cast<Expression>($2));
    $$ = new ParenthesisExpression(std::move(E));
    checked_ptr_cast<ASTNode>($$)
      ->setLineNumber(static_cast<YYValType *>($1)->linenum);
  }
  | call_expression {
    $$ = $1;
  }
  ;

call_expression
  : IDENTIFIER '(' expression_list ')'  {
    var E =
        std::unique_ptr<ExpressionList>(checked_ptr_cast<ExpressionList>($3));
    $$ = new CallExpression(static_cast<YYValType *>($1)->value, std::move(E));
    checked_ptr_cast<ASTNode>($$)->setLineNumber(
        static_cast<YYValType *>($1)->linenum);
  }
  | IDENTIFIER '(' ')' {
    $$ = new CallExpression(static_cast<YYValType *>($1)->value);
    checked_ptr_cast<ASTNode>($$)->setLineNumber(
        static_cast<YYValType *>($1)->linenum);
  }
  ;

unary_expression
  : primary_expression {
    $$ = $1;
  }
  | '~' unary_expression {
    var E = std::unique_ptr<Expression>(checked_ptr_cast<Expression>($2));
    $$ = new UnaryExpression(UnaryOp::invertOp, std::move(E));
    checked_ptr_cast<ASTNode>($$)->setLineNumber(
        static_cast<YYValType *>($1)->linenum);
  }
  | '!' unary_expression {
    var E = std::unique_ptr<Expression>(checked_ptr_cast<Expression>($2));
    $$ = new UnaryExpression(UnaryOp::notOp, std::move(E));
    checked_ptr_cast<ASTNode>($$)
      ->setLineNumber(static_cast<YYValType *>($1)->linenum);
  }
  | '-' unary_expression {
    var E = std::unique_ptr<Expression>(checked_ptr_cast<Expression>($2));
    $$ = new UnaryExpression(UnaryOp::negOp, std::move(E));
    checked_ptr_cast<ASTNode>($$)
      ->setLineNumber(static_cast<YYValType *>($1)->linenum);
  }
  ;

expression_list
  : expression {
    $$ = new ExpressionList(checked_ptr_cast<Expression>($1));
  }
  | expression_list ',' expression {
    $$ = checked_ptr_cast<ExpressionList>($1)->
      append(checked_ptr_cast<Expression>($3));
    checked_ptr_cast<ASTNode>($$)
      ->setLineNumber(static_cast<YYValType *>($2)->linenum);
  }
  ;

expression
  : c_expression {
    $$ = $1;
  }
  ;

/*** grammar below are expression handling inherited from C: ***/
c_expression
  : logical_or_expression {
    $$ = $1;
  }
  ;

multiplicative_expression
  : unary_expression {
    $$ = $1;
  }
  | multiplicative_expression '*' unary_expression {
    $$ = new BinaryExpression(BinaryOp::mulOp, checked_ptr_cast<Expression>($1),
                              checked_ptr_cast<Expression>($3));
    checked_ptr_cast<ASTNode>($$)
      ->setLineNumber(static_cast<YYValType *>($2)->linenum);
  }
  | multiplicative_expression '/' unary_expression {
    $$ = new BinaryExpression(BinaryOp::divOp, checked_ptr_cast<Expression>($1),
                              checked_ptr_cast<Expression>($3));
    checked_ptr_cast<ASTNode>($$)
      ->setLineNumber(static_cast<YYValType *>($2)->linenum);
  }
  | multiplicative_expression '%' unary_expression {
    $$ = new BinaryExpression(BinaryOp::modOp, checked_ptr_cast<Expression>($1),
                              checked_ptr_cast<Expression>($3));
    checked_ptr_cast<ASTNode>($$)
      ->setLineNumber(static_cast<YYValType *>($2)->linenum);
  }
  ;

additive_expression
  : multiplicative_expression {
    $$ = $1;
  }
  | additive_expression '+' multiplicative_expression {
    $$ = new BinaryExpression(BinaryOp::addOp, checked_ptr_cast<Expression>($1),
                              checked_ptr_cast<Expression>($3));
    checked_ptr_cast<ASTNode>($$)
      ->setLineNumber(static_cast<YYValType *>($2)->linenum);
  }
  | additive_expression '-' multiplicative_expression {
    $$ = new BinaryExpression(BinaryOp::subOp, checked_ptr_cast<Expression>($1),
                              checked_ptr_cast<Expression>($3));
    checked_ptr_cast<ASTNode>($$)
      ->setLineNumber(static_cast<YYValType *>($2)->linenum);
  }
  ;

shift_expression
  : additive_expression {
    $$ = $1;
  }
  | shift_expression LEFT_OP additive_expression {
    $$ = new BinaryExpression(BinaryOp::lshOp, checked_ptr_cast<Expression>($1),
                              checked_ptr_cast<Expression>($3));
    checked_ptr_cast<ASTNode>($$)
      ->setLineNumber(static_cast<YYValType *>($2)->linenum);
  }
  | shift_expression RIGHT_OP additive_expression {
    $$ = new BinaryExpression(BinaryOp::rshOp, checked_ptr_cast<Expression>($1),
                              checked_ptr_cast<Expression>($3));
    checked_ptr_cast<ASTNode>($$)
      ->setLineNumber(static_cast<YYValType *>($2)->linenum);
  }
  ;

relational_expression
  : shift_expression {
    $$ = $1;
  }
  | relational_expression '<' shift_expression {
    $$ = new BinaryExpression(BinaryOp::ltOp, checked_ptr_cast<Expression>($1),
                              checked_ptr_cast<Expression>($3));
    checked_ptr_cast<ASTNode>($$)
      ->setLineNumber(static_cast<YYValType *>($2)->linenum);
  }
  | relational_expression '>' shift_expression {
    $$ = new BinaryExpression(BinaryOp::gtOp, checked_ptr_cast<Expression>($1),
                              checked_ptr_cast<Expression>($3));
    checked_ptr_cast<ASTNode>($$)
      ->setLineNumber(static_cast<YYValType *>($2)->linenum);
  }
  | relational_expression LE_OP shift_expression {
    $$ = new BinaryExpression(BinaryOp::leOp, checked_ptr_cast<Expression>($1),
                              checked_ptr_cast<Expression>($3));
    checked_ptr_cast<ASTNode>($$)
      ->setLineNumber(static_cast<YYValType *>($2)->linenum);
  }
  | relational_expression GE_OP shift_expression {
    $$ = new BinaryExpression(BinaryOp::geOp, checked_ptr_cast<Expression>($1),
                              checked_ptr_cast<Expression>($3));
    checked_ptr_cast<ASTNode>($$)
      ->setLineNumber(static_cast<YYValType *>($2)->linenum);
  }
  ;

equality_expression
  : relational_expression {
    $$ = $1;
  }
  | equality_expression EQ_OP relational_expression {
    $$ = new BinaryExpression(BinaryOp::eqOp, checked_ptr_cast<Expression>($1),
                              checked_ptr_cast<Expression>($3));
    checked_ptr_cast<ASTNode>($$)
      ->setLineNumber(static_cast<YYValType *>($2)->linenum);
  }
  | equality_expression NE_OP relational_expression {
    $$ = new BinaryExpression(BinaryOp::neOp, checked_ptr_cast<Expression>($1),
                              checked_ptr_cast<Expression>($3));
    checked_ptr_cast<ASTNode>($$)
      ->setLineNumber(static_cast<YYValType *>($2)->linenum);
  }
  ;

and_expression
  : equality_expression {
    $$ = $1;
  }
  | and_expression '&' equality_expression {
    $$ = new BinaryExpression(BinaryOp::andOp, checked_ptr_cast<Expression>($1),
                              checked_ptr_cast<Expression>($3));
    checked_ptr_cast<ASTNode>($$)
      ->setLineNumber(static_cast<YYValType *>($2)->linenum);
  }
  ;

exclusive_or_expression
  : and_expression {
    $$ = $1;
  }
  | exclusive_or_expression '^' and_expression {
    $$ = new BinaryExpression(BinaryOp::xorOp, checked_ptr_cast<Expression>($1),
                              checked_ptr_cast<Expression>($3));
    checked_ptr_cast<ASTNode>($$)
      ->setLineNumber(static_cast<YYValType *>($2)->linenum);
  }
  ;

inclusive_or_expression
  : exclusive_or_expression {
    $$ = $1;
  }
  | inclusive_or_expression '|' exclusive_or_expression {
    $$ = new BinaryExpression(BinaryOp::orOp, checked_ptr_cast<Expression>($1),
                              checked_ptr_cast<Expression>($3));
    checked_ptr_cast<ASTNode>($$)
      ->setLineNumber(static_cast<YYValType *>($2)->linenum);
  }
  ;

logical_and_expression
  : inclusive_or_expression {
    $$ = $1;
  }
  | logical_and_expression AND_OP inclusive_or_expression {
    $$ = new BinaryExpression(BinaryOp::andbOp, checked_ptr_cast<Expression>($1),
                              checked_ptr_cast<Expression>($3));
    checked_ptr_cast<ASTNode>($$)
      ->setLineNumber(static_cast<YYValType *>($2)->linenum);
  }
  ;

logical_or_expression
  : logical_and_expression {
    $$ = $1;
  }
  | logical_or_expression OR_OP logical_and_expression {
    $$ = new BinaryExpression(BinaryOp::orbOp, checked_ptr_cast<Expression>($1),
                              checked_ptr_cast<Expression>($3));
    checked_ptr_cast<ASTNode>($$)
      ->setLineNumber(static_cast<YYValType *>($2)->linenum);
  }
  ;

%%
#include <cstdio>

extern char yytext[];
extern int g_column;
extern int g_line;
extern std::string g_lastLine;

void yyerror(const char *s) {
  llvm::errs() << "error: parse error ("
               << s
               << ") at line " << g_line << " column "
               << g_column << ":\n";
  llvm::errs() << g_lastLine << "\n";
  // fprintf(stderr, "\n%*s\n%*s\n", g_column, "^", g_column, s);
}
