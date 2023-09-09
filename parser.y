%{
#include <cstdio>
#include <cstdlib>
#include "ast.h"
#include "yyvaltype.h"

#if 1
#define debug_print(...) printf(__VA_ARGS__)
#else
#define debug_print(...)
#endif

int yylex();
void yyerror(const char *s);

#define YYSTYPE void*

TranslationUnit *topnode;
%}

%token VAR FUNCTION
%token PTR_OP
%token LEFT_OP RIGHT_OP
%token AND_OP OR_OP
%token LE_OP GE_OP EQ_OP NE_OP
%token IDENTIFIER CONSTANT STRING_LITERAL
%token TYPE_NAME
%token CHAR SHORT INT LONG FLOAT DOUBLE USHORT UINT ULONG INT8 UINT8
%token IF ELSE WHILE RETURN

%start translation_unit
%%

/*** mu top-level constructs ***/
translation_unit
  : toplevel_declaration_list {
    topnode = new TranslationUnit(static_cast<DefunList *>($1));
  }
  ;

toplevel_declaration_list
  : toplevel_declaration {
    $$ = new DefunList(static_cast<Defun *>($1));
  }
  | toplevel_declaration_list toplevel_declaration {
    $$ = static_cast<DefunList *>($1)->append(static_cast<Defun *>($2));
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
    $$ = new Defun(static_cast<yyvalType *>($2)->value,
                   static_cast<ParamList *>($4), Type::sint32_mut,
                   static_cast<CompoundStatement *>($8));
    static_cast<ASTNode*>($$)
      ->setLineNumber(static_cast<yyvalType *>($2)->linenum);
  }
  | FUNCTION IDENTIFIER '(' ')' PTR_OP  type_specifier compound_statement {
    $$ = new Defun(static_cast<yyvalType *>($2)->value, nullptr,
                   Type::sint32_mut, static_cast<CompoundStatement *>($7));
    static_cast<ASTNode*>($$)
      ->setLineNumber(static_cast<yyvalType *>($2)->linenum);
  }
  ;

parameter_list
  : parameter_declaration {
    $$ = new ParamList(static_cast<ParamDecl *>($1));
    static_cast<ASTNode*>($$)
      ->setLineNumber(static_cast<ParamDecl *>($1)->getLineNumber());
  }
  | parameter_list ',' parameter_declaration {
    $$ = static_cast<ParamList *>($1)->append(static_cast<ParamDecl *>($3));
    static_cast<ASTNode*>($$)
      ->setLineNumber(static_cast<yyvalType *>($2)->linenum);
  }
  ;

parameter_declaration
  : IDENTIFIER ':' type_specifier {
    $$ = new ParamDecl(static_cast<yyvalType *>($1)->value, Type::sint32_mut);
    static_cast<ASTNode*>($$)
      ->setLineNumber(static_cast<yyvalType *>($2)->linenum);
  }
  ;

/*** mu statements ***/
statement_list
  : statement {
    $$ = new StatementList(static_cast<Statement *>($1));
  }
  | statement_list statement {
    $$ = static_cast<StatementList *>($1)->append(static_cast<Statement *>($2));
  }
  ;

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
    static_cast<ASTNode*>($$)
      ->setLineNumber(static_cast<yyvalType *>($1)->linenum);
  }
  | '{' statement_list '}' {
    $$ = new CompoundStatement(static_cast<StatementList *>($2));
    static_cast<ASTNode*>($$)
      ->setLineNumber(static_cast<yyvalType *>($1)->linenum);
  }
  ;

selection_statement
  : IF expression compound_statement {
    $$ = new SelectionIfStatement(static_cast<Expression *>($2),
                                  static_cast<CompoundStatement *>($3));
    static_cast<ASTNode*>($$)
      ->setLineNumber(static_cast<yyvalType *>($1)->linenum);
  }
  | IF expression compound_statement ELSE compound_statement {
    $$ = new SelectionIfStatement(static_cast<Expression *>($2),
                                  static_cast<CompoundStatement *>($3),
                                  static_cast<CompoundStatement *>($5));
    static_cast<ASTNode*>($$)
      ->setLineNumber(static_cast<yyvalType *>($1)->linenum);
  }
  ;

iteration_statement
  : WHILE expression compound_statement {
    $$ = new IterationWhileStatement(static_cast<Expression *>($2),
                                     static_cast<CompoundStatement *>($3));
    static_cast<ASTNode*>($$)
      ->setLineNumber(static_cast<yyvalType *>($1)->linenum);
  }
  ;

jump_statement
  : RETURN expression ';' {
    $$ = new JumpReturnStatement(static_cast<Expression*>($2));
    static_cast<ASTNode*>($$)
      ->setLineNumber(static_cast<yyvalType *>($1)->linenum);
  }
  ;

assignment_statement
  : IDENTIFIER '=' expression ';' {
    $$ = new AssignmentStatement(static_cast<Expression *>($3),
                                 static_cast<yyvalType *>($1)->value);
    static_cast<ASTNode*>($$)
      ->setLineNumber(static_cast<yyvalType *>($1)->linenum);
  }
  ;

init_statement
  : VAR IDENTIFIER ':' type_specifier '=' expression ';' {
    $$ = new InitializationStatement(static_cast<Expression *>($6),
                                     static_cast<yyvalType *>($2)->value,
                                     Type::sint32_mut);
    static_cast<ASTNode*>($$)
      ->setLineNumber(static_cast<yyvalType *>($2)->linenum);
  }
  ;

/*** mu specifiers ***/
type_specifier
  : CHAR
  | UINT8
  | USHORT
  | UINT
  | ULONG
  | INT8
  | SHORT
  | INT
  | LONG
  | FLOAT
  | DOUBLE
  | TYPE_NAME
  ;

/*** mu expressions ***/
primary_expression
  : IDENTIFIER {
    $$ = new IdentifierExpression(static_cast<yyvalType *>($1)->value);
    static_cast<ASTNode*>($$)
      ->setLineNumber(static_cast<yyvalType *>($1)->linenum);
  }
  | CONSTANT {
    $$ = new ConstantExpression(static_cast<yyvalType *>($1)->value);
    static_cast<ASTNode*>($$)
      ->setLineNumber(static_cast<yyvalType *>($1)->linenum);
  }
  | STRING_LITERAL {
    $$ = nullptr;
  }
  | '(' expression ')' {
    $$ = new ParenthesisExpression(static_cast<Expression *>($2));
    static_cast<ASTNode*>($$)
      ->setLineNumber(static_cast<yyvalType *>($1)->linenum);
  }
  | call_expression {
    $$ = $1;
  }
  ;

call_expression
  : IDENTIFIER '(' expression_list ')'  {
    $$ = new CallExpression(static_cast<yyvalType *>($1)->value,
                            static_cast<ExpressionList*>($3));
    static_cast<ASTNode*>($$)
      ->setLineNumber(static_cast<yyvalType *>($1)->linenum);
  }
  | IDENTIFIER '(' ')' {
    $$ = new CallExpression(static_cast<yyvalType *>($1)->value);
    static_cast<ASTNode*>($$)
      ->setLineNumber(static_cast<yyvalType *>($1)->linenum);
  }
  ;

unary_expression
  : primary_expression {
    $$ = $1;
  }
  | '~' unary_expression {
    $$ = new UnaryExpression(UnaryOp::invertOp, static_cast<Expression*>($2));
    static_cast<ASTNode*>($$)
      ->setLineNumber(static_cast<yyvalType *>($1)->linenum);
  }
  | '!' unary_expression {
    $$ = new UnaryExpression(UnaryOp::notOp, static_cast<Expression*>($2));
    static_cast<ASTNode*>($$)
      ->setLineNumber(static_cast<yyvalType *>($1)->linenum);
  }
  | '-' unary_expression {
    $$ = new UnaryExpression(UnaryOp::negOp, static_cast<Expression*>($2));
    static_cast<ASTNode*>($$)
      ->setLineNumber(static_cast<yyvalType *>($1)->linenum);
  }
  ;

expression_list
  : expression {
    $$ = new ExpressionList(static_cast<Expression *>($1));
  }
  | expression_list ',' expression {
    $$ = static_cast<ExpressionList *>($1)->
      append(static_cast<Expression *>($3));
    static_cast<ASTNode*>($$)
      ->setLineNumber(static_cast<yyvalType *>($2)->linenum);
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
    $$ = new BinaryExpression(BinaryOp::mulOp, static_cast<Expression *>($1),
                              static_cast<Expression *>($3));
    static_cast<ASTNode*>($$)
      ->setLineNumber(static_cast<yyvalType *>($2)->linenum);
  }
  | multiplicative_expression '/' unary_expression {
    $$ = new BinaryExpression(BinaryOp::divOp, static_cast<Expression *>($1),
                              static_cast<Expression *>($3));
    static_cast<ASTNode*>($$)
      ->setLineNumber(static_cast<yyvalType *>($2)->linenum);
  }
  | multiplicative_expression '%' unary_expression {
    $$ = new BinaryExpression(BinaryOp::modOp, static_cast<Expression *>($1),
                              static_cast<Expression *>($3));
    static_cast<ASTNode*>($$)
      ->setLineNumber(static_cast<yyvalType *>($2)->linenum);
  }
  ;

additive_expression
  : multiplicative_expression {
    $$ = $1;
  }
  | additive_expression '+' multiplicative_expression {
    $$ = new BinaryExpression(BinaryOp::addOp, static_cast<Expression *>($1),
                              static_cast<Expression *>($3));
    static_cast<ASTNode*>($$)
      ->setLineNumber(static_cast<yyvalType *>($2)->linenum);
  }
  | additive_expression '-' multiplicative_expression {
    $$ = new BinaryExpression(BinaryOp::subOp, static_cast<Expression *>($1),
                              static_cast<Expression *>($3));
    static_cast<ASTNode*>($$)
      ->setLineNumber(static_cast<yyvalType *>($2)->linenum);
  }
  ;

shift_expression
  : additive_expression {
    $$ = $1;
  }
  | shift_expression LEFT_OP additive_expression {
    $$ = new BinaryExpression(BinaryOp::lshOp, static_cast<Expression *>($1),
                              static_cast<Expression *>($3));
    static_cast<ASTNode*>($$)
      ->setLineNumber(static_cast<yyvalType *>($2)->linenum);
  }
  | shift_expression RIGHT_OP additive_expression {
    $$ = new BinaryExpression(BinaryOp::rshOp, static_cast<Expression *>($1),
                              static_cast<Expression *>($3));
    static_cast<ASTNode*>($$)
      ->setLineNumber(static_cast<yyvalType *>($2)->linenum);
  }
  ;

relational_expression
  : shift_expression {
    $$ = $1;
  }
  | relational_expression '<' shift_expression {
    $$ = new BinaryExpression(BinaryOp::ltOp, static_cast<Expression *>($1),
                              static_cast<Expression *>($3));
    static_cast<ASTNode*>($$)
      ->setLineNumber(static_cast<yyvalType *>($2)->linenum);
  }
  | relational_expression '>' shift_expression {
    $$ = new BinaryExpression(BinaryOp::gtOp, static_cast<Expression *>($1),
                              static_cast<Expression *>($3));
    static_cast<ASTNode*>($$)
      ->setLineNumber(static_cast<yyvalType *>($2)->linenum);
  }
  | relational_expression LE_OP shift_expression {
    $$ = new BinaryExpression(BinaryOp::leOp, static_cast<Expression *>($1),
                              static_cast<Expression *>($3));
    static_cast<ASTNode*>($$)
      ->setLineNumber(static_cast<yyvalType *>($2)->linenum);
  }
  | relational_expression GE_OP shift_expression {
    $$ = new BinaryExpression(BinaryOp::geOp, static_cast<Expression *>($1),
                              static_cast<Expression *>($3));
    static_cast<ASTNode*>($$)
      ->setLineNumber(static_cast<yyvalType *>($2)->linenum);
  }
  ;

equality_expression
  : relational_expression {
    $$ = $1;
  }
  | equality_expression EQ_OP relational_expression {
    $$ = new BinaryExpression(BinaryOp::eqOp, static_cast<Expression *>($1),
                              static_cast<Expression *>($3));
    static_cast<ASTNode*>($$)
      ->setLineNumber(static_cast<yyvalType *>($2)->linenum);
  }
  | equality_expression NE_OP relational_expression {
    $$ = new BinaryExpression(BinaryOp::neOp, static_cast<Expression *>($1),
                              static_cast<Expression *>($3));
    static_cast<ASTNode*>($$)
      ->setLineNumber(static_cast<yyvalType *>($2)->linenum);
  }
  ;

and_expression
  : equality_expression {
    $$ = $1;
  }
  | and_expression '&' equality_expression {
    $$ = new BinaryExpression(BinaryOp::andOp, static_cast<Expression *>($1),
                              static_cast<Expression *>($3));
    static_cast<ASTNode*>($$)
      ->setLineNumber(static_cast<yyvalType *>($2)->linenum);
  }
  ;

exclusive_or_expression
  : and_expression {
    $$ = $1;
  }
  | exclusive_or_expression '^' and_expression {
    $$ = new BinaryExpression(BinaryOp::xorOp, static_cast<Expression *>($1),
                              static_cast<Expression *>($3));
    static_cast<ASTNode*>($$)
      ->setLineNumber(static_cast<yyvalType *>($2)->linenum);
  }
  ;

inclusive_or_expression
  : exclusive_or_expression {
    $$ = $1;
  }
  | inclusive_or_expression '|' exclusive_or_expression {
    $$ = new BinaryExpression(BinaryOp::orOp, static_cast<Expression *>($1),
                              static_cast<Expression *>($3));
    static_cast<ASTNode*>($$)
      ->setLineNumber(static_cast<yyvalType *>($2)->linenum);
  }
  ;

logical_and_expression
  : inclusive_or_expression {
    $$ = $1;
  }
  | logical_and_expression AND_OP inclusive_or_expression {
    $$ = new BinaryExpression(BinaryOp::andbOp, static_cast<Expression *>($1),
                              static_cast<Expression *>($3));
    static_cast<ASTNode*>($$)
      ->setLineNumber(static_cast<yyvalType *>($2)->linenum);
  }
  ;

logical_or_expression
  : logical_and_expression {
    $$ = $1;
  }
  | logical_or_expression OR_OP logical_and_expression {
    $$ = new BinaryExpression(BinaryOp::orbOp, static_cast<Expression *>($1),
                              static_cast<Expression *>($3));
    static_cast<ASTNode*>($$)
      ->setLineNumber(static_cast<yyvalType *>($2)->linenum);
  }
  ;

%%
#include <cstdio>

extern char yytext[];
extern int g_column;
extern int g_line;
extern std::string g_lastLine;

void yyerror(const char *s) {
  std::cerr << "error: parse error at line " << g_line << " column "
            << g_column << ":\n";
  std::cerr << g_lastLine << "\n";
  fprintf(stderr, "\n%*s\n%*s\n", g_column, "^", g_column, s);
}
