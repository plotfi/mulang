%{
#include <cstdio>
#include <cstdlib>
#include "ast.h"

#if 0
#define debug_print(...) printf(__VA_ARGS__)
#else
#define debug_print(...)
#endif

int yylex();
void yyerror(const char *s);
TranslationUnit *topnode;

typedef struct yyvalType {
  int linenum;
  char *value;
  char *tokText;
} yyvalType;

yyvalType *makeyyvalType(int linenum, char *value, char *token);

#define YYSTYPE void*
%}

%token VAR FUNCTION
%token IDENTIFIER CONSTANT STRING_LITERAL

%token PTR_OP

%token LEFT_OP RIGHT_OP
%token AND_OP OR_OP
%token LE_OP GE_OP EQ_OP NE_OP

%token TYPE_NAME
%token CHAR SHORT INT LONG SIGNED UNSIGNED FLOAT DOUBLE

%token IF ELSE WHILE RETURN

%start translation_unit
%%

/*** syntax ripped out of C and into mu above ***/

/*** mu top-level constructs ***/
translation_unit: toplevel_declaration_list {
  topnode = new TranslationUnit(static_cast<DefunList*>($1));
};

toplevel_declaration_list:
  toplevel_declaration {
    $$ = new DefunList(static_cast<Defun*>($1));
  }
  | toplevel_declaration_list toplevel_declaration {
    $$ = static_cast<DefunList*>($1)->append(static_cast<Defun*>($2));
  };

toplevel_declaration: mu_function_definition { $$ = $1; };

mu_function_definition
  : FUNCTION IDENTIFIER '(' parameter_list ')' PTR_OP type_specifier compound_statement {
    $$ = new Defun(static_cast<yyvalType*>($2)->value);
  }
  | FUNCTION IDENTIFIER '(' ')' PTR_OP  type_specifier compound_statement {
    $$ = new Defun(static_cast<yyvalType*>($2)->value);
  }
  | FUNCTION IDENTIFIER '(' parameter_list ')' compound_statement {
    $$ = new Defun(static_cast<yyvalType*>($2)->value);

  }
  | FUNCTION IDENTIFIER '(' ')' compound_statement {
    $$ = new Defun(static_cast<yyvalType*>($2)->value);
  }
  ;
parameter_list: parameter_declaration | parameter_list ',' parameter_declaration ;
parameter_declaration: IDENTIFIER ':' type_specifier ;

/*** mu statements ***/
statement_list: statement | statement_list statement ;
statement:
  compound_statement {
    debug_print("\n\n>> statement -> compound_statement");
    debug_print("...  dolla: %s", static_cast<char*>($1));
  }
  | selection_statement {
    debug_print("\n\n>> statement -> selection_statement");
    debug_print("...  dolla: %s", static_cast<char*>($1));
  }
  | iteration_statement {
    debug_print("\n\n>> statement -> iteration_statement");
    debug_print("...  dolla: %s", static_cast<char*>($1));
  }
  | jump_statement {
    debug_print("\n\n>> statement -> jump_statement");
    debug_print("...  dolla: %s", static_cast<char*>($1));
  }
  | assignment_statement {
    debug_print("\n\n>> statement -> assignment_statement");
    debug_print("...  dolla: %s", static_cast<char*>($1));
  }
  | init_statement {
    debug_print("\n\n>> statement -> init_statement");
    debug_print("...  dolla: %s", static_cast<char*>($1));
  };

compound_statement
  : '{' '}'
  | '{' statement_list '}'
  ;

selection_statement
  : IF expression compound_statement
  | IF expression compound_statement ELSE compound_statement
  ;

iteration_statement: WHILE expression compound_statement
jump_statement:
  RETURN expression ';' {
    debug_print("\n\n>> RETURN <expression>;");
    debug_print("...  dolla: %s", static_cast<char*>($1));
  } ;

assignment_statement: IDENTIFIER '=' expression ';' ;
init_statement: VAR IDENTIFIER ':' type_specifier '=' expression ';' ;

/*** mu specifiers ***/
type_specifier: CHAR | SHORT | INT | LONG | FLOAT | DOUBLE | SIGNED | UNSIGNED | TYPE_NAME ;

/*** mu expressions ***/
primary_expression
  : IDENTIFIER
  | CONSTANT
  | STRING_LITERAL
  | '(' expression ')'
  | call_expression
  ;

call_expression
  : IDENTIFIER '(' expression_list ')'
  | IDENTIFIER '(' ')' {
    debug_print("\n\n>> call_expression!!");
    debug_print("...  dolla: %s", static_cast<char*>($1));
  }
  ;

unary_expression
  : primary_expression
  | '~' unary_expression
  | '!' unary_expression
  | '-' unary_expression
  ;

expression_list: expression | expression_list ',' expression ;
expression:
  c_expression {
    debug_print("\n\n>> expression -> c_expression");
    debug_print("...  dolla: %s", static_cast<char*>($1));
  };

/*** grammar below are expression handling inherited from C: ***/
c_expression: logical_or_expression ;

multiplicative_expression
  : unary_expression
  | multiplicative_expression '*' unary_expression
  | multiplicative_expression '/' unary_expression
  | multiplicative_expression '%' unary_expression
  ;

additive_expression
  : multiplicative_expression
  | additive_expression '+' multiplicative_expression
  | additive_expression '-' multiplicative_expression
  ;

shift_expression
  : additive_expression
  | shift_expression LEFT_OP additive_expression
  | shift_expression RIGHT_OP additive_expression
  ;

relational_expression
  : shift_expression
  | relational_expression '<' shift_expression
  | relational_expression '>' shift_expression
  | relational_expression LE_OP shift_expression
  | relational_expression GE_OP shift_expression
  ;

equality_expression
  : relational_expression
  | equality_expression EQ_OP relational_expression
  | equality_expression NE_OP relational_expression
  ;

and_expression
  : equality_expression
  | and_expression '&' equality_expression
  ;

exclusive_or_expression
  : and_expression
  | exclusive_or_expression '^' and_expression
  ;

inclusive_or_expression
  : exclusive_or_expression
  | inclusive_or_expression '|' exclusive_or_expression
  ;

logical_and_expression
  : inclusive_or_expression
  | logical_and_expression AND_OP inclusive_or_expression
  ;

logical_or_expression
  : logical_and_expression
  | logical_or_expression OR_OP logical_and_expression
  ;

%%
#include <cstdio>

extern char yytext[];
extern int g_column;

void yyerror(const char *s) {
  fflush(stdout);
  debug_print("\n%*s\n%*s\n", g_column, "^", g_column, s);
}
