%{
#include <cstdio>
#include <cstdlib>
#include "ast.h"

int yylex();
void yyerror(const char *s);
TranslationUnitDecl *topnode;

#define YYSTYPE void*
%}

%token TYPE LET FUNCTION
%token IDENTIFIER CONSTANT STRING_LITERAL

%token PTR_OP

%token LEFT_OP RIGHT_OP
%token AND_OP OR_OP

%token LE_OP GE_OP EQ_OP NE_OP
%token TYPE_NAME

%token CHAR SHORT INT LONG SIGNED UNSIGNED FLOAT DOUBLE CONST VOLATILE VOID
%token IF ELSE WHILE RETURN

%start translation_unit
%%

translation_unit: external_declaration_list ;
external_declaration_list
  : external_declaration
  | external_declaration_list external_declaration
  ;
external_declaration
  : mu_function_definition
  | function_definition
  ;

mu_function_definition
  : FUNCTION IDENTIFIER '(' identifier_list ')' PTR_OP type_specifier compound_statement
  | FUNCTION IDENTIFIER '(' ')' PTR_OP  type_specifier compound_statement
  ;

function_definition
  : declaration_specifiers declarator declaration_list compound_statement
  | declaration_specifiers declarator compound_statement
  | declarator declaration_list compound_statement
  | declarator compound_statement
  ;

parameter_list: parameter_declaration | parameter_list ',' parameter_declaration ;
identifier_list: IDENTIFIER | identifier_list ',' IDENTIFIER ;
parameter_type_list: parameter_list ;

/*** mu statements ***/
expression_statement: ';' | expression ';' ;
statement: compound_statement | expression_statement | selection_statement | iteration_statement | jump_statement ;
statement_list: statement | statement_list statement ;
jump_statement: RETURN ';' | RETURN expression ';' ;

compound_statement
  : '{' '}'
  | '{' statement_list '}'
  | '{' declaration_list '}'
  | '{' declaration_list statement_list '}'
  ;

selection_statement
  : IF '(' expression ')' statement
  | IF '(' expression ')' statement ELSE statement
  ;

iteration_statement: WHILE '(' expression ')' statement ;

/*** mu specifiers ***/
declaration_specifiers: type_specifier | type_specifier declaration_specifiers ;
type_specifier: VOID | CHAR | SHORT | INT | LONG | FLOAT | DOUBLE | SIGNED | UNSIGNED | TYPE_NAME ;

/*** mu initializers ***/
initializer: assignment_expression ;
init_declarator: declarator | declarator '=' initializer ;
init_declarator_list : init_declarator | init_declarator_list ',' init_declarator ;

/*** mu operators ***/
unary_operator: '&' | '*' | '+' | '-' | '~' | '!' ;
assignment_operator: '=' ;

/*** mu decls ***/
declarator: direct_declarator ;
declaration: declaration_specifiers ';' | declaration_specifiers init_declarator_list ';' ;
declaration_list: declaration | declaration_list declaration ;
parameter_declaration: declaration_specifiers declarator | declaration_specifiers ;

/*** mu expressions ***/
primary_expression: IDENTIFIER | CONSTANT | STRING_LITERAL | '(' expression ')' ;
postfix_expression: primary_expression ;
conditional_expression: logical_or_expression ;
constant_expression: conditional_expression ;
unary_expression: postfix_expression | unary_operator cast_expression ;
cast_expression: unary_expression ;
assignment_expression: conditional_expression | unary_expression assignment_operator assignment_expression ;
expression: assignment_expression | expression ',' assignment_expression;

/*** syntax ripped out of C and into mu above ***/

direct_declarator
  : IDENTIFIER
  | '(' declarator ')'
  | direct_declarator '[' constant_expression ']'
  | direct_declarator '[' ']'
  | direct_declarator '(' parameter_type_list ')'
  | direct_declarator '(' identifier_list ')'
  | direct_declarator '(' ')'
  ;


/*** grammar below are expression handling inherited from C: ***/

multiplicative_expression
  : cast_expression
  | multiplicative_expression '*' cast_expression
  | multiplicative_expression '/' cast_expression
  | multiplicative_expression '%' cast_expression
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
  printf("\n%*s\n%*s\n", g_column, "^", g_column, s);
}
