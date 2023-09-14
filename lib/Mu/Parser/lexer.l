D  [0-9]
L  [a-zA-Z_]
H  [a-fA-F0-9]
E  [Ee][+-]?{D}+
FS (f|F|l|L)
IS (u|U|l|L)*

%{
#include <stdio.h>
#include <string>

#define YYSTYPE void*

#include "parser.hpp"
#include "Mu/Parser/yyvaltype.h"

using namespace mu::ast;

int check_type();
void count();
void comment();
void comment2();

static const char *identifier_str = "IDENTIFIER";
static const char *constant_str = "CONSTANT";
static const char *keyword_str = "KEYWORD";
static const char *type_str = "TYPE";
static const char *strliteral_str = "STRING_LITERAL";
static const char *operator_str = "OPERATOR";
static const char *puctuation_str = "PUCTUATION";

unsigned g_line = 1;
unsigned g_column = 1;
std::string g_lastLine = "";

#define COPY_YYVAL(num, text, kindstr) \
  yylval = yytext; \
  yylval = (void*) makeYYValType(num, text, kindstr)

%}

%%
"/*"                   { comment(); }
"//"                   { comment2(); }

"var"                  { count(); COPY_YYVAL(g_line, yytext, keyword_str); return(VAR); }
"fn"                   { count(); COPY_YYVAL(g_line, yytext, keyword_str); return(FUNCTION); }

"else"                 { count(); COPY_YYVAL(g_line, yytext, keyword_str); return(ELSE); }
"if"                   { count(); COPY_YYVAL(g_line, yytext, keyword_str); return(IF); }
"return"               { count(); COPY_YYVAL(g_line, yytext, keyword_str); return(RETURN); }
"while"                { count(); COPY_YYVAL(g_line, yytext, keyword_str); return(WHILE); }

"char"                 { count(); COPY_YYVAL(g_line, yytext, type_str); return(CHAR); }
"float64"              { count(); COPY_YYVAL(g_line, yytext, type_str); return(DOUBLE); }
"float32"              { count(); COPY_YYVAL(g_line, yytext, type_str); return(FLOAT); }
"int32"                { count(); COPY_YYVAL(g_line, yytext, type_str); return(INT); }
"int64"                { count(); COPY_YYVAL(g_line, yytext, type_str); return(LONG); }
"int16"                { count(); COPY_YYVAL(g_line, yytext, type_str); return(SHORT); }
"int8"                 { count(); COPY_YYVAL(g_line, yytext, type_str); return(INT8); }
"uint8"                { count(); COPY_YYVAL(g_line, yytext, type_str); return(UINT8); }
"uint32"               { count(); COPY_YYVAL(g_line, yytext, type_str); return(UINT); }
"uint64"               { count(); COPY_YYVAL(g_line, yytext, type_str); return(ULONG); }
"uint16"               { count(); COPY_YYVAL(g_line, yytext, type_str); return(USHORT); }

{L}({L}|{D})*          { count(); COPY_YYVAL(g_line, yytext, identifier_str); return(check_type()); }

0[xX]{H}+{IS}?         { count(); COPY_YYVAL(g_line, yytext, constant_str); return(CONSTANT); }
0{D}+{IS}?             { count(); COPY_YYVAL(g_line, yytext, constant_str); return(CONSTANT); }
{D}+{IS}?              { count(); COPY_YYVAL(g_line, yytext, constant_str); return(CONSTANT); }
L?'(\\.|[^\\'])+'      { count(); COPY_YYVAL(g_line, yytext, constant_str); return(CONSTANT); }

{D}+{E}{FS}?           { count(); COPY_YYVAL(g_line, yytext, constant_str); return(CONSTANT); }
{D}*"."{D}+({E})?{FS}? { count(); COPY_YYVAL(g_line, yytext, constant_str); return(CONSTANT); }
{D}+"."{D}*({E})?{FS}? { count(); COPY_YYVAL(g_line, yytext, constant_str); return(CONSTANT); }

L?\"(\\.|[^\\"])*\"    { count(); COPY_YYVAL(g_line, yytext, strliteral_str); return(STRING_LITERAL); }

">>"                   { count(); COPY_YYVAL(g_line, yytext, operator_str);  return(RIGHT_OP); }
"<<"                   { count(); COPY_YYVAL(g_line, yytext, operator_str);  return(LEFT_OP); }
"->"                   { count(); COPY_YYVAL(g_line, yytext, operator_str);  return(PTR_OP); }
"&&"                   { count(); COPY_YYVAL(g_line, yytext, operator_str);  return(AND_OP); }
"||"                   { count(); COPY_YYVAL(g_line, yytext, operator_str);  return(OR_OP); }
"<="                   { count(); COPY_YYVAL(g_line, yytext, operator_str);  return(LE_OP); }
">="                   { count(); COPY_YYVAL(g_line, yytext, operator_str);  return(GE_OP); }
"=="                   { count(); COPY_YYVAL(g_line, yytext, operator_str);  return(EQ_OP); }
"!="                   { count(); COPY_YYVAL(g_line, yytext, operator_str);  return(NE_OP); }

"&"                    { count(); COPY_YYVAL(g_line, yytext, operator_str);  return('&'); }
"!"                    { count(); COPY_YYVAL(g_line, yytext, operator_str);  return('!'); }
"~"                    { count(); COPY_YYVAL(g_line, yytext, operator_str);  return('~'); }
"-"                    { count(); COPY_YYVAL(g_line, yytext, operator_str);  return('-'); }
"+"                    { count(); COPY_YYVAL(g_line, yytext, operator_str);  return('+'); }
"*"                    { count(); COPY_YYVAL(g_line, yytext, operator_str);  return('*'); }
"/"                    { count(); COPY_YYVAL(g_line, yytext, operator_str);  return('/'); }
"%"                    { count(); COPY_YYVAL(g_line, yytext, operator_str);  return('%'); }
"<"                    { count(); COPY_YYVAL(g_line, yytext, operator_str);  return('<'); }
">"                    { count(); COPY_YYVAL(g_line, yytext, operator_str);  return('>'); }
"^"                    { count(); COPY_YYVAL(g_line, yytext, operator_str);  return('^'); }
"|"                    { count(); COPY_YYVAL(g_line, yytext, operator_str);  return('|'); }
"?"                    { count(); COPY_YYVAL(g_line, yytext, operator_str);  return('?'); }

";"                    { count(); COPY_YYVAL(g_line, yytext, puctuation_str);  return(';'); }
("{"|"<%")             { count(); COPY_YYVAL(g_line, yytext, puctuation_str);  return('{'); }
("}"|"%>")             { count(); COPY_YYVAL(g_line, yytext, puctuation_str);  return('}'); }
","                    { count(); COPY_YYVAL(g_line, yytext, puctuation_str);  return(','); }
":"                    { count(); COPY_YYVAL(g_line, yytext, puctuation_str);  return(':'); }
"="                    { count(); COPY_YYVAL(g_line, yytext, puctuation_str);  return('='); }
"("                    { count(); COPY_YYVAL(g_line, yytext, puctuation_str);  return('('); }
")"                    { count(); COPY_YYVAL(g_line, yytext, puctuation_str);  return(')'); }
("["|"<:")             { count(); COPY_YYVAL(g_line, yytext, puctuation_str);  return('['); }
("]"|":>")             { count(); COPY_YYVAL(g_line, yytext, puctuation_str);  return(']'); }
"."                    { count(); COPY_YYVAL(g_line, yytext, puctuation_str);  return('.'); }

[ \t\v\n\f]            { count(); }
[\n]                   { g_line++; count(); }
.                      { fprintf(stderr, "Unrecognized character: '%c'\n", yytext[0]); }

%%

int yywrap() { return 1; }

#ifdef __cplusplus
#define YYINPUT yyinput
#else
#define YYINPUT input
#endif

void comment2() {
  char c;

  while (true) {
    while ((c = YYINPUT()) != '\n' && c != 0) {
      // putchar(c)
      if (c == '\n') {
        g_line++;
      }
    }
    break;
  }
}

void comment() {
  char c, c1;

  while (true) {
    while ((c = YYINPUT()) != '*' && c != 0) {
      // putchar(c);
      if (c == '\n') {
        g_line++;
      }
    }


    if ((c1 = YYINPUT()) != '/' && c != 0) {
      // unput(c1);
      if (c1 == '\n') {
        g_line++;
      }
      continue;
    }

    break;
  }

  // if (c != 0)
  //   putchar(c1);
}


void count() {
  int i;

  for (i = 0; yytext[i] != '\0'; i++) {
    if (yytext[i] == '\n') {
      g_lastLine = "";
      g_line++;
      g_column = 0;
    } else if (yytext[i] == '\t') {
      g_lastLine += '\t';
      g_column += 8 - (g_column % 8);
    } else {
      g_lastLine += yytext[i];
      g_column++;
    }
  }

  // Remove the ECHO to skip echoing the input
  // ECHO;
}

int check_type() {
  /// pseudo code --- this is what it should check
  ///
  /// if (yytext == type_name)
  ///   return(TYPE_NAME);
  /// return(IDENTIFIER);
  ///
  /// it actually will only return IDENTIFIER
  return IDENTIFIER;
}

/* Ideas or supporting recursive types:
  
  enum VariantTag { ... };
  struct Pair;
  
  struct Variant {
    VariantTag tag;
    union u {
      char c,
      uint32_t ui32;
      Pair *cons;
      ...
    };
  };
  
  Struct Pair {
    Variant v1;
    Variant v2;
  };
  
  Pair *make_pair(Variant v1, Variant v2) {
    Pair p;
    p.v1 = v1;
    p.v2 = v2;
    return p;
  }
  
  List a = make_pair(1, make_pair(2, make_pair(3, nullptr)));
*/
