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
#include "yyvaltype.h"

int check_type();
void count();
void comment();
void comment2();

static const char *identifier_str = "IDENTIFIER";
int linenum = 0;
%}

%%
"/*"                   { comment(); }
"//"                   { comment2(); }

"var"                  { count(); return(VAR); }
"fn"                   { count(); return(FUNCTION); }

"else"                 { count(); return(ELSE); }
"if"                   { count(); return(IF); }
"return"               { count(); return(RETURN); }
"while"                { count(); return(WHILE); }

"char"                 { count(); return(CHAR); }
"float64"              { count(); return(DOUBLE); }
"float32"              { count(); return(FLOAT); }
"int32"                { count(); return(INT); }
"int64"                { count(); return(LONG); }
"int16"                { count(); return(SHORT); }
"int8"                 { count(); return(INT8); }
"uint8"                { count(); return(UINT8); }
"uint32"               { count(); return(UINT); }
"uint64"               { count(); return(ULONG); }
"uint16"               { count(); return(USHORT); }

{L}({L}|{D})*          { count();
                         yylval = yytext;
                         yylval =
                           (void*)
                           makeyyvalType(linenum, yytext, identifier_str);
                         return(check_type()); }

0[xX]{H}+{IS}?         { count(); return(CONSTANT); }
0{D}+{IS}?             { count(); return(CONSTANT); }
{D}+{IS}?              { count(); return(CONSTANT); }
L?'(\\.|[^\\'])+'      { count(); return(CONSTANT); }

{D}+{E}{FS}?           { count(); return(CONSTANT); }
{D}*"."{D}+({E})?{FS}? { count(); return(CONSTANT); }
{D}+"."{D}*({E})?{FS}? { count(); return(CONSTANT); }

L?\"(\\.|[^\\"])*\"    { count(); return(STRING_LITERAL); }

">>"                   { count(); return(RIGHT_OP); }
"<<"                   { count(); return(LEFT_OP); }
"->"                   { count(); return(PTR_OP); }
"&&"                   { count(); return(AND_OP); }
"||"                   { count(); return(OR_OP); }
"<="                   { count(); return(LE_OP); }
">="                   { count(); return(GE_OP); }
"=="                   { count(); return(EQ_OP); }
"!="                   { count(); return(NE_OP); }
";"                    { count(); return(';'); }
("{"|"<%")             { count(); return('{'); }
("}"|"%>")             { count(); return('}'); }
","                    { count(); return(','); }
":"                    { count(); return(':'); }
"="                    { count(); return('='); }
"("                    { count(); return('('); }
")"                    { count(); return(')'); }
("["|"<:")             { count(); return('['); }
("]"|":>")             { count(); return(']'); }
"."                    { count(); return('.'); }
"&"                    { count(); return('&'); }
"!"                    { count(); return('!'); }
"~"                    { count(); return('~'); }
"-"                    { count(); return('-'); }
"+"                    { count(); return('+'); }
"*"                    { count(); return('*'); }
"/"                    { count(); return('/'); }
"%"                    { count(); return('%'); }
"<"                    { count(); return('<'); }
">"                    { count(); return('>'); }
"^"                    { count(); return('^'); }
"|"                    { count(); return('|'); }
"?"                    { count(); return('?'); }

[ \t\v\n\f]            { count(); }
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
    while ((c = YYINPUT()) != '\n' && c != 0)
      // putchar(c)
      ;
    break;
  }
}

void comment() {
  char c, c1;

  while (true) {
    while ((c = YYINPUT()) != '*' && c != 0)
      // putchar(c)
      ;

    if ((c1 = YYINPUT()) != '/' && c != 0) {
      // unput(c1);
      continue;
    }

    break;
  }

  // if (c != 0)
  //   putchar(c1);
}

int g_column = 0;
int g_line = 0;
std::string g_lastLine = "";

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