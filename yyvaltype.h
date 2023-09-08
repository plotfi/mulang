#ifndef YYVALTYPE
#define YYVALTYPE

typedef struct yyvalType {
  int linenum;
  char *value;
  char *tokText;
} yyvalType;

yyvalType *makeyyvalType(int linenum, char *value, const char *token);

#endif
