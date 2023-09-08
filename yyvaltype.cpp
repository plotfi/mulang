#include "yyvaltype.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

yyvalType *makeyyvalType(int linenum, char *value, const char *token) {
  yyvalType *ret = (yyvalType *)malloc(sizeof(yyvalType));
  ret->linenum = linenum;
  ret->value = strdup(value);
  ret->tokText = strdup(token);
  return ret;
}