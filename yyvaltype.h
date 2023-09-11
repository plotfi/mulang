#ifndef YYVALTYPE
#define YYVALTYPE

#include "ast.h"

struct YYValType {
  YYValType(unsigned linenum, std::string value, std::string tokText)
      : linenum(linenum), value(value), tokText(tokText) {}
  unsigned linenum;
  std::string value;
  std::string tokText;
};

Ref<YYValType> makeYYValType(unsigned linenum, Ref<char> value, Ref<char> token);
fv clearYYValStorage();

#endif
