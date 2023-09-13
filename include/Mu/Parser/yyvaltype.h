#ifndef YYVALTYPE
#define YYVALTYPE

#include "Mu/Parser/ast.h"

namespace muast {

struct YYValType {
  YYValType(unsigned linenum, std::string value, std::string tokText)
      : linenum(linenum), value(value), tokText(tokText) {}
  unsigned linenum;
  std::string value;
  std::string tokText;
};

Ref<YYValType> makeYYValType(unsigned linenum, Ref<char> value, Ref<char> token);
fv clearYYValStorage();

} // namespace muast

#endif
