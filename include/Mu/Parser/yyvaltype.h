#ifndef YYVALTYPE
#define YYVALTYPE

#include <string>
#include "Mu/Support/µt8.h"

namespace mu {
namespace ast {

struct YYValType {
  YYValType(unsigned linenum, std::string value, std::string tokText)
      : linenum(linenum), value(value), tokText(tokText) {}
  unsigned linenum;
  std::string value;
  std::string tokText;
};

Ref<YYValType> makeYYValType(unsigned linenum, Ref<char> value, Ref<char> token);
fv clearYYValStorage();

} // namespace ast
} // namespace mu

#endif
