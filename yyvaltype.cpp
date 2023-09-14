#include <memory>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>

#include "Mu/Parser/yyvaltype.h"
#include "Mu/Support/µt8.h"

#include "llvm/Support/raw_ostream.h"

static std::vector<const mu::ast::YYValType> YYValStorage;

fn mu::ast::makeYYValType(unsigned linenum, Ref<char> value, Ref<char> token)
    ->Ref<mu::ast::YYValType> {
  YYValStorage.emplace_back(linenum, std::string(value), std::string(token));
  return &YYValStorage.back();
}

fv mu::ast::clearYYValStorage() {
  llvm::errs() << "Total YYVals Used: " << YYValStorage.size() << "\n";
  YYValStorage.clear();
}
