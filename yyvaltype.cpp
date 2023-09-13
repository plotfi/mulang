#include <memory>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>

#include "Mu/Parser/yyvaltype.h"
#include "Mu/Support/µt8.h"

#include "llvm/Support/raw_ostream.h"

static std::vector<const muast::YYValType> YYValStorage;

fn muast::makeYYValType(unsigned linenum, Ref<char> value, Ref<char> token)
    ->Ref<muast::YYValType> {
  YYValStorage.emplace_back(linenum, std::string(value), std::string(token));
  return &YYValStorage.back();
}

fv muast::clearYYValStorage() {
  llvm::errs() << "Total YYVals Used: " << YYValStorage.size() << "\n";
  YYValStorage.clear();
}
