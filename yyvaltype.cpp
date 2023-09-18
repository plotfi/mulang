#include <memory>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>

#include "Mu/Parser/yyvaltype.h"
#include "Mu/Support/µt8.h"

#include "llvm/Support/raw_ostream.h"

static std::vector<std::unique_ptr<mu::ast::YYValType>> YYValStorage;

fn mu::ast::makeYYValType(unsigned linenum, Ref<char> value, Ref<char> token)
    ->Ref<mu::ast::YYValType> {

  #ifndef NDEBUG
  llvm::errs() << "makeYYValType-ing: " << value << " " << token << " "
               << YYValStorage.size() << "\n";
  #endif

  auto yyval = std::make_unique<YYValType>(linenum, std::string(value),
                                           std::string(token));
  YYValStorage.push_back(std::move(yyval));
  return YYValStorage.back().get();
}

fv mu::ast::clearYYValStorage() {
#ifndef NDEBUG
  llvm::errs() << "Total YYVals Used: " << YYValStorage.size() << "\n";
#endif
  YYValStorage.clear();
}
