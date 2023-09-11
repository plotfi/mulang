#include <memory>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>

#include "Mu/Parser/yyvaltype.h"
#include "Mu/Support/µt8.h"

static std::vector<const YYValType> YYValStorage;

fn makeYYValType(unsigned linenum, Ref<char> value, Ref<char> token)
    ->Ref<YYValType> {
  YYValStorage.emplace_back(linenum, std::string(value), std::string(token));
  return &YYValStorage.back();
}

fv clearYYValStorage() {
  std::cout << "Total YYVals Used: " << YYValStorage.size() << "\n";
  YYValStorage.clear();
}
