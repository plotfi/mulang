#include <fstream>
#include <iostream>
#include <vector>
#include <ast.h>

int yyparse();
extern FILE *yyin;
extern TranslationUnit *topnode;


#if YYDEBUG
extern int yydebug;
#endif

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Expected: " << argv[0] << " filename.c\n";
    return -1;
  }

#if YYDEBUG
  yydebug = 1;
#endif

  yyin = fopen(argv[1], "r");
  yyparse();
  fclose(yyin);

  std::cout << "TOPNODE NAME: " << "main" << "\n";
  topnode->dump();
  std::cout << "\n\n";
}
