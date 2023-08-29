#include <fstream>
#include <iostream>
#include <vector>
#include <ast.h>

int yyparse();
extern FILE *yyin;
extern TranslationUnit *topnode;
ASTNodeTracker* ASTNodeTracker::instance = nullptr;;
unsigned astNodeCreateCount;
unsigned astNodeDestroyCount;


void dumpASTNode(const ASTNode *node) {
  node->dump();
}


#if YYDEBUG
extern int yydebug;
#endif

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Expected: " << argv[0] << " filename.c\n";
    return -1;
  }

#if YYDEBUG
  yydebug = 0;
#endif

  yyin = fopen(argv[1], "r");
  yyparse();
  fclose(yyin);

  std::cout << "TOPNODE NAME: " << "main" << "\n";
  topnode->dump();
  std::cout << "\n\n";

  std::cout << "Tracked AST Node Count: " << ASTNodeTracker::get()->size() << "\n";

  std::cout << "Iterate over all tracked AST Nodes:\n\n";

  std::cout << "Dump Tracked:\n";
  ASTNodeTracker::get()->dumpTracked();


  // ASTNodeTracker::get()->dump();
  delete topnode;

  std::cout << "\n\nDumping Untracked Nodes:\n\n";
  ASTNodeTracker::get()->dumpUntracked();

}
