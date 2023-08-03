#include <vector>
#include <string>
#include <iostream>

struct NamedDecl {
  std::string name;
};

struct TranslationUnitDecl {
  std::string name;
  std::vector<NamedDecl*> children;

  void dump() const {
    std::cout << "TU Name: " << name << "\n";
    for (auto *ND : children) {
      std::cout << "\tNamedDecl: " << ND->name << "\n";
    }
  };
};

