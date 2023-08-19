#include <vector>
#include <string>
#include <iostream>

struct Defun {
  std::string name;

  Defun() = default;

  Defun(const char *name): name(name) {}

  void dump() const {
    std::cout << "\n(defun " << name << ")";
  }
};

struct DefunList {
  std::vector<Defun*> funcs;

  DefunList() = delete;
  DefunList(Defun *func) {
    funcs.push_back(func);
  }

  DefunList *append(Defun *func) {
    funcs.push_back(func);
    return this;
  };

  void dump() const {
    for (auto func : funcs)
      func->dump();
  }
};

struct TranslationUnit {
  std::string name = "main";
  DefunList *funcs = nullptr;

  TranslationUnit() = delete;
  TranslationUnit(DefunList *funcs): funcs(funcs) {}

  void dump() const {
    funcs->dump();
  };
};

