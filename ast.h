#include <vector>
#include <string>
#include <iostream>

using Type = std::string;

struct TypedDecl {
  std::string name;
  Type type;
};

struct Decl {
  virtual std::string str() const = 0;
};

struct LamdaDecl : public Decl {
  std::vector<TypedDecl> inputArgs;
  Type returnType;

  virtual std::string str() const {
    return "lamda";
  }
};

struct Definition {
  std::string name;
  Decl *content;

  Definition(std::string name, Decl *content):
    name(name), content(content) {}

  std::string str() const {
    return "(define " + name + " " +
           content->str() +
           ")";
  }
};

struct TranslationUnitDecl {
  std::string name;
  std::vector<Definition*> children;

  void dump() const {
    for (auto *child : children)
      std::cout << "\n" << child->str();
  };
};



inline std::vector<Definition*> *AppendDecl(void *decls, const char *name) {
  std::vector<Definition*> *Decls = (std::vector<Definition*>*)decls;
  if (!Decls)
    Decls = new std::vector<Definition*>();
  Decls->push_back(new Definition(std::string(name), nullptr));
  return Decls;
}
