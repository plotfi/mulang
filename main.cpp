//===- main.cpp - The Mu Compiler -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the entry point for the Toy compiler.
//
//===----------------------------------------------------------------------===//

#include "Mu/MuDialect.h"
#include <memory>
#include <fstream>
#include <iostream>
#include <optional>
#include <valarray>
#include <vector>

#include "Mu/Parser/ast.h"

#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir::mu;
namespace cl = llvm::cl;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input toy file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

//===- Bison Parse Handling Code ------------------------------------------===//
int yyparse();
extern FILE *yyin;
#if YYDEBUG
extern int yydebug;
#endif

void parse(Ref<char> filename) {
#if YYDEBUG
  yydebug = 0;
#endif
  yyin = fopen(filename, "r");
  yyparse();
  fclose(yyin);
}

extern muast::TranslationUnit *topnode;
std::optional<Ref<muast::ASTNodeTracker>> muast::ASTNodeTracker::instance;
const unsigned muast::ASTNode::static_magic_number = 0xdeadbeef;

//===----------------------------------------------------------------------===//

namespace {
enum InputType { Mu , MLIR };
} // namespace
static cl::opt<enum InputType> inputType(
    "x", cl::init(Mu), cl::desc("Decided the kind of output desired"),
    cl::values(clEnumValN(Mu, "toy", "load the input file as a Toy source.")),
    cl::values(clEnumValN(MLIR, "mlir",
                          "load the input file as an MLIR file")));

namespace {
enum Action { None, DumpAST, DumpMLIR };
} // namespace
static cl::opt<enum Action> emitAction(
    "emit", cl::desc("Select the kind of output desired"),
    cl::values(clEnumValN(DumpAST, "ast", "output the AST dump")),
    cl::values(clEnumValN(DumpMLIR, "mlir", "output the MLIR dump")));


/// Returns a Toy AST resulting from parsing the file or a nullptr on error.
int* parseInputFile(llvm::StringRef filename) {
  return nullptr;
}

int dumpMLIR() {
  mlir::MLIRContext context;
  // Load our Dialect in this MLIR Context.
  context.getOrLoadDialect<mlir::mu::MuDialect>();

  // Handle '.mu' input to the compiler.
  if (inputType != InputType::MLIR &&
      !llvm::StringRef(inputFilename).endswith(".mlir")) {
    auto moduleAST = parseInputFile(inputFilename);
    if (!moduleAST)
      return 6;
    mlir::OwningOpRef<mlir::ModuleOp> module = nullptr; //mlirGen(context, *moduleAST);
    if (!module)
      return 1;

    // module->dump();
    return 0;
  }

  // Otherwise, the input is '.mlir'.
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return -1;
  }

  // Parse the input mlir.
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Error can't load file " << inputFilename << "\n";
    return 3;
  }

  module->dump();
  return 0;
}

int dumpAST() {
  if (inputType == InputType::MLIR) {
    llvm::errs() << "Can't dump a Toy AST when the input is MLIR\n";
    return 5;
  }

  auto moduleAST = parseInputFile(inputFilename);
  if (!moduleAST)
    return 1;

  // dump(*moduleAST);
  return 0;
}

auto main(int argc, char **argv) -> int {

  // Register any command line options.
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  cl::ParseCommandLineOptions(argc, argv, "toy compiler\n");

  switch (emitAction) {
  case Action::DumpAST:
    return dumpAST();
  case Action::DumpMLIR:
    return dumpMLIR();
  default:
    llvm::errs() << "No action specified (parsing only?), use -emit=<action>\n";
  }

  if (argc < 2) {
    std::cerr << "Expected: " << argv[0] << " filename.c\n";
    return -1;
  }

  parse(argv[1]);

  std::cout << "TOPNODE NAME: " << "main"  << "\n";
  topnode->dump();
  std::cout << "Tracked Node Count: " << muast::ASTNodeTracker::get().size() << "\n";

  delete topnode;
  muast::ASTNodeTracker::destroy();
}
