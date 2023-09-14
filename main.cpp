//===- main.cpp - The Mu Compiler -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the entry point for the Mu compiler.
//
//===----------------------------------------------------------------------===//

#include "Mu/MuDialect.h"
#include <algorithm>
#include <cstdlib>
#include <memory>
#include <fstream>
#include <iostream>
#include <optional>
#include <valarray>
#include <vector>

#include "Mu/Parser/ast.h"

#include "Mu/Support/µt8.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir::mu;
namespace cl = llvm::cl;

std::optional<Ref<mu::ast::ASTNodeTracker>> mu::ast::ASTNodeTracker::instance;
const unsigned mu::ast::ASTNode::static_magic_number = 0xdeadbeef;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input mu file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

//===- Bison Parse Handling Code ------------------------------------------===//
int yyparse();
extern FILE *yyin;

#if YYDEBUG
extern int yydebug;
#endif

extern mu::ast::TranslationUnit *topnode;

//===----------------------------------------------------------------------===//
/// Returns a Mu AST resulting from parsing the file or a nullptr on error.

namespace {

fv bisonReset() {
  // Bison is gross. Reset everything Bison related here.
  #if YYDEBUG
  yydebug = 0;
  #endif
  yyin = nullptr;
  topnode = nullptr;
}

fn parseInputFile(llvm::StringRef filename)
    -> std::unique_ptr<mu::ast::TranslationUnit> {
  bisonReset();

  // Sure wish this was C23
  Defer<decltype(yyin)> D {
    yyin = fopen(filename.data(), "r"),
    [](auto f) {
      fclose(f);
      bisonReset();
    }
  };

  // Bison is gross, especially GNU Bison 2.3 on macOS where global yyin is the
  // input to yyparse()
  assert(yyin != nullptr && topnode == nullptr && "Test pre-parse pointers.");
  yyparse();

  assert(topnode != nullptr && "Expected non-null topnode");
  return std::unique_ptr<mu::ast::TranslationUnit>(topnode);
}

enum InputType { Mu, MLIR };
enum Action { None, DumpAST, DumpMLIR };

cl::opt<enum InputType> inputType(
    "x", cl::init(Mu), cl::desc("Decided the kind of output desired"),
    cl::values(clEnumValN(Mu, "mu", "load the input file as a Mu source.")),
    cl::values(clEnumValN(MLIR, "mlir",
                          "load the input file as an MLIR file")));

cl::opt<enum Action> emitAction(
    "emit", cl::desc("Select the kind of output desired"),
    cl::values(clEnumValN(DumpAST, "ast", "output the AST dump")),
    cl::values(clEnumValN(DumpMLIR, "mlir", "output the MLIR dump")));

fn dumpMLIR() -> int {
  mlir::MLIRContext context;
  // Load our Dialect in this MLIR Context.
  context.getOrLoadDialect<mlir::mu::MuDialect>();

  // Handle '.mu' input to the compiler.
  if (inputType != InputType::MLIR &&
      !llvm::StringRef(inputFilename).endswith(".mlir")) {
    auto moduleAST = parseInputFile(inputFilename);
    if (!moduleAST)
      return 6;
    mlir::OwningOpRef<mlir::ModuleOp> module = nullptr;
    // TODO: Get mlirGen building
    #if 0
    mlir::OwningOpRef<mlir::ModuleOp> module = mlirGen(context, *moduleAST);
    #endif
    if (!module)
      return 1;

    module->dump();
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

fn dumpAST() -> int {
  if (inputType == InputType::MLIR) {
    llvm::errs() << "Can't dump a Mu AST when the input is MLIR\n";
    return 5;
  }

  auto moduleAST = parseInputFile(inputFilename);
  if (!moduleAST)
    return 1;

  // dump(*moduleAST);
  moduleAST->dump();
  llvm::errs() << "Tracked Node Count: "
               << mu::ast::ASTNodeTracker::get().size()
               << "\n";

  return 0;
}
} // namespace

fn main(int argc, char **argv)->int {

  // Register any command line options.
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  cl::ParseCommandLineOptions(argc, argv, "mu compiler\n");

  switch (emitAction) {
  case Action::DumpAST: {
    if (dumpAST())
      return -1;
    break;
  }
  case Action::DumpMLIR: {
    if (dumpMLIR())
      return -1;
    break;
  }
  default:
    llvm::errs() << "No action specified (parsing only?), use -emit=<action>\n";
    return -1;
  }

  mu::ast::ASTNodeTracker::destroy();
}
