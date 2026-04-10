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

#include "Mu/MuMLIRGen.h"
#include "Mu/MuPasses.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include "Mu/Parser/ast.h"
#include "Mu/Support/µt8.h"

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

  if (nullptr == yyin) {
    llvm::errs() << "File not found: " << filename << "\n";
    exit(EXIT_FAILURE);
  }

  // Bison is gross, especially GNU Bison 2.3 on macOS where global yyin is the
  // input to yyparse()
  assert(yyin != nullptr && topnode == nullptr && "Test pre-parse pointers.");
  yyparse();

  assert(topnode != nullptr && "Expected non-null topnode");
  return std::unique_ptr<mu::ast::TranslationUnit>(topnode);
}

enum InputType { Mu, MLIR };
enum Action { None, DumpAST, DumpMLIR, DumpLLVM };

cl::opt<enum InputType> inputType(
    "x", cl::init(Mu), cl::desc("Decided the kind of output desired"),
    cl::values(clEnumValN(Mu, "mu", "load the input file as a Mu source.")),
    cl::values(clEnumValN(MLIR, "mlir",
                          "load the input file as an MLIR file")));

cl::opt<enum Action> emitAction(
    "emit", cl::desc("Select the kind of output desired"),
    cl::values(clEnumValN(DumpAST, "ast", "output the AST dump")),
    cl::values(clEnumValN(DumpMLIR, "mlir", "output the MLIR dump")),
    cl::values(clEnumValN(DumpLLVM, "llvm", "output LLVM IR")));

/// Load a module from a .mu source file or .mlir file.
fn loadModule(mlir::MLIRContext &context)
    -> mlir::OwningOpRef<mlir::ModuleOp> {
  // Handle '.mu' input to the compiler.
  if (inputType != InputType::MLIR &&
      !llvm::StringRef(inputFilename).ends_with(".mlir")) {

    if (!llvm::StringRef(inputFilename).ends_with(".mu") &&
        !llvm::StringRef(inputFilename).ends_with(".mulang") &&
        !llvm::StringRef(inputFilename).ends_with(".\342\232\233")) {
      llvm::errs() << "Invalid filetype: " << inputFilename << "\n";
      llvm::errs() << "Files given to muc must end in .mu or .⚛️\n";
      exit(EXIT_FAILURE);
    }

    auto moduleAST = parseInputFile(inputFilename);
    if (!moduleAST)
      return nullptr;
    return mu::mlirGen(context, *moduleAST);
  }

  // Otherwise, the input is '.mlir'.
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return nullptr;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  return mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
}

fn dumpMLIR() -> int {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::mu::MuDialect>();
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::memref::MemRefDialect>();
  context.getOrLoadDialect<mlir::cf::ControlFlowDialect>();

  auto module = loadModule(context);
  if (!module)
    return 1;

  module->dump();
  return 0;
}

fn dumpLLVM() -> int {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::mu::MuDialect>();
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::cf::ControlFlowDialect>();
  context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
  context.getOrLoadDialect<mlir::memref::MemRefDialect>();

  auto module = loadModule(context);
  if (!module)
    return 1;

  // Build the lowering pipeline:
  // 1. Mu -> arith/func/cf/memref
  // 2. arith/func/cf/memref -> LLVM dialect
  // 3. Reconcile unrealized casts
  mlir::PassManager pm(&context);
  pm.addPass(mlir::mu::createMuLoweringPass());
  pm.addPass(mlir::createArithToLLVMConversionPass());
  pm.addPass(mlir::createConvertControlFlowToLLVMPass());
  pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(mlir::createConvertFuncToLLVMPass());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());

  if (failed(pm.run(*module))) {
    llvm::errs() << "Failed to lower Mu to LLVM dialect\n";
    return 1;
  }

  // Translate MLIR LLVM dialect to LLVM IR.
  mlir::registerBuiltinDialectTranslation(context);
  mlir::registerLLVMDialectTranslation(context);
  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(*module, llvmContext);
  if (!llvmModule) {
    llvm::errs() << "Failed to translate to LLVM IR\n";
    return 1;
  }

  llvmModule->print(llvm::outs(), nullptr);
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

#ifndef NDEBUG
  llvm::errs() << "Tracked Node Count: "
               << mu::ast::ASTNodeTracker::get().size() << "\n";
#endif

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
  case Action::DumpLLVM: {
    if (dumpLLVM())
      return -1;
    break;
  }
  default:
    llvm::errs() << "No action specified (parsing only?), use -emit=<action>\n";
    return -1;
  }

  mu::ast::ASTNodeTracker::destroy();
}
