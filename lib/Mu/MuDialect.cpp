//===- MuDialect.cpp - Mu dialect -------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Mu/MuDialect.h"
#include "Mu/MuOps.h"
#include "Mu/MuTypes.h"

using namespace mlir;
using namespace mlir::mu;

#include "Mu/MuOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Mu dialect.
//===----------------------------------------------------------------------===//

void MuDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Mu/MuOps.cpp.inc"
      >();
  registerTypes();
}
