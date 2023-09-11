//===- MuTypes.cpp - Mu dialect types ---------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Mu/MuTypes.h"

#include "Mu/MuDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir::mu;

#define GET_TYPEDEF_CLASSES
#include "Mu/MuOpsTypes.cpp.inc"

void MuDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "Mu/MuOpsTypes.cpp.inc"
      >();
}
