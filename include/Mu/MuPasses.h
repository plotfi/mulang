//===- MuPasses.h - Mu passes  ----------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MU_MUPASSES_H
#define MU_MUPASSES_H

#include "Mu/MuDialect.h"
#include "Mu/MuOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace mu {
#define GEN_PASS_DECL
#include "Mu/MuPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "Mu/MuPasses.h.inc"
} // namespace mu
} // namespace mlir

#endif
