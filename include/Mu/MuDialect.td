//===- MuDialect.td - Mu dialect ---------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MU_DIALECT
#define MU_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Mu dialect definition.
//===----------------------------------------------------------------------===//

def Mu_Dialect : Dialect {
    let name = "mu";
    let summary = "An out-of-tree MLIR dialect for Puyan's Micro Language.";
    let description = [{
        This dialect is an example of an out-of-tree MLIR dialect designed to
        illustrate the basic setup required to develop MLIR-based tools without
        working inside of the LLVM source tree.
    }];
    let cppNamespace = "::mlir::mu";

    let useDefaultTypePrinterParser = 1;
    let extraClassDeclaration = [{
        void registerTypes();
    }];
}

//===----------------------------------------------------------------------===//
// Base mu operation definition.
//===----------------------------------------------------------------------===//

class Mu_Op<string mnemonic, list<Trait> traits = []> :
        Op<Mu_Dialect, mnemonic, traits>;

#endif // MU_DIALECT
