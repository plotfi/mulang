//===- MuTypes.td - Mu dialect types -----------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MU_TYPES
#define MU_TYPES

include "mlir/IR/AttrTypeBase.td"
include "Mu/MuDialect.td"

//===----------------------------------------------------------------------===//
// Mu type definitions
//===----------------------------------------------------------------------===//

class Mu_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Mu_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def Mu_CustomType : Mu_Type<"Custom", "custom"> {
    let summary = "Mu custom type";
    let description = "Custom type in Mu dialect";
    let parameters = (ins StringRefParameter<"the custom value">:$value);
    let assemblyFormat = "`<` $value `>`";
}

#endif // MU_TYPES
