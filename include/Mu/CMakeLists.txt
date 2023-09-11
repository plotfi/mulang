add_mlir_dialect(MuOps mu)
add_mlir_doc(MuDialect MuDialect Mu/ -gen-dialect-doc)
add_mlir_doc(MuOps MuOps Mu/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS MuPasses.td)
mlir_tablegen(MuPasses.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIRMuPassesIncGen)
