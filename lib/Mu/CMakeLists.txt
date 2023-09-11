add_mlir_dialect_library(MLIRMu
        MuTypes.cpp
        MuDialect.cpp
        MuOps.cpp
        MuPasses.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Mu

        DEPENDS
        MLIRMuOpsIncGen
        MLIRMuPassesIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRFuncDialect
        )
