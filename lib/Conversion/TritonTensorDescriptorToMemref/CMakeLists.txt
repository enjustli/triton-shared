add_triton_library(TritonTensorDescriptorToMemref
  TritonTensorDescriptorToMemrefPass.cpp

  DEPENDS
  TritonTensorDescriptorToMemrefConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRDialectUtils
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRPtrDialect
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRTransforms
  MLIRSupport
  TPtrIR
  TritonIR
  TritonStructuredIR
  TritonTransforms
  TritonSharedUtils
)
