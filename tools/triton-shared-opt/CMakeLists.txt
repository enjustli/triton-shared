get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(triton_libs GLOBAL PROPERTY TRITON_LIBS)

add_llvm_executable(triton-shared-opt triton-shared-opt.cpp PARTIAL_SOURCES_INTENDED)

# TODO: what's this?
llvm_update_compile_flags(triton-shared-opt)
target_link_libraries(triton-shared-opt PRIVATE
  TritonTransforms
  TritonSharedAnalysis
  ${dialect_libs}
  ${conversion_libs}
  ${triton_libs}
  # MLIR core
  MLIROptLib
  MLIRPass
  MLIRRegisterAllPasses
  MLIRTransforms
)

mlir_check_all_link_libraries(triton-shared-opt)

add_llvm_executable(triton-shared-lsp triton-shared-lsp.cpp PARTIAL_SOURCES_INTENDED)

llvm_update_compile_flags(triton-shared-lsp)
target_link_libraries(triton-shared-lsp PRIVATE
  TritonTransforms
  TritonSharedAnalysis
  ${dialect_libs}
  ${conversion_libs}
  ${triton_libs}
  # tests
  TritonTestAnalysis
  TritonTestDialect
  TritonAMDGPUTestAnalysis
  TritonTestProton
  # MLIR core
  MLIRLspServerLib
  MLIRPass
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
  MLIRTransforms
)

mlir_check_all_link_libraries(triton-shared-lsp)