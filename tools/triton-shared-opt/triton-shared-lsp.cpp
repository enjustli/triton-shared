#include "./RegisterTritonSharedDialects.h"

#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registerTritonSharedDialects(registry);

  return mlir::failed(mlir::MlirLspServerMain(argc, argv, registry));
}
