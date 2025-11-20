#ifndef CONVERSION_LINALGTOVECTOR_LINALGTOVECTOR_H
#define CONVERSION_LINALGTOVECTOR_LINALGTOVECTOR_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace triton {

#define GEN_PASS_DECL
#include "triton-shared/Conversion/LinalgToVector/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createLinalgToVectorPass();

} // namespace triton
} // namespace mlir

#endif // CONVERSION_LINALGTOVECTOR_LINALGTOVECTOR_H
