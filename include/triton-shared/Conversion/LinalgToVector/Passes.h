#ifndef LINALG_TO_VECTOR_CONVERSION_PASSES_H
#define LINALG_TO_VECTOR_CONVERSION_PASSES_H

#include "triton-shared/Conversion/LinalgToVector/LinalgToVector.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "triton-shared/Conversion/LinalgToVector/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
