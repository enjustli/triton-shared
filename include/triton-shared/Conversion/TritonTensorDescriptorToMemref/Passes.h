//===----------------------------------------------------------------------===//
//
// Copyright (c) Meta Platforms, Inc. and affiliates, Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_TENSOR_DESCRIPTOR_TO_MEMREF_CONVERSION_PASSES_H
#define TRITON_TENSOR_DESCRIPTOR_TO_MEMREF_CONVERSION_PASSES_H

#include "triton-shared/Conversion/TritonTensorDescriptorToMemref/TritonTensorDescriptorToMemref.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "triton-shared/Conversion/TritonTensorDescriptorToMemref/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
