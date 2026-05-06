//===----------------------------------------------------------------------===//
//
// Copyright (c) Meta Platforms, Inc. and affiliates, Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_CONVERSION_TRITON_TENSOR_DESCRIPTOR_TO_MEMREF_H
#define TRITON_CONVERSION_TRITON_TENSOR_DESCRIPTOR_TO_MEMREF_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>>
createTritonTensorDescriptorToMemrefPass();

} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_TRITON_TENSOR_DESCRIPTOR_TO_MEMREF_H
