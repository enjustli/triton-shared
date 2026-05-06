//===----------------------------------------------------------------------===//
//
// Copyright (c) Meta Platforms, Inc. and affiliates, Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_TENSOR_DESCRIPTOR_TO_MEMREF_CONVERSION_PASSES
#define TRITON_TENSOR_DESCRIPTOR_TO_MEMREF_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def TritonTensorDescriptorToMemref : Pass<"triton-tensor-descriptor-to-memref", "mlir::ModuleOp"> {
  let summary = "Convert Triton tensor descriptor ops to memref-based IR";
  let constructor = "triton::createTritonTensorDescriptorToMemrefPass()";
}

#endif
