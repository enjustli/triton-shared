//===----------------------------------------------------------------------===//
//
// Copyright (c) Meta Platforms, Inc. and affiliates, Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_TENSOR_DESCRIPTOR_TO_STRUCTURED_CONVERSION_PASSES
#define TRITON_TENSOR_DESCRIPTOR_TO_STRUCTURED_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def TritonTensorDescriptorToStructured : Pass<"triton-tensor-descriptor-to-structured", "mlir::ModuleOp"> {
  let summary = "Convert Triton tensor descriptor ops to Triton structured IR";
  let constructor = "triton::createTritonTensorDescriptorToStructuredPass()";
}

#endif
