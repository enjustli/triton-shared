//===----------------------------------------------------------------------===//
//
// Copyright (c) Meta Platforms, Inc. and affiliates, Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//
// This pass lowers Triton tensor descriptor ops to Triton structured IR.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Ptr/IR/PtrDialect.h"
#include "mlir/Dialect/Ptr/IR/PtrOps.h"
#include "mlir/Dialect/Ptr/IR/PtrTypes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

#include "triton-shared/Analysis/OpFoldResultUtils.h"
#include "triton-shared/AnalysisStructured/PtrAnalysis.h"
#include "triton-shared/Conversion/TritonTensorDescriptorToStructured/TritonTensorDescriptorToStructured.h"
#include "triton-shared/Dialect/TPtr/IR/TPtrDialect.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton-shared/Utils/Utils.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/Transforms/ArithTypeConversion.h"
#include "triton/Dialect/Triton/Transforms/FunctionTypeConversion.h"

#include "llvm/ADT/STLExtras.h"

#define DEBUG_TYPE "triton-tensor-descriptor-to-structured"

using namespace mlir;

namespace {

#define GEN_PASS_DEF_TRITONTENSORDESCRIPTORTOSTRUCTURED
#include "triton-shared/Conversion/TritonTensorDescriptorToStructured/Passes.h.inc"

static Value castToIndex(OpBuilder &builder, Value value) {
  if (value.getType().isIndex())
    return value;
  return arith::IndexCastOp::create(builder, value.getLoc(),
                                    builder.getIndexType(), value);
}

static SmallVector<Value> castToIndex(OpBuilder &builder, ValueRange values) {
  return llvm::map_to_vector(
      values, [&](Value value) { return castToIndex(builder, value); });
}

static Value castToI64(OpBuilder &builder, Value value) {
  auto i64Type = builder.getI64Type();
  if (value.getType() == i64Type)
    return value;
  auto loc = value.getLoc();
  if (value.getType().isIndex())
    return arith::IndexCastOp::create(builder, loc, i64Type, value);
  auto intType = cast<IntegerType>(value.getType());
  if (intType.getWidth() < 64)
    return arith::ExtSIOp::create(builder, loc, i64Type, value);
  return arith::TruncIOp::create(builder, loc, i64Type, value);
}

static SmallVector<Value> castToI64(OpBuilder &builder, ValueRange values) {
  return llvm::map_to_vector(
      values, [&](Value value) { return castToI64(builder, value); });
}

static bool canRankReduceShape(ArrayRef<int64_t> sourceShape,
                               ArrayRef<int64_t> resultShape) {
  unsigned resultDim = 0;
  for (int64_t sourceDim : sourceShape) {
    if (resultDim < resultShape.size() && sourceDim == resultShape[resultDim]) {
      ++resultDim;
      continue;
    }
    if (sourceDim != 1)
      return false;
  }
  return resultDim == resultShape.size();
}

static Value getRankReducingExtractSlice(Value source,
                                         RankedTensorType resultType,
                                         ArrayRef<int64_t> sourceShape,
                                         Location loc, OpBuilder &builder) {
  auto sourceType = cast<RankedTensorType>(source.getType());
  if (sourceType == resultType)
    return source;

  SmallVector<OpFoldResult> offsets(sourceType.getRank(),
                                    builder.getIndexAttr(0));
  SmallVector<OpFoldResult> sizes;
  for (int64_t size : sourceShape)
    sizes.push_back(builder.getIndexAttr(size));
  SmallVector<OpFoldResult> strides(sourceType.getRank(),
                                    builder.getIndexAttr(1));
  return tensor::ExtractSliceOp::create(builder, loc, resultType, source,
                                        offsets, sizes, strides);
}

static Value getRankExpandingValue(Value source, RankedTensorType resultType,
                                   ArrayRef<int64_t> resultShape, Location loc,
                                   OpBuilder &builder) {
  auto sourceType = cast<RankedTensorType>(source.getType());
  if (sourceType == resultType)
    return source;

  SmallVector<ReassociationIndices> reassociation(sourceType.getRank());
  unsigned sourceDim = 0;
  for (auto [resultDim, resultSize] : llvm::enumerate(resultShape)) {
    if (sourceDim >= reassociation.size()) {
      reassociation.back().push_back(resultDim);
      continue;
    }

    reassociation[sourceDim].push_back(resultDim);
    if (resultSize == sourceType.getDimSize(sourceDim))
      ++sourceDim;
  }

  return tensor::ExpandShapeOp::create(builder, loc, resultType, source,
                                       reassociation);
}

static Value getPadValue(OpBuilder &builder, Location loc, Type elementType,
                         Value paddingIsNan) {
  Value zero =
      arith::ConstantOp::create(builder, loc, builder.getZeroAttr(elementType));
  if (!isa<FloatType>(elementType))
    return zero;

  auto floatType = cast<FloatType>(elementType);
  auto nan = llvm::APFloat::getNaN(floatType.getFloatSemantics());
  Value nanValue = arith::ConstantFloatOp::create(builder, loc, floatType, nan);
  return arith::SelectOp::create(builder, loc, paddingIsNan, nanValue, zero);
}

static Type getI32TypeLike(OpBuilder &builder, Type type) {
  if (auto shapedType = dyn_cast<ShapedType>(type))
    return shapedType.clone(builder.getI32Type());
  return builder.getI32Type();
}

static Value getI32ConstLike(OpBuilder &builder, Location loc, Type likeType,
                             int32_t value) {
  Type i32Type = getI32TypeLike(builder, likeType);
  if (auto shapedType = dyn_cast<ShapedType>(i32Type)) {
    auto attr =
        DenseElementsAttr::get(shapedType, builder.getI32IntegerAttr(value));
    return arith::ConstantOp::create(builder, loc, shapedType, attr);
  }
  return arith::ConstantOp::create(builder, loc, i32Type,
                                   builder.getI32IntegerAttr(value));
}

static Value roundF32ToTF32(OpBuilder &builder, Location loc, Value value) {
  auto valueType = value.getType();
  if (auto tensorType = dyn_cast<RankedTensorType>(valueType)) {
    SmallVector<utils::IteratorType> iteratorTypes(
        tensorType.getRank(), utils::IteratorType::parallel);
    auto map = AffineMap::getMultiDimIdentityMap(tensorType.getRank(),
                                                 builder.getContext());
    SmallVector<AffineMap> indexingMaps(2, map);
    auto empty = tensor::EmptyOp::create(builder, loc, tensorType.getShape(),
                                         tensorType.getElementType());
    auto genericOp = linalg::GenericOp::create(
        builder, loc, TypeRange{tensorType}, ValueRange{value},
        ValueRange{empty}, indexingMaps, iteratorTypes,
        [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
          Type i32Type = b.getI32Type();
          Value bits = arith::BitcastOp::create(b, nestedLoc, i32Type, args[0]);
          Value expMask = getI32ConstLike(b, nestedLoc, i32Type, 0x7F800000);
          Value exp = arith::AndIOp::create(b, nestedLoc, bits, expMask);
          Value isSpecial = arith::CmpIOp::create(
              b, nestedLoc, arith::CmpIPredicate::eq, exp, expMask);

          Value shift = getI32ConstLike(b, nestedLoc, i32Type, 13);
          Value lsb = arith::AndIOp::create(
              b, nestedLoc, arith::ShRUIOp::create(b, nestedLoc, bits, shift),
              getI32ConstLike(b, nestedLoc, i32Type, 1));
          Value roundBias = arith::AddIOp::create(
              b, nestedLoc, lsb,
              getI32ConstLike(b, nestedLoc, i32Type, 0x00000FFF));
          Value rounded = arith::AndIOp::create(
              b, nestedLoc,
              arith::AddIOp::create(b, nestedLoc, bits, roundBias),
              getI32ConstLike(b, nestedLoc, i32Type, 0xFFFFE000));
          Value outBits =
              arith::SelectOp::create(b, nestedLoc, isSpecial, bits, rounded);
          Value out = arith::BitcastOp::create(b, nestedLoc, args[0].getType(),
                                               outBits);
          linalg::YieldOp::create(b, nestedLoc, out);
        });
    return genericOp.getResult(0);
  }

  auto i32Type = getI32TypeLike(builder, valueType);
  Value bits = arith::BitcastOp::create(builder, loc, i32Type, value);

  Value expMask = getI32ConstLike(builder, loc, i32Type, 0x7F800000);
  Value exp = arith::AndIOp::create(builder, loc, bits, expMask);
  Value isSpecial = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::eq, exp, expMask);

  Value shift = getI32ConstLike(builder, loc, i32Type, 13);
  Value lsb = arith::AndIOp::create(
      builder, loc, arith::ShRUIOp::create(builder, loc, bits, shift),
      getI32ConstLike(builder, loc, i32Type, 1));
  Value roundBias = arith::AddIOp::create(
      builder, loc, lsb, getI32ConstLike(builder, loc, i32Type, 0x00000FFF));
  Value rounded = arith::AndIOp::create(
      builder, loc, arith::AddIOp::create(builder, loc, bits, roundBias),
      getI32ConstLike(builder, loc, i32Type, 0xFFFFE000));
  Value outBits =
      arith::SelectOp::create(builder, loc, isSpecial, bits, rounded);
  return arith::BitcastOp::create(builder, loc, valueType, outBits);
}
struct Descriptor {
  Value base;
  ValueRange shape;
  ValueRange strides;
  Value paddingOption;
  Value roundF32ToTF32;
};

Descriptor unpackDescriptor(triton::TensorDescType type, ValueRange pack) {
  int rank = type.getShape().size();
  assert(pack.size() == 1 + 2 * static_cast<size_t>(rank) + 2 &&
         "Expected tensor descriptors to consist of a pointer, "
         "followed by 'rank' shape values and 'rank' stride values, "
         "followed by padding and TF32 rounding option values.");

  Descriptor res;
  res.base = pack[0];
  res.shape = pack.slice(1, rank);
  res.strides = pack.slice(1 + rank, rank);
  res.paddingOption = pack[1 + 2 * rank];
  res.roundF32ToTF32 = pack[2 + 2 * rank];
  return res;
}

static tts::MakeTensorPtrOp
createDescriptorTensorPtr(OpBuilder &builder, Location loc, Descriptor desc,
                          triton::TensorDescType descType) {
  SmallVector<Value> offsets(descType.getShape().size());
  llvm::transform(offsets, offsets.begin(), [&](Value) {
    return arith::ConstantIndexOp::create(builder, loc, 0);
  });

  SmallVector<OpFoldResult> mixedStrides;
  for (Value stride : castToIndex(builder, desc.strides))
    mixedStrides.push_back(stride);

  SmallVector<OpFoldResult> mixedOffsets;
  for (Value offset : castToIndex(builder, offsets))
    mixedOffsets.push_back(offset);

  SmallVector<OpFoldResult> mixedShape;
  for (Value dim : castToIndex(builder, desc.shape))
    mixedShape.push_back(dim);

  return tts::MakeTensorPtrOp::create(
      builder, loc, desc.base, descType.getShape(), mixedStrides, mixedOffsets,
      mixedShape, SmallVector<int32_t>{});
}

static tts::MakeTensorPtrOp
createDescriptorAccessTensorPtr(OpBuilder &builder, Location loc,
                                tts::MakeTensorPtrOp desc, ValueRange offsets) {
  SmallVector<OpFoldResult> mixedOffsets;
  for (auto [offset, stride] :
       llvm::zip(castToIndex(builder, offsets), desc.getMixedStrides()))
    mixedOffsets.push_back(mulOFRs(offset, stride, loc, builder));

  SmallVector<OpFoldResult> noWrapShape(desc.getSizes().size(),
                                        builder.getIndexAttr(0));
  return tts::MakeTensorPtrOp::create(
      builder, loc, desc.getBase(), desc.getSizes(), desc.getMixedStrides(),
      mixedOffsets, noWrapShape, desc.getOrder());
}

static tts::MakeTensorPtrOp
createDescriptorLinearAccessTensorPtr(OpBuilder &builder, Location loc,
                                      tts::MakeTensorPtrOp desc,
                                      ValueRange offsets) {
  SmallVector<OpFoldResult> mixedOffsets;
  for (auto [offset, stride] :
       llvm::zip(castToIndex(builder, offsets), desc.getMixedStrides()))
    mixedOffsets.push_back(mulOFRs(offset, stride, loc, builder));

  SmallVector<OpFoldResult> noWrapShape(desc.getSizes().size(),
                                        builder.getIndexAttr(0));
  return tts::MakeTensorPtrOp::create(
      builder, loc, desc.getBase(), desc.getSizes(), desc.getMixedStrides(),
      mixedOffsets, noWrapShape, desc.getOrder());
}

static Value createGatherScatterMask(OpBuilder &builder, Location loc,
                                     Value offsets, OpFoldResult dimSize) {
  auto offsetsType = cast<RankedTensorType>(offsets.getType());
  auto offsetElementType = cast<IntegerType>(offsetsType.getElementType());
  Value zero = arith::ConstantIntOp::create(builder, loc, 0,
                                            offsetElementType.getWidth());
  Value splatZero = triton::SplatOp::create(builder, loc, offsetsType, zero);
  Value nonNegative = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::sge, offsets, splatZero);

  Value dim = getValueOrCreateConstantIndexOp(builder, loc, dimSize);
  dim = arith::IndexCastOp::create(builder, loc, offsetElementType, dim);
  Value splatDim = triton::SplatOp::create(builder, loc, offsetsType, dim);
  Value inUpperBound = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::slt, offsets, splatDim);

  return arith::AndIOp::create(builder, loc, nonNegative, inUpperBound);
}

static SmallVector<OpFoldResult> getDescriptorGatherScatterMaskDims(
    OpBuilder &builder, Location loc, ArrayRef<int64_t> tensorShape,
    ArrayRef<OpFoldResult> descShape, Value yOffset) {
  assert(tensorShape.size() == 2 && descShape.size() == 2);
  SmallVector<OpFoldResult> maskDims;
  maskDims.push_back(builder.getIndexAttr(0));

  Value dim1 = getValueOrCreateConstantIndexOp(builder, loc, descShape[1]);
  Value y = castToIndex(builder, yOffset);
  Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
  Value expected = arith::ConstantIndexOp::create(builder, loc, tensorShape[1]);
  Value available = arith::SubIOp::create(builder, loc, dim1, y);
  Value nonNegativeAvailable =
      arith::MaxSIOp::create(builder, loc, available, zero);
  Value tooSmall = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::slt, nonNegativeAvailable, expected);
  Value clamped = arith::SelectOp::create(builder, loc, tooSmall,
                                          nonNegativeAvailable, expected);
  maskDims.push_back(clamped);
  return maskDims;
}

static tts::MakeGatherScatterTensorPtrOp createDescriptorGatherScatterTensorPtr(
    OpBuilder &builder, Location loc, tts::MakeTensorPtrOp desc, Value xOffsets,
    Value yOffset, Value gatherScatterMask, ArrayRef<int64_t> tensorShape) {
  SmallVector<OpFoldResult> offsets;
  offsets.push_back(builder.getIndexAttr(0));
  offsets.push_back(mulOFRs(castToIndex(builder, yOffset),
                            desc.getMixedStrides()[1], loc, builder));
  if (gatherScatterMask) {
    return tts::MakeGatherScatterTensorPtrOp::create(
        builder, loc, desc.getBase(), xOffsets, gatherScatterMask,
        /*gatherScatterDim=*/0, tensorShape, desc.getMixedStrides(), offsets);
  }
  return tts::MakeGatherScatterTensorPtrOp::create(
      builder, loc, desc.getBase(), xOffsets,
      /*gatherScatterDim=*/0, tensorShape, desc.getMixedStrides(), offsets);
}

struct BoundedTransferInfo {
  SmallVector<Value> sizes;
  Value needsPad;
  Value hasData;
};

static SmallVector<OpFoldResult> getAsOpFoldResults(ValueRange values) {
  SmallVector<OpFoldResult> result;
  result.reserve(values.size());
  for (Value value : values)
    result.push_back(value);
  return result;
}

static BoundedTransferInfo getBoundedTransferInfoFromShape(
    ArrayRef<OpFoldResult> shape, ValueRange indices,
    ArrayRef<int64_t> blockShape, Location loc, OpBuilder &builder) {
  BoundedTransferInfo info;
  info.needsPad = nullptr;
  info.hasData = nullptr;

  for (auto [dim, staticSize] : llvm::enumerate(blockShape)) {
    Value size = getValueOrCreateConstantIndexOp(builder, loc, shape[dim]);
    Value index = castToIndex(builder, indices[dim]);
    Value expected = arith::ConstantIndexOp::create(builder, loc, staticSize);
    Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
    Value available = arith::SubIOp::create(builder, loc, size, index);
    Value nonNegativeAvailable =
        arith::MaxSIOp::create(builder, loc, available, zero);
    Value tooSmall =
        arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::slt,
                              nonNegativeAvailable, expected);
    Value clamped = arith::SelectOp::create(builder, loc, tooSmall,
                                            nonNegativeAvailable, expected);
    info.sizes.push_back(clamped);

    Value dimHasData = arith::CmpIOp::create(
        builder, loc, arith::CmpIPredicate::sgt, clamped, zero);
    if (!info.needsPad) {
      info.needsPad = tooSmall;
      info.hasData = dimHasData;
    } else {
      info.needsPad =
          arith::OrIOp::create(builder, loc, info.needsPad, tooSmall);
      info.hasData =
          arith::AndIOp::create(builder, loc, info.hasData, dimHasData);
    }
  }

  return info;
}

// Convert tt.make_tensor_descriptor to a descriptor payload.
struct MakeTensorDescConverter
    : public OpConversionPattern<triton::MakeTensorDescOp> {
  using OpConversionPattern<triton::MakeTensorDescOp>::OpConversionPattern;

  MakeTensorDescConverter(const TypeConverter &typeConverter,
                          MLIRContext *context)
      : OpConversionPattern<triton::MakeTensorDescOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(triton::MakeTensorDescOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<mlir::Value> ptrShapeStridesPaddingOption;
    llvm::append_values(ptrShapeStridesPaddingOption, adaptor.getBase());
    llvm::append_range(ptrShapeStridesPaddingOption,
                       castToI64(rewriter, adaptor.getShape()));
    llvm::append_range(ptrShapeStridesPaddingOption, adaptor.getStrides());
    auto paddingOption = mlir::arith::ConstantOp::create(
        rewriter, op.getLoc(), rewriter.getI1Type(),
        rewriter.getBoolAttr(adaptor.getPadding() ==
                             triton::PaddingOption::PAD_NAN));
    llvm::append_values(ptrShapeStridesPaddingOption, paddingOption);
    auto roundF32ToTF32 = mlir::arith::ConstantOp::create(
        rewriter, op.getLoc(), rewriter.getI1Type(),
        rewriter.getBoolAttr(false));
    llvm::append_values(ptrShapeStridesPaddingOption, roundF32ToTF32);
    rewriter.replaceOpWithMultiple(op, {ptrShapeStridesPaddingOption});
    return mlir::success();
  }
};

// Convert tt.descriptor_load to tts.load.
struct DescriptorLoadConverter
    : public OpConversionPattern<triton::DescriptorLoadOp> {
  using OpConversionPattern<triton::DescriptorLoadOp>::OpConversionPattern;

  DescriptorLoadConverter(const TypeConverter &typeConverter,
                          MLIRContext *context)
      : OpConversionPattern<triton::DescriptorLoadOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(triton::DescriptorLoadOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto descType = op.getDesc().getType();
    auto descPayload = unpackDescriptor(descType, adaptor.getDesc());
    auto resultType = cast<RankedTensorType>(op.getType());
    auto blockShape = llvm::to_vector(descType.getShape());

    if (!canRankReduceShape(blockShape, resultType.getShape())) {
      return rewriter.notifyMatchFailure(
          op, "descriptor load result shape is not a valid rank-reduction of "
              "the block shape");
    }

    auto indices = castToIndex(rewriter, op.getIndices());
    auto descPtr =
        createDescriptorTensorPtr(rewriter, loc, descPayload, descType);
    auto transferInfo = getBoundedTransferInfoFromShape(
        descPtr.getMixedShape(), indices, blockShape, loc, rewriter);
    auto accessPtr =
        createDescriptorAccessTensorPtr(rewriter, loc, descPtr, indices);
    auto padValue = getPadValue(rewriter, loc, resultType.getElementType(),
                                descPayload.paddingOption);
    auto maskDims = getAsOpFoldResults(transferInfo.sizes);
    Value tensor = tts::LoadOp::create(rewriter, loc, accessPtr.getResult(),
                                       maskDims, padValue)
                       .getResult();
    tensor = getRankReducingExtractSlice(tensor, resultType, blockShape, loc,
                                         rewriter);

    if (resultType.getElementType().isF32()) {
      auto ifOp = scf::IfOp::create(rewriter, loc, resultType,
                                    descPayload.roundF32ToTF32,
                                    /*withElseRegion=*/true);
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(ifOp.thenBlock());
      scf::YieldOp::create(rewriter, loc,
                           roundF32ToTF32(rewriter, loc, tensor));
      rewriter.setInsertionPointToStart(ifOp.elseBlock());
      scf::YieldOp::create(rewriter, loc, tensor);
      tensor = ifOp.getResult(0);
    }

    rewriter.replaceOp(op, tensor);
    return success();
  }
};

// Convert tt.descriptor_store to tts.store.
struct DescriptorStoreConverter
    : public OpConversionPattern<triton::DescriptorStoreOp> {
  using OpConversionPattern<triton::DescriptorStoreOp>::OpConversionPattern;

  DescriptorStoreConverter(const TypeConverter &typeConverter,
                           MLIRContext *context)
      : OpConversionPattern<triton::DescriptorStoreOp>(typeConverter, context) {
  }

  LogicalResult
  matchAndRewrite(triton::DescriptorStoreOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto descType = op.getDesc().getType();
    auto descPayload = unpackDescriptor(descType, adaptor.getDesc());
    auto srcType = cast<RankedTensorType>(op.getSrc().getType());
    auto blockShape = llvm::to_vector(descType.getShape());

    if (!canRankReduceShape(blockShape, srcType.getShape())) {
      return rewriter.notifyMatchFailure(
          op, "descriptor store source shape is not a valid rank-reduction of "
              "the block shape");
    }

    auto indices = castToIndex(rewriter, op.getIndices());
    auto descPtr =
        createDescriptorTensorPtr(rewriter, loc, descPayload, descType);
    auto transferInfo = getBoundedTransferInfoFromShape(
        descPtr.getMixedShape(), indices, blockShape, loc, rewriter);
    auto accessPtr =
        createDescriptorAccessTensorPtr(rewriter, loc, descPtr, indices);
    auto fullSrcType =
        RankedTensorType::get(blockShape, srcType.getElementType());
    Value fullSrc = getRankExpandingValue(op.getSrc(), fullSrcType, blockShape,
                                          loc, rewriter);
    auto maskDims = getAsOpFoldResults(transferInfo.sizes);
    tts::StoreOp::create(rewriter, loc, accessPtr.getResult(), fullSrc,
                         maskDims);
    rewriter.eraseOp(op);
    return success();
  }
};

// Convert tt.descriptor_reduce to tts.reduce.
struct DescriptorReduceConverter
    : public OpConversionPattern<triton::DescriptorReduceOp> {
  using OpConversionPattern<triton::DescriptorReduceOp>::OpConversionPattern;

  DescriptorReduceConverter(const TypeConverter &typeConverter,
                            MLIRContext *context)
      : OpConversionPattern<triton::DescriptorReduceOp>(typeConverter,
                                                        context) {}

  LogicalResult
  matchAndRewrite(triton::DescriptorReduceOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto descType = op.getDesc().getType();
    auto descPayload = unpackDescriptor(descType, adaptor.getDesc());
    auto srcType = cast<RankedTensorType>(op.getSrc().getType());
    auto blockShape = llvm::to_vector(descType.getShape());

    if (static_cast<int64_t>(blockShape.size()) != srcType.getRank() ||
        !llvm::equal(blockShape, srcType.getShape())) {
      return rewriter.notifyMatchFailure(
          op, "descriptor reduce currently expects source shape to match block "
              "shape");
    }

    auto indices = castToIndex(rewriter, op.getIndices());
    auto descPtr =
        createDescriptorTensorPtr(rewriter, loc, descPayload, descType);
    auto accessPtr =
        createDescriptorLinearAccessTensorPtr(rewriter, loc, descPtr, indices);
    bool isUnsigned = descType.getElementType().isUnsignedInteger();
    auto transferInfo = getBoundedTransferInfoFromShape(
        descPtr.getMixedShape(), indices, blockShape, loc, rewriter);
    auto maskDims = getAsOpFoldResults(transferInfo.sizes);
    tts::ReduceOp::create(rewriter, loc, op.getKind(), isUnsigned,
                          accessPtr.getResult(), op.getSrc(), maskDims);
    rewriter.eraseOp(op);
    return success();
  }
};

// Convert tt.descriptor_gather to tts.load from a gather/scatter tensor ptr.
struct DescriptorGatherConverter
    : public OpConversionPattern<triton::DescriptorGatherOp> {
  using OpConversionPattern<triton::DescriptorGatherOp>::OpConversionPattern;

  DescriptorGatherConverter(const TypeConverter &typeConverter,
                            MLIRContext *context)
      : OpConversionPattern<triton::DescriptorGatherOp>(typeConverter,
                                                        context) {}

  LogicalResult
  matchAndRewrite(triton::DescriptorGatherOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto descType = op.getDesc().getType();
    auto descPayload = unpackDescriptor(descType, adaptor.getDesc());
    auto resultType = cast<RankedTensorType>(op.getResult().getType());

    if (resultType.getRank() != 2) {
      return rewriter.notifyMatchFailure(
          op, "descriptor gather currently expects rank-2 result");
    }

    auto descPtr =
        createDescriptorTensorPtr(rewriter, loc, descPayload, descType);
    Value gatherScatterMask = createGatherScatterMask(
        rewriter, loc, op.getXOffsets(), descPtr.getMixedShape()[0]);
    auto ptr = createDescriptorGatherScatterTensorPtr(
        rewriter, loc, descPtr, op.getXOffsets(), op.getYOffset(),
        gatherScatterMask, resultType.getShape());
    auto padValue = getPadValue(rewriter, loc, resultType.getElementType(),
                                descPayload.paddingOption);
    auto maskDims = getDescriptorGatherScatterMaskDims(
        rewriter, loc, resultType.getShape(), descPtr.getMixedShape(),
        op.getYOffset());
    Value result =
        tts::LoadOp::create(rewriter, loc, ptr.getResult(), maskDims, padValue)
            .getResult();
    if (resultType.getElementType().isF32()) {
      Value rounded = roundF32ToTF32(rewriter, loc, result);
      result = arith::SelectOp::create(
          rewriter, loc, descPayload.roundF32ToTF32, rounded, result);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

// Convert tt.descriptor_scatter to tts.store through a gather/scatter tensor
// ptr.
struct DescriptorScatterConverter
    : public OpConversionPattern<triton::DescriptorScatterOp> {
  using OpConversionPattern<triton::DescriptorScatterOp>::OpConversionPattern;

  DescriptorScatterConverter(const TypeConverter &typeConverter,
                             MLIRContext *context)
      : OpConversionPattern<triton::DescriptorScatterOp>(typeConverter,
                                                         context) {}

  LogicalResult
  matchAndRewrite(triton::DescriptorScatterOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto descType = op.getDesc().getType();
    auto descPayload = unpackDescriptor(descType, adaptor.getDesc());
    auto srcType = cast<RankedTensorType>(op.getSrc().getType());

    if (srcType.getRank() != 2) {
      return rewriter.notifyMatchFailure(
          op, "descriptor scatter currently expects rank-2 source");
    }

    auto descPtr =
        createDescriptorTensorPtr(rewriter, loc, descPayload, descType);
    Value gatherScatterMask = createGatherScatterMask(
        rewriter, loc, op.getXOffsets(), descPtr.getMixedShape()[0]);
    auto ptr = createDescriptorGatherScatterTensorPtr(
        rewriter, loc, descPtr, op.getXOffsets(), op.getYOffset(),
        gatherScatterMask, srcType.getShape());
    auto maskDims = getDescriptorGatherScatterMaskDims(
        rewriter, loc, srcType.getShape(), descPtr.getMixedShape(),
        op.getYOffset());
    tts::StoreOp::create(rewriter, loc, ptr.getResult(), op.getSrc(), maskDims);
    rewriter.eraseOp(op);
    return success();
  }
};

class TensorDescriptorTypeConverter : public TypeConverter {
public:
  TensorDescriptorTypeConverter(MLIRContext *context) {
    addConversion([](Type type) { return type; });
    addConversion(
        [context](triton::TensorDescType descType, SmallVectorImpl<Type> &out) {
          auto tensorType = descType.getSignlessBlockType();
          out.push_back(triton::getPointerType(tensorType.getElementType()));
          out.insert(out.end(), 2 * tensorType.getRank(),
                     IntegerType::get(context, 64));
          out.push_back(IntegerType::get(context, 1));
          out.push_back(IntegerType::get(context, 1));
          return success();
        });
  }
};

class TritonTensorDescriptorToStructuredPass
    : public impl::TritonTensorDescriptorToStructuredBase<
          TritonTensorDescriptorToStructuredPass> {

public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<arith::ArithDialect, math::MathDialect, affine::AffineDialect,
                bufferization::BufferizationDialect, scf::SCFDialect,
                tensor::TensorDialect, linalg::LinalgDialect,
                memref::MemRefDialect, triton::TritonDialect, ptr::PtrDialect,
                tptr::TPtrDialect, tts::TritonStructuredDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();

    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());
    TensorDescriptorTypeConverter typeConverter(&getContext());
    auto isLegalValue = [&](Value value) {
      return typeConverter.isLegal(value.getType());
    };

    target
        .addIllegalOp<triton::DescriptorLoadOp, triton::DescriptorStoreOp,
                      triton::DescriptorReduceOp, triton::DescriptorGatherOp,
                      triton::DescriptorScatterOp, triton::MakeTensorDescOp>();

    auto isFunctionSignatureLegal = [&](auto funcOp) {
      auto functionType = cast<FunctionType>(funcOp.getFunctionType());
      return typeConverter.isLegal(functionType.getInputs()) &&
             typeConverter.isLegal(functionType.getResults());
    };
    target.addDynamicallyLegalOp<triton::FuncOp>([&](triton::FuncOp funcOp) {
      return isFunctionSignatureLegal(funcOp);
    });
    auto isCallLegal = [&](auto callOp) {
      return typeConverter.isLegal(callOp->getOperandTypes()) &&
             typeConverter.isLegal(callOp->getResultTypes());
    };
    target.addDynamicallyLegalOp<triton::CallOp>(
        [&](triton::CallOp callOp) { return isCallLegal(callOp); });
    auto isReturnLegal = [&](auto returnOp) {
      return typeConverter.isLegal(returnOp->getOperandTypes());
    };
    target.addDynamicallyLegalOp<triton::ReturnOp>(
        [&](triton::ReturnOp returnOp) { return isReturnLegal(returnOp); });

    target.addDynamicallyLegalOp<tensor::SplatOp, linalg::GenericOp,
                                 linalg::YieldOp, tensor::EmptyOp,
                                 tensor::ExpandShapeOp, tensor::InsertSliceOp,
                                 arith::SelectOp, triton::SplatOp>(
        [&](auto op) {
          return llvm::all_of(
              llvm::concat<Value>(op->getOperands(), op->getResults()),
              isLegalValue);
        });

    target.addLegalDialect<
        arith::ArithDialect, linalg::LinalgDialect, tensor::TensorDialect,
        affine::AffineDialect, bufferization::BufferizationDialect,
        tptr::TPtrDialect, ptr::PtrDialect, memref::MemRefDialect,
        tts::TritonStructuredDialect>();

    patterns.add<MakeTensorDescConverter, DescriptorLoadConverter,
                 DescriptorStoreConverter, DescriptorReduceConverter,
                 DescriptorGatherConverter, DescriptorScatterConverter>(
        typeConverter, patterns.getContext());

    triton::FuncArgRenamer renamer(".");
    renamer.addRenamer([](triton::TensorDescType type,
                          SmallVectorImpl<std::string> &outSuffix) {
      auto tensorType = type.getSignlessBlockType();
      int dims = tensorType.getRank();
      outSuffix.push_back("");
      for (int i = 0; i < dims; i++)
        outSuffix.push_back("shape." + std::to_string(i));
      for (int i = 0; i < dims; i++)
        outSuffix.push_back("stride." + std::to_string(i));
      outSuffix.push_back("padding");
      outSuffix.push_back("roundF32ToTF32");
      return success();
    });
    triton::populateFunctionTypeConversions(typeConverter, renamer, patterns);
    triton::populateArithTypeConversions(typeConverter, patterns);
    scf::populateSCFStructuralTypeConversions(typeConverter, patterns);

    target.addDynamicallyLegalDialect<scf::SCFDialect>([&](Operation *op) {
      return llvm::all_of(
          llvm::concat<Value>(op->getOperands(), op->getResults()),
          isLegalValue);
    });

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    SmallVector<tts::MakeTensorPtrOp> deadTensorPtrs;
    moduleOp.walk([&](tts::MakeTensorPtrOp op) {
      if (op->use_empty())
        deadTensorPtrs.push_back(op);
    });
    for (auto op : deadTensorPtrs)
      op.erase();
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
triton::createTritonTensorDescriptorToStructuredPass() {
  return std::make_unique<TritonTensorDescriptorToStructuredPass>();
}
