//===----------------------------------------------------------------------===//
//
// Copyright (c) Meta Platforms, Inc. and affiliates, Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//
// This pass lowers Triton tensor descriptor ops to memref-based IR.
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
#include "triton-shared/Conversion/TritonTensorDescriptorToMemref/TritonTensorDescriptorToMemref.h"
#include "triton-shared/Dialect/TPtr/IR/TPtrDialect.h"
#include "triton-shared/Utils/Utils.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/Transforms/ArithTypeConversion.h"
#include "triton/Dialect/Triton/Transforms/FunctionTypeConversion.h"

#include "llvm/ADT/STLExtras.h"

#define DEBUG_TYPE "triton-tensor-descriptor-to-memref"

using namespace mlir;

namespace {

#define GEN_PASS_DEF_TRITONTENSORDESCRIPTORTOMEMREF
#include "triton-shared/Conversion/TritonTensorDescriptorToMemref/Passes.h.inc"

static Value castToIndex(OpBuilder &builder, Location loc, Value value) {
  if (value.getType().isIndex())
    return value;
  return arith::IndexCastOp::create(builder, loc, builder.getIndexType(),
                                    value);
}

static SmallVector<Value> castToIndex(OpBuilder &builder, Location loc,
                                      ValueRange values) {
  return llvm::map_to_vector(
      values, [&](Value value) { return castToIndex(builder, loc, value); });
}

static memref::SubViewOp getSubview(Value source, ValueRange offsets,
                                    ValueRange sizes, Location loc,
                                    OpBuilder &builder) {
  auto sourceType = cast<MemRefType>(source.getType());
  SmallVector<OpFoldResult> mixedOffsets(offsets.begin(), offsets.end());
  SmallVector<OpFoldResult> mixedSizes(sizes.begin(), sizes.end());
  SmallVector<OpFoldResult> mixedStrides(sourceType.getRank(),
                                         builder.getIndexAttr(1));
  auto dstType = memref::SubViewOp::inferResultType(sourceType, mixedOffsets,
                                                    mixedSizes, mixedStrides);
  return memref::SubViewOp::create(builder, loc, cast<MemRefType>(dstType),
                                   source, mixedOffsets, mixedSizes,
                                   mixedStrides);
}

static memref::SubViewOp getSubview(Value source, ValueRange offsets,
                                    ArrayRef<int64_t> staticSizes, Location loc,
                                    OpBuilder &builder) {
  auto sourceType = cast<MemRefType>(source.getType());
  SmallVector<OpFoldResult> mixedOffsets(offsets.begin(), offsets.end());
  SmallVector<OpFoldResult> mixedSizes;
  for (int64_t size : staticSizes)
    mixedSizes.push_back(builder.getIndexAttr(size));
  SmallVector<OpFoldResult> mixedStrides(sourceType.getRank(),
                                         builder.getIndexAttr(1));
  auto dstType = memref::SubViewOp::inferResultType(sourceType, mixedOffsets,
                                                    mixedSizes, mixedStrides);
  return memref::SubViewOp::create(builder, loc, cast<MemRefType>(dstType),
                                   source, mixedOffsets, mixedSizes,
                                   mixedStrides);
}

static tensor::ExtractSliceOp getExtractSlice(Value source, ValueRange sizes,
                                              Location loc,
                                              OpBuilder &builder) {
  auto sourceType = cast<RankedTensorType>(source.getType());
  SmallVector<OpFoldResult> offsets(sourceType.getRank(),
                                    builder.getIndexAttr(0));
  SmallVector<OpFoldResult> mixedSizes(sizes.begin(), sizes.end());
  SmallVector<OpFoldResult> strides(sourceType.getRank(),
                                    builder.getIndexAttr(1));
  auto sliceType =
      tensor::ExtractSliceOp::inferResultType(sourceType, mixedSizes);
  return tensor::ExtractSliceOp::create(builder, loc, sliceType, source,
                                        offsets, mixedSizes, strides);
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

static Value createDescriptorReduceValue(OpBuilder &builder, Location loc,
                                         triton::DescriptorReduceKind kind,
                                         Type descriptorElementType,
                                         Value current, Value update) {
  Type type = current.getType();
  bool isFloat = isa<FloatType>(type);
  bool isInteger = type.isIntOrIndex();
  bool isUnsignedInteger = descriptorElementType.isUnsignedInteger();

  switch (kind) {
  case triton::DescriptorReduceKind::ADD:
    if (isFloat)
      return arith::AddFOp::create(builder, loc, current, update);
    return arith::AddIOp::create(builder, loc, current, update);
  case triton::DescriptorReduceKind::MIN:
    if (isFloat)
      return arith::MinimumFOp::create(builder, loc, current, update);
    if (isUnsignedInteger)
      return arith::MinUIOp::create(builder, loc, current, update);
    return arith::MinSIOp::create(builder, loc, current, update);
  case triton::DescriptorReduceKind::MAX:
    if (isFloat)
      return arith::MaximumFOp::create(builder, loc, current, update);
    if (isUnsignedInteger)
      return arith::MaxUIOp::create(builder, loc, current, update);
    return arith::MaxSIOp::create(builder, loc, current, update);
  case triton::DescriptorReduceKind::AND:
    return arith::AndIOp::create(builder, loc, current, update);
  case triton::DescriptorReduceKind::OR:
    return arith::OrIOp::create(builder, loc, current, update);
  case triton::DescriptorReduceKind::XOR:
    return arith::XOrIOp::create(builder, loc, current, update);
  case triton::DescriptorReduceKind::INC: {
    assert(isInteger && "descriptor_reduce inc expects integer element type");
    Value zero =
        arith::ConstantOp::create(builder, loc, builder.getZeroAttr(type));
    Value one = arith::ConstantOp::create(builder, loc,
                                          builder.getIntegerAttr(type, 1));
    Value currentPlusOne = arith::AddIOp::create(builder, loc, current, one);
    Value currentGreaterOrEqual =
        arith::CmpIOp::create(builder, loc,
                              isUnsignedInteger ? arith::CmpIPredicate::uge
                                                : arith::CmpIPredicate::sge,
                              current, update);
    return arith::SelectOp::create(builder, loc, currentGreaterOrEqual, zero,
                                   currentPlusOne);
  }
  case triton::DescriptorReduceKind::DEC: {
    assert(isInteger && "descriptor_reduce dec expects integer element type");
    Value zero =
        arith::ConstantOp::create(builder, loc, builder.getZeroAttr(type));
    Value one = arith::ConstantOp::create(builder, loc,
                                          builder.getIntegerAttr(type, 1));
    Value currentMinusOne = arith::SubIOp::create(builder, loc, current, one);
    Value currentIsZero = arith::CmpIOp::create(
        builder, loc, arith::CmpIPredicate::eq, current, zero);
    Value currentGreater =
        arith::CmpIOp::create(builder, loc,
                              isUnsignedInteger ? arith::CmpIPredicate::ugt
                                                : arith::CmpIPredicate::sgt,
                              current, update);
    Value wrapToUpdate =
        arith::OrIOp::create(builder, loc, currentIsZero, currentGreater);
    return arith::SelectOp::create(builder, loc, wrapToUpdate, update,
                                   currentMinusOne);
  }
  }

  llvm_unreachable("unexpected descriptor_reduce kind");
}

static Type convertTensorDescType(MLIRContext *context,
                                  triton::TensorDescType descType,
                                  Attribute memorySpace = {}) {
  return UnrankedMemRefType::get(
      descType.getSignlessBlockType().getElementType(), memorySpace);
}

static MemRefType getTensorDescViewType(MLIRContext *context,
                                        triton::TensorDescType descType,
                                        Attribute memorySpace = {}) {
  auto rank = descType.getShape().size();
  SmallVector<int64_t> dynamicShape(rank, ShapedType::kDynamic);
  SmallVector<int64_t> dynamicStrides(rank, ShapedType::kDynamic);
  auto layout =
      StridedLayoutAttr::get(context, ShapedType::kDynamic, dynamicStrides);
  return MemRefType::get(dynamicShape,
                         descType.getSignlessBlockType().getElementType(),
                         layout, memorySpace);
}

struct Descriptor {
  Value memref;
  Value paddingOption;
  Value roundF32ToTF32;
};

static Descriptor unpackDescriptor(ValueRange values) {
  assert(values.size() == 3 &&
         "expected tensor descriptors to consist of memref, padding, and tf32 "
         "flag");
  return {values[0], values[1], values[2]};
}

static memref::ReinterpretCastOp
createDescriptorMemRef(OpBuilder &builder, Location loc, Value source,
                       triton::TensorDescType descType, ValueRange shape,
                       ValueRange strides) {
  Attribute memorySpace;
  if (auto sourceType = dyn_cast<BaseMemRefType>(source.getType()))
    memorySpace = sourceType.getMemorySpace();
  if (isa<UnrankedMemRefType>(source.getType())) {
    auto rankedSourceType =
        MemRefType::get({ShapedType::kDynamic},
                        descType.getSignlessBlockType().getElementType(),
                        MemRefLayoutAttrInterface{}, memorySpace);
    source = memref::CastOp::create(builder, loc, rankedSourceType, source);
  }
  auto resultType = cast<MemRefType>(
      getTensorDescViewType(builder.getContext(), descType, memorySpace));
  Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
  SmallVector<OpFoldResult> sizes;
  for (Value size : castToIndex(builder, loc, shape))
    sizes.push_back(size);
  SmallVector<OpFoldResult> mixedStrides;
  for (Value stride : castToIndex(builder, loc, strides))
    mixedStrides.push_back(stride);
  return memref::ReinterpretCastOp::create(builder, loc, resultType, source,
                                           zero, sizes, mixedStrides);
}

static Value getDescriptorMemRefView(OpBuilder &builder, Location loc,
                                     Value desc,
                                     triton::TensorDescType descType) {
  auto descMemRefType = cast<BaseMemRefType>(desc.getType());
  auto viewType = getTensorDescViewType(builder.getContext(), descType,
                                        descMemRefType.getMemorySpace());
  if (desc.getType() == viewType)
    return desc;
  return memref::CastOp::create(builder, loc, viewType, desc);
}

struct BoundedTransferInfo {
  SmallVector<Value> sizes;
  Value needsPad;
  Value hasData;
};

static BoundedTransferInfo getBoundedTransferInfo(Value desc,
                                                  ValueRange indices,
                                                  ArrayRef<int64_t> blockShape,
                                                  Location loc,
                                                  OpBuilder &builder) {
  BoundedTransferInfo info;
  info.needsPad = nullptr;
  info.hasData = nullptr;

  for (auto [dim, staticSize] : llvm::enumerate(blockShape)) {
    Value size = memref::DimOp::create(builder, loc, desc, dim);
    Value expected = arith::ConstantIndexOp::create(builder, loc, staticSize);
    Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
    Value available = arith::SubIOp::create(builder, loc, size, indices[dim]);
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
    auto loc = op.getLoc();
    Value source = nullptr;
    if (auto castOp =
            op.getBase().getDefiningOp<UnrealizedConversionCastOp>()) {
      if (castOp.getInputs().size() == 1 &&
          isa<BaseMemRefType>(castOp.getInputs()[0].getType())) {
        source = castOp.getInputs()[0];
      }
    } else if (isa<BaseMemRefType>(op.getBase().getType())) {
      source = op.getBase();
    }

    if (!source) {
      auto ptrType = cast<triton::PointerType>(op.getBase().getType());
      auto fallbackType = UnrankedMemRefType::get(ptrType.getPointeeType(),
                                                  /*memorySpace=*/0);
      source = UnrealizedConversionCastOp::create(rewriter, loc, fallbackType,
                                                  adaptor.getBase())
                   .getResult(0);
    }

    Value memrefView =
        createDescriptorMemRef(rewriter, loc, source, op.getType(),
                               adaptor.getShape(), adaptor.getStrides());
    Attribute memorySpace;
    if (auto memrefType = dyn_cast<BaseMemRefType>(memrefView.getType()))
      memorySpace = memrefType.getMemorySpace();
    Value memref = memref::CastOp::create(
        rewriter, loc,
        convertTensorDescType(rewriter.getContext(), op.getType(), memorySpace),
        memrefView);
    Value paddingIsNan = arith::ConstantOp::create(
        rewriter, loc, rewriter.getI1Type(),
        rewriter.getBoolAttr(adaptor.getPadding() ==
                             triton::PaddingOption::PAD_NAN));
    Value roundF32ToTF32 = arith::ConstantOp::create(
        rewriter, loc, rewriter.getI1Type(), rewriter.getBoolAttr(false));
    SmallVector<Value> replacement{memref, paddingIsNan, roundF32ToTF32};
    rewriter.replaceOpWithMultiple(op, {replacement});
    return success();
  }
};

// Convert tt.descriptor_load to memref.copy
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
    auto descPayload = unpackDescriptor(adaptor.getDesc());
    auto desc =
        getDescriptorMemRefView(rewriter, loc, descPayload.memref, descType);
    auto resultType = cast<RankedTensorType>(op.getType());
    auto blockShape = llvm::to_vector(descType.getShape());

    if (!canRankReduceShape(blockShape, resultType.getShape())) {
      return rewriter.notifyMatchFailure(
          op, "descriptor load result shape is not a valid rank-reduction of "
              "the block shape");
    }

    auto indices = castToIndex(rewriter, loc, op.getIndices());
    auto transferInfo =
        getBoundedTransferInfo(desc, indices, blockShape, loc, rewriter);

    auto allocType = MemRefType::get(blockShape, resultType.getElementType());
    auto alloc = memref::AllocOp::create(rewriter, loc, allocType);

    auto padValue = getPadValue(rewriter, loc, resultType.getElementType(),
                                descPayload.paddingOption);

    scf::IfOp::create(rewriter, loc, transferInfo.needsPad,
                      [&](OpBuilder &b, Location nestedLoc) {
                        linalg::FillOp::create(b, nestedLoc,
                                               ValueRange{padValue},
                                               ValueRange{alloc});
                        scf::YieldOp::create(b, nestedLoc);
                      });

    scf::IfOp::create(
        rewriter, loc, transferInfo.hasData,
        [&](OpBuilder &b, Location nestedLoc) {
          auto srcSubview =
              getSubview(desc, indices, transferInfo.sizes, nestedLoc, b);
          SmallVector<Value> zeroOffsets(transferInfo.sizes.size());
          llvm::transform(transferInfo.sizes, zeroOffsets.begin(), [&](Value) {
            return arith::ConstantIndexOp::create(b, nestedLoc, 0);
          });
          auto dstSubview =
              getSubview(alloc, zeroOffsets, transferInfo.sizes, nestedLoc, b);
          memref::CopyOp::create(b, nestedLoc, srcSubview, dstSubview);
          scf::YieldOp::create(b, nestedLoc);
        });

    auto fullTensorType =
        RankedTensorType::get(blockShape, resultType.getElementType());
    Value tensor =
        bufferization::ToTensorOp::create(rewriter, loc, fullTensorType, alloc,
                                          true /*restrict*/, true /*writable*/);
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

// Convert tt.descriptor_store to subview + materialize_in_destination.
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
    auto descPayload = unpackDescriptor(adaptor.getDesc());
    auto desc =
        getDescriptorMemRefView(rewriter, loc, descPayload.memref, descType);
    auto srcType = cast<RankedTensorType>(op.getSrc().getType());
    auto blockShape = llvm::to_vector(descType.getShape());

    if (!canRankReduceShape(blockShape, srcType.getShape())) {
      return rewriter.notifyMatchFailure(
          op, "descriptor store source shape is not a valid rank-reduction of "
              "the block shape");
    }

    auto indices = castToIndex(rewriter, loc, op.getIndices());
    auto transferInfo =
        getBoundedTransferInfo(desc, indices, blockShape, loc, rewriter);

    auto fullSrcType =
        RankedTensorType::get(blockShape, srcType.getElementType());
    Value fullSrc = getRankExpandingValue(op.getSrc(), fullSrcType, blockShape,
                                          loc, rewriter);
    scf::IfOp::create(
        rewriter, loc, transferInfo.hasData,
        [&](OpBuilder &b, Location nestedLoc) {
          auto srcSlice =
              getExtractSlice(fullSrc, transferInfo.sizes, nestedLoc, b);
          auto dstSubview =
              getSubview(desc, indices, transferInfo.sizes, nestedLoc, b);
          auto storeOp = bufferization::MaterializeInDestinationOp::create(
              b, nestedLoc, srcSlice, dstSubview);
          storeOp.setWritable(true);
          scf::YieldOp::create(b, nestedLoc);
        });

    rewriter.eraseOp(op);
    return success();
  }
};

// Convert tt.descriptor_reduce to a linalg.reduce over the destination subview.
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
    auto descPayload = unpackDescriptor(adaptor.getDesc());
    auto desc =
        getDescriptorMemRefView(rewriter, loc, descPayload.memref, descType);
    auto srcType = cast<RankedTensorType>(op.getSrc().getType());
    auto blockShape = llvm::to_vector(descType.getShape());

    if (static_cast<int64_t>(blockShape.size()) != srcType.getRank() ||
        !llvm::equal(blockShape, srcType.getShape())) {
      return rewriter.notifyMatchFailure(
          op, "descriptor reduce currently expects source shape to match block "
              "shape");
    }

    auto indices = castToIndex(rewriter, loc, op.getIndices());
    auto dstSubview = getSubview(desc, indices, blockShape, loc, rewriter);
    auto partialMemRefType =
        MemRefType::get(srcType.getShape(), srcType.getElementType());
    auto partialAlloc =
        memref::AllocOp::create(rewriter, loc, partialMemRefType);
    memref::CopyOp::create(rewriter, loc, dstSubview, partialAlloc);
    Value partialTensor =
        bufferization::ToTensorOp::create(rewriter, loc, srcType, partialAlloc,
                                          true /*restrict*/, true /*writable*/);
    auto reduceOp = linalg::ReduceOp::create(
        rewriter, loc, ValueRange{op.getSrc()}, ValueRange{partialTensor},
        ArrayRef<int64_t>{},
        [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
          Value reduced = createDescriptorReduceValue(
              b, nestedLoc, op.getKind(), descType.getElementType(), args[1],
              args[0]);
          linalg::YieldOp::create(b, nestedLoc, reduced);
        });
    auto storeOp = bufferization::MaterializeInDestinationOp::create(
        rewriter, loc, reduceOp.getResult(0), dstSubview);
    storeOp.setWritable(true);

    rewriter.eraseOp(op);
    return success();
  }
};

// Convert tt.descriptor_gather to linalg.generic. The row offsets remain a
// tensor input to the generic body, where each element loads one descriptor
// row.
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
    auto descPayload = unpackDescriptor(adaptor.getDesc());
    auto desc =
        getDescriptorMemRefView(rewriter, loc, descPayload.memref, descType);
    auto resultType = cast<RankedTensorType>(op.getResult().getType());

    if (resultType.getRank() != 2) {
      return rewriter.notifyMatchFailure(
          op, "descriptor gather currently expects rank-2 result");
    }

    auto emptyTensor =
        tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                resultType.getElementType())
            .getResult();

    AffineExpr d0 = rewriter.getAffineDimExpr(0);
    AffineExpr d1 = rewriter.getAffineDimExpr(1);
    SmallVector<AffineMap> affineMaps{
        AffineMap::get(2, 0, d0, rewriter.getContext()),
        AffineMap::get(2, 0, {d0, d1}, rewriter.getContext())};
    SmallVector<utils::IteratorType> iteratorTypes(
        resultType.getRank(), utils::IteratorType::parallel);

    auto padValue = getPadValue(rewriter, loc, resultType.getElementType(),
                                descPayload.paddingOption);

    auto genericOp = linalg::GenericOp::create(
        rewriter, loc, TypeRange{resultType}, ValueRange{op.getXOffsets()},
        ValueRange{emptyTensor}, affineMaps, iteratorTypes,
        [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
          Value xOffset = castToIndex(b, nestedLoc, args[0]);
          Value yOffset = castToIndex(b, nestedLoc, op.getYOffset());
          Value yIndex = linalg::IndexOp::create(b, nestedLoc, 1);
          Value y = arith::AddIOp::create(b, nestedLoc, yOffset, yIndex);

          Value zero = arith::ConstantIndexOp::create(b, nestedLoc, 0);
          Value dim0 = memref::DimOp::create(b, nestedLoc, desc, 0);
          Value dim1 = memref::DimOp::create(b, nestedLoc, desc, 1);
          Value xNonNegative = arith::CmpIOp::create(
              b, nestedLoc, arith::CmpIPredicate::sge, xOffset, zero);
          Value yNonNegative = arith::CmpIOp::create(
              b, nestedLoc, arith::CmpIPredicate::sge, y, zero);
          Value xInBounds = arith::CmpIOp::create(
              b, nestedLoc, arith::CmpIPredicate::slt, xOffset, dim0);
          Value yInBounds = arith::CmpIOp::create(
              b, nestedLoc, arith::CmpIPredicate::slt, y, dim1);
          Value inBounds =
              arith::AndIOp::create(b, nestedLoc, xNonNegative, yNonNegative);
          inBounds = arith::AndIOp::create(b, nestedLoc, inBounds, xInBounds);
          inBounds = arith::AndIOp::create(b, nestedLoc, inBounds, yInBounds);

          auto ifOp = scf::IfOp::create(b, nestedLoc,
                                        TypeRange{resultType.getElementType()},
                                        inBounds, true);
          {
            OpBuilder::InsertionGuard guard(b);
            b.setInsertionPointToStart(&ifOp.getThenRegion().front());
            Value loaded = memref::LoadOp::create(b, nestedLoc, desc,
                                                  ValueRange{xOffset, y});
            scf::YieldOp::create(b, nestedLoc, loaded);

            b.setInsertionPointToStart(&ifOp.getElseRegion().front());
            scf::YieldOp::create(b, nestedLoc, padValue);
          }
          linalg::YieldOp::create(b, nestedLoc, ifOp.getResult(0));
        });
    Value result = genericOp.getResult(0);
    if (resultType.getElementType().isF32()) {
      Value rounded = roundF32ToTF32(rewriter, loc, result);
      result = arith::SelectOp::create(
          rewriter, loc, descPayload.roundF32ToTF32, rounded, result);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

// Convert tt.descriptor_scatter to linalg.generic. The generic body performs
// the descriptor-indexed stores for each source element.
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
    auto descPayload = unpackDescriptor(adaptor.getDesc());
    auto desc =
        getDescriptorMemRefView(rewriter, loc, descPayload.memref, descType);
    auto srcType = cast<RankedTensorType>(op.getSrc().getType());

    if (srcType.getRank() != 2) {
      return rewriter.notifyMatchFailure(
          op, "descriptor scatter currently expects rank-2 source");
    }

    AffineExpr d0 = rewriter.getAffineDimExpr(0);
    AffineExpr d1 = rewriter.getAffineDimExpr(1);
    SmallVector<AffineMap> affineMaps{
        AffineMap::get(2, 0, d0, rewriter.getContext()),
        AffineMap::get(2, 0, {d0, d1}, rewriter.getContext())};
    SmallVector<utils::IteratorType> iteratorTypes(
        srcType.getRank(), utils::IteratorType::parallel);

    linalg::GenericOp::create(
        rewriter, loc, TypeRange{}, ValueRange{op.getXOffsets(), op.getSrc()},
        ValueRange{}, affineMaps, iteratorTypes,
        [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
          Value xOffset = castToIndex(b, nestedLoc, args[0]);
          Value yOffset = castToIndex(b, nestedLoc, op.getYOffset());
          Value yIndex = linalg::IndexOp::create(b, nestedLoc, 1);
          Value y = arith::AddIOp::create(b, nestedLoc, yOffset, yIndex);

          Value zero = arith::ConstantIndexOp::create(b, nestedLoc, 0);
          Value dim0 = memref::DimOp::create(b, nestedLoc, desc, 0);
          Value dim1 = memref::DimOp::create(b, nestedLoc, desc, 1);
          Value xNonNegative = arith::CmpIOp::create(
              b, nestedLoc, arith::CmpIPredicate::sge, xOffset, zero);
          Value yNonNegative = arith::CmpIOp::create(
              b, nestedLoc, arith::CmpIPredicate::sge, y, zero);
          Value xInBounds = arith::CmpIOp::create(
              b, nestedLoc, arith::CmpIPredicate::slt, xOffset, dim0);
          Value yInBounds = arith::CmpIOp::create(
              b, nestedLoc, arith::CmpIPredicate::slt, y, dim1);
          Value inBounds =
              arith::AndIOp::create(b, nestedLoc, xNonNegative, yNonNegative);
          inBounds = arith::AndIOp::create(b, nestedLoc, inBounds, xInBounds);
          inBounds = arith::AndIOp::create(b, nestedLoc, inBounds, yInBounds);

          scf::IfOp::create(b, nestedLoc, inBounds,
                            [&](OpBuilder &b, Location ifLoc) {
                              memref::StoreOp::create(b, ifLoc, args[1], desc,
                                                      ValueRange{xOffset, y});
                              scf::YieldOp::create(b, ifLoc);
                            });
          linalg::YieldOp::create(b, nestedLoc);
        });

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
          out.push_back(convertTensorDescType(context, descType));
          out.push_back(IntegerType::get(context, 1));
          out.push_back(IntegerType::get(context, 1));
          return success();
        });
    auto createCast = [&](OpBuilder &builder, Type resultType,
                          ValueRange inputs, Location loc) -> Value {
      return UnrealizedConversionCastOp::create(builder, loc, resultType,
                                                inputs)
          .getResult(0);
    };
    addTargetMaterialization(createCast);
    addSourceMaterialization(createCast);
  }
};

class TritonTensorDescriptorToMemrefPass
    : public impl::TritonTensorDescriptorToMemrefBase<
          TritonTensorDescriptorToMemrefPass> {

public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<
        arith::ArithDialect, math::MathDialect, affine::AffineDialect,
        bufferization::BufferizationDialect, scf::SCFDialect,
        tensor::TensorDialect, linalg::LinalgDialect, memref::MemRefDialect,
        triton::TritonDialect, ptr::PtrDialect, tptr::TPtrDialect>();
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

    target.addDynamicallyLegalOp<
        tensor::SplatOp, linalg::GenericOp, linalg::YieldOp, tensor::EmptyOp,
        tensor::ExpandShapeOp, tensor::InsertSliceOp, arith::SelectOp>(
        [&](auto op) {
          return llvm::all_of(
              llvm::concat<Value>(op->getOperands(), op->getResults()),
              isLegalValue);
        });

    target.addLegalDialect<
        arith::ArithDialect, linalg::LinalgDialect, tensor::TensorDialect,
        affine::AffineDialect, bufferization::BufferizationDialect,
        tptr::TPtrDialect, ptr::PtrDialect, memref::MemRefDialect>();
    target.addLegalOp<UnrealizedConversionCastOp>();

    patterns.add<MakeTensorDescConverter, DescriptorLoadConverter,
                 DescriptorStoreConverter, DescriptorReduceConverter,
                 DescriptorGatherConverter, DescriptorScatterConverter>(
        typeConverter, patterns.getContext());

    triton::FuncArgRenamer renamer(".");
    renamer.addRenamer([](triton::TensorDescType type,
                          SmallVectorImpl<std::string> &outSuffix) {
      outSuffix.push_back("");
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
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
triton::createTritonTensorDescriptorToMemrefPass() {
  return std::make_unique<TritonTensorDescriptorToMemrefPass>();
}
