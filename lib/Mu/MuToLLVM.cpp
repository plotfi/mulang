//===- MuToLLVM.cpp - Mu to LLVM IR lowering --------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements lowering of Mu dialect operations to a combination of
// arith, func, and cf dialects, which can then be lowered to LLVM IR via the
// standard MLIR lowering pipeline.
//
//===----------------------------------------------------------------------===//

#include "Mu/MuDialect.h"
#include "Mu/MuOps.h"
#include "Mu/MuPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::mu {
#define GEN_PASS_DEF_MULOWERINGPASS
#include "Mu/MuPasses.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Helper: is this type a float type?
static bool isFloat(Type t) { return isa<FloatType>(t); }

/// Convert a value to i1 if it is not already i1 (compare != 0).
static Value ensureI1(Value val, Location loc, ConversionPatternRewriter &rw) {
  if (val.getType().isInteger(1))
    return val;
  if (isa<FloatType>(val.getType())) {
    Value zero = arith::ConstantOp::create(
        rw, loc, rw.getFloatAttr(val.getType(), 0.0));
    return arith::CmpFOp::create(rw, loc, arith::CmpFPredicate::UNE, val,
                                 zero);
  }
  Value zero = arith::ConstantOp::create(rw, loc, rw.getI32Type(),
                                         rw.getI32IntegerAttr(0));
  return arith::CmpIOp::create(rw, loc, arith::CmpIPredicate::ne, val, zero);
}

//===----------------------------------------------------------------------===//
// Op Lowering Patterns
//===----------------------------------------------------------------------===//

struct ConstantOpLowering : public OpConversionPattern<ConstantOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOp(
        op, arith::ConstantOp::create(
                rewriter, op.getLoc(),
                cast<TypedAttr>(op.getValue())));
    return success();
  }
};

struct ParenOpLowering : public OpConversionPattern<ParenOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ParenOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOp(op, adaptor.getValue());
    return success();
  }
};

struct NegOpLowering : public OpConversionPattern<NegOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(NegOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    if (isFloat(op.getType())) {
      rewriter.replaceOp(
          op, arith::NegFOp::create(rewriter, loc, adaptor.getValue()));
    } else {
      Value zero = arith::ConstantOp::create(
          rewriter, loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
      rewriter.replaceOp(
          op, arith::SubIOp::create(rewriter, loc, zero, adaptor.getValue()));
    }
    return success();
  }
};

struct NotOpLowering : public OpConversionPattern<NotOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(NotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    Value zero = arith::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                           rewriter.getI32IntegerAttr(0));
    Value cmp = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq,
                                      adaptor.getValue(), zero);
    rewriter.replaceOp(
        op, arith::ExtUIOp::create(rewriter, loc, rewriter.getI32Type(), cmp));
    return success();
  }
};

struct InvertOpLowering : public OpConversionPattern<InvertOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(InvertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    Value allOnes =
        arith::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                  rewriter.getI32IntegerAttr(-1));
    rewriter.replaceOp(op, arith::XOrIOp::create(rewriter, loc,
                                                  adaptor.getValue(), allOnes));
    return success();
  }
};

// Macro to define lowering for binary ops with int/float dispatch.
#define DEFINE_BINOP_LOWERING(MuOp, IntArithOp, FloatArithOp)                  \
  struct MuOp##Lowering : public OpConversionPattern<MuOp> {                   \
    using OpConversionPattern::OpConversionPattern;                            \
    LogicalResult                                                              \
    matchAndRewrite(MuOp op, OpAdaptor adaptor,                                \
                    ConversionPatternRewriter &rewriter) const final {          \
      auto loc = op.getLoc();                                                  \
      if (isFloat(op.getType())) {                                             \
        rewriter.replaceOp(                                                    \
            op, FloatArithOp::create(rewriter, loc, adaptor.getLhs(),           \
                                     adaptor.getRhs()));                       \
      } else {                                                                 \
        rewriter.replaceOp(                                                    \
            op, IntArithOp::create(rewriter, loc, adaptor.getLhs(),             \
                                   adaptor.getRhs()));                         \
      }                                                                        \
      return success();                                                        \
    }                                                                          \
  };

DEFINE_BINOP_LOWERING(AddOp, arith::AddIOp, arith::AddFOp)
DEFINE_BINOP_LOWERING(SubOp, arith::SubIOp, arith::SubFOp)
DEFINE_BINOP_LOWERING(MulOp, arith::MulIOp, arith::MulFOp)
DEFINE_BINOP_LOWERING(DivOp, arith::DivSIOp, arith::DivFOp)
DEFINE_BINOP_LOWERING(ModOp, arith::RemSIOp, arith::RemFOp)

#undef DEFINE_BINOP_LOWERING

// Bitwise ops only apply to integers.
#define DEFINE_INT_BINOP_LOWERING(MuOp, ArithOp)                               \
  struct MuOp##Lowering : public OpConversionPattern<MuOp> {                   \
    using OpConversionPattern::OpConversionPattern;                            \
    LogicalResult                                                              \
    matchAndRewrite(MuOp op, OpAdaptor adaptor,                                \
                    ConversionPatternRewriter &rewriter) const final {          \
      rewriter.replaceOp(op,                                                   \
                         ArithOp::create(rewriter, op.getLoc(),                 \
                                         adaptor.getLhs(), adaptor.getRhs())); \
      return success();                                                        \
    }                                                                          \
  };

DEFINE_INT_BINOP_LOWERING(AndOp, arith::AndIOp)
DEFINE_INT_BINOP_LOWERING(OrOp, arith::OrIOp)

#undef DEFINE_INT_BINOP_LOWERING

//===----------------------------------------------------------------------===//
// CmpOp lowering: mu.cmp -> arith.cmpi / arith.cmpf
//===----------------------------------------------------------------------===//

struct CmpOpLowering : public OpConversionPattern<CmpOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(CmpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    auto pred = op.getPredicate();

    if (isFloat(adaptor.getLhs().getType())) {
      arith::CmpFPredicate fpred;
      if (pred == "lt")      fpred = arith::CmpFPredicate::OLT;
      else if (pred == "gt") fpred = arith::CmpFPredicate::OGT;
      else if (pred == "le") fpred = arith::CmpFPredicate::OLE;
      else if (pred == "ge") fpred = arith::CmpFPredicate::OGE;
      else if (pred == "eq") fpred = arith::CmpFPredicate::OEQ;
      else if (pred == "ne") fpred = arith::CmpFPredicate::ONE;
      else return failure();
      rewriter.replaceOp(
          op, arith::CmpFOp::create(rewriter, loc, fpred, adaptor.getLhs(),
                                    adaptor.getRhs()));
    } else {
      arith::CmpIPredicate ipred;
      if (pred == "lt")      ipred = arith::CmpIPredicate::slt;
      else if (pred == "gt") ipred = arith::CmpIPredicate::sgt;
      else if (pred == "le") ipred = arith::CmpIPredicate::sle;
      else if (pred == "ge") ipred = arith::CmpIPredicate::sge;
      else if (pred == "eq") ipred = arith::CmpIPredicate::eq;
      else if (pred == "ne") ipred = arith::CmpIPredicate::ne;
      else return failure();
      rewriter.replaceOp(
          op, arith::CmpIOp::create(rewriter, loc, ipred, adaptor.getLhs(),
                                    adaptor.getRhs()));
    }
    return success();
  }
};

//===----------------------------------------------------------------------===//
// CallOp lowering: mu.call -> func.call
//===----------------------------------------------------------------------===//

struct CallOpLowering : public OpConversionPattern<CallOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOp(
        op, func::CallOp::create(rewriter, op.getLoc(), op.getCallee(),
                                 op.getResultTypes(), adaptor.getArgs()));
    return success();
  }
};

struct OrBoolOpLowering : public OpConversionPattern<OrBoolOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(OrBoolOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    Value lhs = ensureI1(adaptor.getLhs(), loc, rewriter);
    Value rhs = ensureI1(adaptor.getRhs(), loc, rewriter);
    rewriter.replaceOp(op, arith::OrIOp::create(rewriter, loc, lhs, rhs));
    return success();
  }
};

struct AndBoolOpLowering : public OpConversionPattern<AndBoolOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AndBoolOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    Value lhs = ensureI1(adaptor.getLhs(), loc, rewriter);
    Value rhs = ensureI1(adaptor.getRhs(), loc, rewriter);
    rewriter.replaceOp(op, arith::AndIOp::create(rewriter, loc, lhs, rhs));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// FuncOp lowering: mu.func -> func.func
//===----------------------------------------------------------------------===//

struct FuncOpLowering : public OpConversionPattern<FuncOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto funcOp = func::FuncOp::create(rewriter, op.getLoc(), op.getSymName(),
                                       op.getFunctionType());

    // Move the body regions over.
    rewriter.inlineRegionBefore(op.getBody(), funcOp.getBody(),
                                funcOp.end());

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ReturnOp lowering: mu.return -> func.return
//===----------------------------------------------------------------------===//

struct ReturnOpLowering : public OpConversionPattern<ReturnOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOp(
        op,
        func::ReturnOp::create(rewriter, op.getLoc(), adaptor.getInput()));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// IfOp lowering: mu.if -> cf.cond_br
//
// mu.if can contain mu.return (early return from the enclosing function)
// or mu.break (no-op terminator). We lower to control flow branches:
//
//   cf.cond_br %cond, ^then, ^cont
// ^then:
//   <then body...>
//   // if terminated by mu.break: cf.br ^cont
//   // if terminated by mu.return: func.return
// ^cont:
//   <continuation...>
//===----------------------------------------------------------------------===//

struct IfOpLowering : public OpConversionPattern<IfOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    auto *parentBlock = op->getBlock();
    auto *contBlock = rewriter.splitBlock(parentBlock, Block::iterator(op));

    // Get the then region's entry block.
    Region &thenRegion = op.getTrueBranch();
    Block *thenBlock = &thenRegion.front();

    // Handle the terminator of the then block. We need to walk all blocks
    // in the region and replace terminators.
    for (Block &block : thenRegion) {
      auto *terminator = block.getTerminator();
      rewriter.setInsertionPointToEnd(&block);
      if (isa<BreakOp>(terminator)) {
        rewriter.replaceOp(terminator,
                           cf::BranchOp::create(rewriter, loc, contBlock));
      }
      // Note: ReturnOps are left as mu.return and will be converted by
      // ReturnOpLowering after inlining into the parent function.
    }

    // Inline the then region before the continuation block.
    rewriter.inlineRegionBefore(thenRegion, contBlock);

    // Add the conditional branch.
    rewriter.setInsertionPointToEnd(parentBlock);
    cf::CondBranchOp::create(rewriter, loc, adaptor.getCond(), thenBlock,
                             contBlock);

    // Remove dead continuation block if the then branch always returns.
    if (contBlock->hasNoPredecessors())
      rewriter.eraseBlock(contBlock);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// IfElseOp lowering: mu.ifelse -> cf.cond_br with two branches
//===----------------------------------------------------------------------===//

struct IfElseOpLowering : public OpConversionPattern<IfElseOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IfElseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    auto *parentBlock = op->getBlock();
    auto *contBlock = rewriter.splitBlock(parentBlock, Block::iterator(op));

    Region &thenRegion = op.getTrueBranch();
    Region &elseRegion = op.getFalseBranch();
    Block *thenBlock = &thenRegion.front();
    Block *elseBlock = &elseRegion.front();

    // Replace terminators in both regions.
    for (Region *region : {&thenRegion, &elseRegion}) {
      for (Block &block : *region) {
        auto *terminator = block.getTerminator();
        rewriter.setInsertionPointToEnd(&block);
        if (isa<BreakOp>(terminator)) {
          rewriter.replaceOp(terminator,
                             cf::BranchOp::create(rewriter, loc, contBlock));
        }
        // Note: ReturnOps are left as mu.return and will be converted by
        // ReturnOpLowering after inlining into the parent function.
      }
    }

    // Inline both regions before the continuation block.
    rewriter.inlineRegionBefore(thenRegion, contBlock);
    rewriter.inlineRegionBefore(elseRegion, contBlock);

    // Add the conditional branch.
    rewriter.setInsertionPointToEnd(parentBlock);
    cf::CondBranchOp::create(rewriter, loc, adaptor.getCond(), thenBlock,
                             elseBlock);

    // Remove dead continuation block if both branches return (no predecessors).
    if (contBlock->hasNoPredecessors())
      rewriter.eraseBlock(contBlock);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

class MuLoweringPass : public impl::MuLoweringPassBase<MuLoweringPass> {
public:
  using impl::MuLoweringPassBase<MuLoweringPass>::MuLoweringPassBase;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, func::FuncDialect,
                    cf::ControlFlowDialect>();
  }

  void runOnOperation() final {
    ConversionTarget target(getContext());

    // We want to lower all Mu ops and produce arith/func/cf ops.
    target.addIllegalDialect<MuDialect>();
    target.addLegalDialect<arith::ArithDialect, func::FuncDialect,
                           cf::ControlFlowDialect, memref::MemRefDialect,
                           BuiltinDialect>();

    RewritePatternSet patterns(&getContext());
    patterns.add<ConstantOpLowering, ParenOpLowering, NegOpLowering,
                 NotOpLowering, InvertOpLowering, AddOpLowering, SubOpLowering,
                 MulOpLowering, DivOpLowering, ModOpLowering, AndOpLowering,
                 OrOpLowering, OrBoolOpLowering, AndBoolOpLowering,
                 CmpOpLowering, CallOpLowering,
                 FuncOpLowering, ReturnOpLowering, IfOpLowering,
                 IfElseOpLowering>(&getContext());

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::mu
