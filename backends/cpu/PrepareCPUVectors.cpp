// SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation. All rights
// reserved.
//
// SPDX-License-Identifier: Apache-2.0

#include "PrepareCPUVectors.h"

#include <algorithm>
#include <utility>

#include "mlir/Dialect/Affine/IR/AffineDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/X86/Transforms.h"
#include "mlir/Dialect/X86/X86Dialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/CSE.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"

namespace cutile {
namespace {

class AllocaCollector : public mlir::RewriterBase::Listener {
public:
  void notifyOperationInserted(mlir::Operation *operation,
                               mlir::OpBuilder::InsertPoint previous) override {
    if (!previous.isSet() && mlir::isa<mlir::memref::AllocaOp>(operation))
      allocas.insert(operation);
  }

  void notifyOperationErased(mlir::Operation *operation) override {
    if (mlir::isa<mlir::memref::AllocaOp>(operation))
      allocas.remove(operation);
  }

  llvm::SmallVector<mlir::Operation *> takeAllocas() {
    return allocas.takeVector();
  }

private:
  llvm::SetVector<mlir::Operation *> allocas;
};

class PrepareCPUVectorsPass
    : public mlir::PassWrapper<PrepareCPUVectorsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrepareCPUVectorsPass)

  explicit PrepareCPUVectorsPass(llvm::StringRef entryPoint)
      : prepareX86(entryPoint != "__transform_nano"),
        prepareAMX(entryPoint == "__transform_nano_amx") {}

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::affine::AffineDialect, mlir::arith::ArithDialect,
                    mlir::LLVM::LLVMDialect, mlir::math::MathDialect,
                    mlir::memref::MemRefDialect, mlir::vector::VectorDialect,
                    mlir::x86::X86Dialect>();
  }

  void runOnOperation() override {
    llvm::SmallVector<mlir::func::FuncOp> functions;
    getOperation().walk(
        [&](mlir::func::FuncOp function) { functions.push_back(function); });
    auto apply = [&](auto populatePatterns,
                     mlir::RewriterBase::Listener *listener = nullptr) {
      for (mlir::func::FuncOp function : functions) {
        mlir::RewritePatternSet patterns(&getContext());
        populatePatterns(patterns);
        mlir::GreedyRewriteConfig config;
        config.setListener(listener);
        config.setMaxIterations(mlir::GreedyRewriteConfig::kNoLimit);
        config.setMaxNumRewrites(mlir::GreedyRewriteConfig::kNoLimit);
        if (mlir::failed(mlir::applyPatternsGreedily(
                function, std::move(patterns), config)))
          return mlir::failure();
      }
      return mlir::success();
    };
    if (mlir::failed(apply([](mlir::RewritePatternSet &patterns) {
          mlir::vector::populateCastAwayVectorLeadingOneDimPatterns(patterns);
          mlir::vector::populateDropUnitDimWithShapeCastPatterns(patterns);
        }))) {
      signalPassFailure();
      return;
    }
    if (!prepareX86)
      return;
    if (mlir::failed(
            apply(mlir::x86::populatePrepareVectorContractLoopsPatterns))) {
      signalPassFailure();
      return;
    }
    mlir::IRRewriter rewriter(&getContext());
    for (mlir::func::FuncOp function : functions)
      for (mlir::Region &region : function->getRegions())
        mlir::eliminateTriviallyDeadOps(rewriter, region);
    for (mlir::func::FuncOp function : functions) {
      mlir::DominanceInfo dominance;
      mlir::eliminateCommonSubExpressions(rewriter, dominance, function);
    }
    if (!prepareAMX)
      return;
    AllocaCollector collector;
    if (mlir::failed(apply(
            [](mlir::RewritePatternSet &patterns) {
              mlir::x86::populateVectorContractMultiLevelUnrollForAMXPatterns(
                  patterns);
              mlir::x86::populateVectorContractToAMXDotProductPatterns(
                  patterns);
            },
            &collector))) {
      signalPassFailure();
      return;
    }
    // Non-entry allocas are dynamic in LLVM; 64 B is a cache line/AMX tile row.
    llvm::SmallVector<mlir::Operation *> allocas = collector.takeAllocas();
    for (mlir::Operation *operation : llvm::reverse(allocas)) {
      auto alloca = mlir::cast<mlir::memref::AllocaOp>(operation);
      if (alloca->getNumOperands())
        continue;
      mlir::Block &entry =
          alloca->getParentOfType<mlir::func::FuncOp>().getBody().front();
      alloca->moveBefore(&entry, entry.begin());
      alloca.setAlignment(
          std::max<uint64_t>(alloca.getAlignment().value_or(0), 64));
    }
  }

private:
  bool prepareX86;
  bool prepareAMX;
};

} // namespace

std::unique_ptr<mlir::Pass>
createPrepareCPUVectorsPass(llvm::StringRef entryPoint) {
  return std::make_unique<PrepareCPUVectorsPass>(entryPoint);
}

} // namespace cutile
