// SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation. All rights
// reserved.
//
// SPDX-License-Identifier: Apache-2.0

#include "compiler.h"
#include "cuda_tile/Bytecode/Reader/BytecodeReader.h"
#include "cuda_tile/Dialect/CudaTile/IR/Attributes.h"
#include "cuda_tile/Dialect/CudaTile/IR/Dialect.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/TileIRToMLIR/Passes.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/Affine/IR/AffineDialect.h"
#include "mlir/Dialect/Affine/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/X86/Transforms.h"
#include "mlir/Dialect/X86/X86Dialect.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/MemoryBuffer.h"
#include <cstdlib>
#include <stdexcept>

namespace cutile {

static mlir::DialectRegistry createRegistry() {
  mlir::DialectRegistry registry;
  registry.insert<mlir::affine::AffineDialect, mlir::arith::ArithDialect,
                  mlir::cf::ControlFlowDialect, mlir::func::FuncDialect,
                  mlir::gpu::GPUDialect, mlir::index::IndexDialect,
                  mlir::LLVM::LLVMDialect, mlir::math::MathDialect,
                  mlir::memref::MemRefDialect, mlir::scf::SCFDialect,
                  mlir::ub::UBDialect, mlir::vector::VectorDialect,
                  mlir::x86::X86Dialect, mlir::cuda_tile::CudaTileDialect>();
  mlir::func::registerInlinerExtension(registry);
  mlir::affine::registerValueBoundsOpInterfaceExternalModels(registry);
  mlir::arith::registerValueBoundsOpInterfaceExternalModels(registry);
  mlir::memref::registerValueBoundsOpInterfaceExternalModels(registry);
  mlir::scf::registerValueBoundsOpInterfaceExternalModels(registry);
  mlir::vector::registerValueBoundsOpInterfaceExternalModels(registry);
  mlir::arith::registerConvertArithToLLVMInterface(registry);
  mlir::cf::registerConvertControlFlowToLLVMInterface(registry);
  mlir::registerConvertFuncToLLVMInterface(registry);
  mlir::index::registerConvertIndexToLLVMInterface(registry);
  mlir::registerConvertMathToLLVMInterface(registry);
  mlir::registerConvertMemRefToLLVMInterface(registry);
  mlir::ub::registerConvertUBToLLVMInterface(registry);
  mlir::vector::registerConvertVectorToLLVMInterface(registry);
  mlir::registerConvertX86ToLLVMInterface(registry);
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  registerBackendTranslations(registry);
  return registry;
}

Compilation::Compilation(bool printIR)
    : context(createRegistry(), mlir::MLIRContext::Threading::DISABLED),
      printIR(printIR ||
              [] {
                const char *value = std::getenv("MLIR_ENABLE_DUMP");
                return value && llvm::StringRef(value) == "1";
              }()),
      handler(&context, [this](mlir::Diagnostic &diagnostic) {
        llvm::raw_string_ostream stream(diagnostics);
        diagnostic.print(stream);
        stream << '\n';
        return mlir::success();
      }) {}

mlir::OwningOpRef<mlir::ModuleOp>
Compilation::readTileIR(llvm::StringRef bytecode) {
  context.getOrLoadDialect<mlir::cuda_tile::CudaTileDialect>();
  auto buffer = llvm::MemoryBuffer::getMemBufferCopy(bytecode, "<TileIR>");
  if (!mlir::cuda_tile::isTileIRBytecode(buffer->getMemBufferRef()))
    throw std::invalid_argument("cuTile compiler: invalid TileIR bytecode");
  auto decoded =
      mlir::cuda_tile::readBytecode(buffer->getMemBufferRef(), context);
  if (!decoded)
    fail("reading TileIR bytecode");
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  module->getBody()->push_back(decoded.release().getOperation());
  return module;
}

mlir::OwningOpRef<mlir::ModuleOp>
Compilation::readMLIR(llvm::StringRef source) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(source, &context);
  if (!module)
    fail("parsing MLIR");
  return module;
}

void Compilation::lowerTileIR(
    mlir::ModuleOp module, llvm::StringRef options,
    llvm::function_ref<void(mlir::OpPassManager &)> addBackendPasses) {
  module.walk([](mlir::Operation *op) {
    auto rounding =
        op->getAttrOfType<mlir::cuda_tile::RoundingModeAttr>("rounding_mode");
    if (!rounding)
      return;
    using Mode = mlir::cuda_tile::RoundingMode;
    switch (rounding.getValue()) {
    case Mode::NEAREST_EVEN:
    case Mode::APPROX:
    case Mode::FULL:
    case Mode::NEAREST_INT_TO_ZERO:
      return;
    default:
      throw std::invalid_argument(
          std::string(backendName) + " backend: unsupported rounding mode " +
          mlir::cuda_tile::stringifyRoundingMode(rounding.getValue()).str() +
          " on " + op->getName().getStringRef().str());
    }
  });
  mlir::PassManager pm(&context);
  auto onError = [&](const llvm::Twine &message) {
    diagnostics += message.str() + "\n";
    return mlir::failure();
  };
  if (failed(mlir::buildTileIRToMLIRPipeline(pm, options, onError)))
    fail("constructing the TileIR pipeline");
  addBackendPasses(pm);
  // Translating TileIR source locations to LLVM IR crashes; drop them.
  pm.addPass(mlir::createStripDebugInfoPass());
  run(module, pm);
  // Nearest-even is arith's default, including for narrowing casts.
  module.walk([](mlir::Operation *op) {
    auto rounding =
        op->getAttrOfType<mlir::arith::RoundingModeAttr>("roundingmode");
    if (rounding &&
        rounding.getValue() == mlir::arith::RoundingMode::to_nearest_even)
      op->removeAttr("roundingmode");
  });
}

void Compilation::run(mlir::ModuleOp module, mlir::PassManager &pm) {
  if (printIR)
    pm.enableIRPrinting([](mlir::Pass *, mlir::Operation *) { return true; },
                        [](mlir::Pass *, mlir::Operation *) { return false; });
  if (failed(pm.run(module)))
    fail("running the pass pipeline");
  if (!diagnostics.empty()) {
    llvm::errs() << diagnostics;
    diagnostics.clear();
  }
}

void Compilation::fail(llvm::StringRef stage) {
  throw std::runtime_error("cuTile compiler: failed " + stage.str() + "\n" +
                           diagnostics);
}

std::string printModule(mlir::ModuleOp module) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  module.print(stream);
  return text;
}

} // namespace cutile

NB_MODULE(_compiler, module) {
  cutile::registerBackendPasses();
  module.attr("backend") = cutile::backendName;
  module.attr("llvm_version") = LLVM_VERSION_STRING;
  cutile::bindBackend(module);
}
