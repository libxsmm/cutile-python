// SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation. All rights
// reserved.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>

namespace llvm {
class StringRef;
}

namespace mlir {
class Pass;
}

namespace cutile {
std::unique_ptr<mlir::Pass>
createPrepareCPUVectorsPass(llvm::StringRef entryPoint);
} // namespace cutile
