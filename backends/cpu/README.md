<!--- SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation. All rights reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0 -->

# CPU backend

Run cuTile kernels on Linux CPUs with NumPy-compatible arrays or CPU PyTorch
tensors. Grids can have one, two, or three dimensions. The backend is
experimental, so verify results for your workload; compile times can be long
and performance depends on the kernel and host CPU.

## Prerequisites

- Linux, Git, and [uv](https://docs.astral.sh/uv/getting-started/installation/).
- GCC with OpenMP development headers and runtime, plus Clang and LLD to build
  LLVM/MLIR. GCC builds cuTile's extensions.

An NVIDIA GPU, CUDA driver, or system CUDA Toolkit is not required. The build
dependencies provide the CUDA headers needed by the extension.

## Build and install

From the repository root:

```bash
uv venv --python 3.12
. .venv/bin/activate
CUTILE_BUILD_VARIANT=cpu uv sync --group cpu --no-build-isolation-package cuda-tile
```

The initial build compiles LLVM/MLIR and can take a while. Later builds reuse
the build tree. Keep `--no-build-isolation-package cuda-tile` when syncing this
environment.

Running kernels does not require a compiler toolchain, LLVM executables,
development files, or a source checkout. The target system must provide the
C/C++ runtimes and GCC's OpenMP runtime (`libgomp.so.1`), which are not bundled.
On Debian or Ubuntu, install the runtime with:

```bash
sudo apt-get install libgomp1
```

## Run a sample

With the environment active, from the repository root:

```bash
python samples/cpu/VectorAddition-cpu.py --correctness-check
```

More examples and benchmarks are in [samples/cpu](../../samples/cpu). Check each
script's `--help` for supported sizes, correctness checks, and thread settings.

## Use in a program

Select the backend, pass CPU arrays, and use `None` as the stream:

```python
import cuda.tile as ct

ct.set_backend("cpu")
with ct.backend_options({"num_cpu_threads": 4}):
    ct.launch(None, grid, kernel, cpu_args)
```

`kernel` is a decorated cuTile kernel, `grid` is its block grid, and `cpu_args`
contains its arguments. Launches are synchronous. When porting a CUDA program,
keep the kernel where supported, move its arrays to the CPU, and remove CUDA
stream and device assumptions from the host code. `ct.set_backend(None)`
restores the CUDA launch path.

### Options

`num_cpu_threads` sets the number of OpenMP workers. The default, `0`, uses
OpenMP's maximum and honors `OMP_NUM_THREADS`. Changing this option does not
recompile kernels.

Bounds checks are enabled by default. Set `assume_in_bounds=True` only when
every tile load and store is in bounds.

## Troubleshooting

Set `MLIR_ENABLE_DUMP=1` to print generated MLIR during compilation, or set
`CUTILE_CPU_DUMP_MLIR` to a file path to write the lowered MLIR.

If the CPU launcher cannot find `libgomp.so.1`, install GCC's OpenMP runtime.
If loading a compiled kernel fails, check that `TMPDIR` permits executable
files; kernels are loaded from temporary shared libraries.

## Run tests

From the repository root, install the test tools and run CPU backend tests:

```bash
uv pip install pytest pytest-benchmark pytest-env
python -m pytest backends/cpu
```

Tests requiring another native build variant are skipped.
