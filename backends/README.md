<!--- SPDX-FileCopyrightText: Copyright (c) <2026> Intel Corporation. All rights reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0 -->

# Non-CUDA backends

The built-in CPU backend runs cuTile kernels on Linux host arrays and CPU
PyTorch tensors. It is experimental; see the [CPU guide](cpu/README.md) for
requirements, installation, and a sample.

## Select a backend

```python
import cuda.tile as ct

ct.set_backend("cpu")
ct.launch(None, grid, kernel, args)
ct.set_backend(None)  # Restore the CUDA launch path.
```

Backend selection applies to the whole process and is not inferred from the
arguments. Coordinate changes between threads. CPU launches are synchronous;
pass CPU arrays and `None` for the stream. CUDA-only launch options are not
available on the CPU path.

The CPU backend must be installed in the environment before selecting it.
Runtime selection does not change the installed compiler or build variant.

## Options and limits

Use `ct.backend_options({...})` to scope backend options to a block of code.
Options apply to the current thread, and unknown options are rejected. The CPU
backend supports `num_cpu_threads` and `assume_in_bounds`; see the
[CPU guide](cpu/README.md#use-in-a-program).

Compiled kernels are cached. Changing a code-generation option selects a
different compiled kernel; changing the CPU thread count does not recompile.
CUDA-specific hints such as `num_ctas` and `occupancy` do not apply.

## Tests

Run the backend tests from the repository root with `python -m pytest backends`.
The CPU-specific guide has instructions for installing the test
dependencies and running only CPU tests.
