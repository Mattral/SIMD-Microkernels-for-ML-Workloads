# Security

This repository is a native C++ project with Python bindings. Security considerations are focused on safe memory handling, dependency hygiene, and build integrity.

## Native code safety

* SIMD kernels use raw pointers and aligned memory.
* The code should be audited for out-of-bounds accesses when modifying block loops.
* Undefined behavior in C++ can lead to memory corruption and must be avoided.

Recommended mitigation:

* Use sanitizers during development (`-fsanitize=address,undefined`).
* Run unit tests and fuzz edge cases for matrix dimension handling.
* Validate pointer lifetimes when exposing buffers to Python.

## Python bindings

The `pybind11` wrapper exposes native functions to Python.

* Inputs are validated for shape and type before dispatch.
* Output arrays are allocated with the correct size when `C=None`.
* The wrapper avoids unnecessary copies where possible.

Recommended mitigation:

* Keep the binding layer minimal and avoid complex reference lifetimes.
* Audit `pybind11` usage for any conversion or buffer protocol assumptions.

## Dependency hygiene

Third-party dependencies are minimal:

* `pybind11` for Python bindings
* CMake and the host toolchain for building

Recommended mitigation:

* Pin tooling versions in documentation or CI workflows.
* Verify package sources if using system package managers or pip.
* Use reproducible build environments where possible.

## Supply chain and build integrity

* Prefer a known compiler/toolchain for release builds.
* Validate build artifacts using code review and trusted OS packages.
* Do not run untrusted binaries in privileged environments.

## Operational guidance

This project is intended for experimentation and benchmarking, not as a secure production runtime.
For production use, a hardened native library should include additional controls such as sandboxing, explicit memory ownership models, and platform-specific mitigations.
