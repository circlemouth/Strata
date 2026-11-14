# Isolated HIP compile plan: baseline B, then candidate C

Status: **not executed**. CPU tests do not substitute for this build. A working
HIP compiler and development packages have not been demonstrated for this
candidate. Compiling on a host that also runs a service still consumes RAM, CPU
and disk and needs a separately approved resource budget and location.

This plan does not authorize service interruption, a model load, GPU access,
package installation, container creation, driver changes, toolchain updates,
network changes or a production cutover. Keep live qualification separate.

## Freeze two independent inputs

- **B:** unmodified `Niko1221/Strata` at
  `13eedf552417a6a791a7d13c6597e24300ee3a09`.
- **C:** the exact reviewed candidate commit from this development branch. Record
  its full SHA before configuring. Do not substitute an advancing branch tip.
- Use separate source and build directories. Never reuse the live service's
  checkout, configuration, output directory, installed binary or model files.
- Record the same compiler, CMake, Ninja, HIP/ROCm development-package versions,
  compile definitions, vendored dependency revisions and target architecture for
  both. Keep machine-specific paths and raw logs private.
- Do not set `HSA_OVERRIDE_GFX_VERSION`. `gfx1151` must remain the actual compile
  target; a successful `gfx1100` build is not this test.

## Verify development dependencies without modifying the host

The pinned `cmake/hip_backend.cmake` requires HIP and hipBLAS CMake development
packages and their exported targets. It explicitly requires rocBLAS for
`hipblas_VERSION < 1.0` and otherwise searches it optionally. hipBLASLt is optional.
A runtime shared library or installed running application is not proof that its
headers, compiler and CMake development targets exist. rocWMMA is not a required
CMake dependency in this pinned build path.

Use an already approved isolated toolchain. If it is missing, stop with the exact
missing dependency; do not install or alter anything to make this plan proceed.
Keep optional-package discovery identical between B and C. In particular, a
hipBLASLt-enabled C build cannot be compared to a B build that lacked it without
recording and resolving the mismatch.

## Compile-only procedure after approval

The following are templates. `SOURCE_B`, `SOURCE_C`, `BUILD_B` and `BUILD_C` denote
separate approved directories, not production locations. Run each configure from
its respective source directory so `--preset` resolves there. Use the same low,
explicit job limit for both; increase it only within the approved resource budget.

```sh
# From SOURCE_B:
cmake --preset hip-compile -B BUILD_B \
  -DCMAKE_HIP_ARCHITECTURES=gfx1151 \
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF \
  -DSTRATA_BUILD_TESTS=OFF
cmake --build BUILD_B --target strata --parallel 1

# Only after B succeeds, from SOURCE_C:
cmake --preset hip-compile -B BUILD_C \
  -DCMAKE_HIP_ARCHITECTURES=gfx1151 \
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF \
  -DSTRATA_BUILD_TESTS=OFF
cmake --build BUILD_C --target strata --parallel 1
```

Do not run either binary, `ctest`, a device probe, model setup, model conversion or
inference as part of compile-only validation. Do not install either build. Hash
successful binary artifacts and retain the exact configuration and exit status.
Never publish private absolute paths, environment values, process details, model
identifiers or unreviewed build logs alongside source code.

## Diagnose without obscuring the baseline

- **B configure fails:** missing toolchain/development dependency or baseline
  configuration issue. C has not been tested; do not blame or excuse its code.
- **B compile fails:** preserve the first reproducible error and stop this matrix.
  Do not silently change B, the target architecture or build definitions.
- **B passes, C configure/compile fails:** candidate-specific regression until
  investigated. Fix in a new local commit, then rerun the same B/C conditions.
- **Both compile:** this proves compilation and linking only. It does not prove
  device eligibility, kernel numerical parity, P2 execution, memory ceilings,
  cancellation safety, graph lifecycle or performance.

A later separately authorized synthetic GPU phase must compile and run the
candidate parity tests on the real supported device and verify both register and
split behavior. The batched top-k implementation still uses register selection.
The FP16 planner/workspace has no production kernel path. Neither feature becomes
qualified or enabled merely because C compiles.
