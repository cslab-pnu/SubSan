#!/usr/bin/env bash
# Build LLVM/Clang/lld + compiler-rt safestack for SubSan (TBI variant).
# Requirements: cmake, ninja, host clang or gcc.
# Output: ./llvm-build/bin/clang, ./llvm-build/bin/clang++.
# Supported: AArch64 Linux (TBI / PAC capable, e.g., Apple M2 / Asahi), x86_64 Linux.

set -euo pipefail

# Resolve paths from this script's location (no external env.sh required).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLVM_SRC="$SCRIPT_DIR/llvm-project-16-tbi/llvm"
BUILD_DIR="$SCRIPT_DIR/llvm-build"
CLANG_BIN="$BUILD_DIR/bin/clang"

ARCH=$(uname -m)
echo "Detected architecture: $ARCH"

# Workaround: some LLVM 16 headers omit a direct <cstdint>/<stdint.h> include.
EXTRA_C_FLAGS=""
EXTRA_CXX_FLAGS="-include cstdint -include stdint.h -std=c++17"

if [ "$ARCH" = "aarch64" ]; then
  TARGET="AArch64"
  # PAC-capable target; disable outline atomics so the bootstrap clang does not
  # require libcompiler-rt at link time.
  EXTRA_C_FLAGS="-mno-outline-atomics -march=armv8.3-a+pauth"
  EXTRA_CXX_FLAGS="$EXTRA_CXX_FLAGS -mno-outline-atomics -march=armv8.3-a+pauth"
elif [ "$ARCH" = "x86_64" ]; then
  TARGET="X86"
else
  echo "Unsupported architecture: $ARCH" >&2
  exit 1
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake \
  -DLLVM_ENABLE_PROJECTS="clang;lld" \
  -DLLVM_ENABLE_RUNTIMES="compiler-rt" \
  -DCMAKE_BUILD_TYPE=Release \
  -GNinja \
  -DLLVM_PARALLEL_LINK_JOBS=1 \
  -DCOMPILER_RT_SANITIZERS_TO_BUILD="safestack" \
  -DLLVM_TARGETS_TO_BUILD="$TARGET" \
  -DCOMPILER_RT_DEFAULT_TARGET_ONLY=ON \
  -DCLANG_ENABLE_STATIC_ANALYZER=OFF \
  -DCLANG_ENABLE_ARCMT=OFF \
  -DLLVM_INCLUDE_TESTS=OFF \
  -DLLVM_INCLUDE_EXAMPLES=OFF \
  -DCMAKE_SHARED_LINKER_FLAGS="-latomic" \
  -DCMAKE_C_FLAGS="$EXTRA_C_FLAGS" \
  -DCMAKE_CXX_FLAGS="$EXTRA_CXX_FLAGS" \
  "$LLVM_SRC"

ninja -j"$(nproc)"

if [[ ! -f "$CLANG_BIN" ]]; then
  echo "Build failed: clang binary not found at $CLANG_BIN" >&2
  exit 1
fi

echo
echo "=== Build complete ==="
echo "  clang   : $CLANG_BIN"
echo "  clang++ : $BUILD_DIR/bin/clang++"
echo "  lld     : $BUILD_DIR/bin/ld.lld"
