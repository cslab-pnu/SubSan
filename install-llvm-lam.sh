#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLVM_SRC="$SCRIPT_DIR/llvm-project-16-lam/llvm"
BUILD_DIR="$SCRIPT_DIR/llvm-build"
CLANG_BIN="$BUILD_DIR/bin/clang"

ARCH=$(uname -m)
if [ "$ARCH" != "x86_64" ]; then
  echo "This script is for x86_64 only (detected: $ARCH). Use install-llvm.sh." >&2
  exit 1
fi
echo "Detected architecture: $ARCH"

EXTRA_C_FLAGS=""
EXTRA_CXX_FLAGS="-include cstdint -include stdint.h -std=c++17"

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake \
  -DLLVM_ENABLE_PROJECTS="clang;lld" \
  -DLLVM_ENABLE_RUNTIMES="compiler-rt" \
  -DCMAKE_BUILD_TYPE=Release \
  -GNinja \
  -DLLVM_PARALLEL_LINK_JOBS=1 \
  -DCOMPILER_RT_SANITIZERS_TO_BUILD="safestack" \
  -DLLVM_TARGETS_TO_BUILD="X86" \
  -DCOMPILER_RT_DEFAULT_TARGET_ONLY=ON \
  -DCLANG_ENABLE_STATIC_ANALYZER=OFF \
  -DCLANG_ENABLE_ARCMT=OFF \
  -DLLVM_INCLUDE_TESTS=OFF \
  -DLLVM_INCLUDE_EXAMPLES=OFF \
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
