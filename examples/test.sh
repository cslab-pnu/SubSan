#!/usr/bin/env bash
ulimit -c 0
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBSAN_TOP="$(cd "$SCRIPT_DIR/.." && pwd)"

SUBSAN_LLVM_BUILD="$SUBSAN_TOP/llvm-build"
SUBSAN_C="$SUBSAN_LLVM_BUILD/bin/clang"
SUBSAN_CXX="$SUBSAN_LLVM_BUILD/bin/clang++"

if [[ ! -x "$SUBSAN_C" ]]; then
  echo "Error: clang not found at $SUBSAN_C" >&2
  echo "Run ../install-llvm-{tbi,lam}.sh first to build the subsan-patched clang." >&2
  exit 1
fi

ARCH="$(uname -m)"

SUBSAN_CFLAGS_COMMON="-O2 -g -no-pie -fsanitize=safe-stack -Wno-int-conversion -Wno-deprecated-non-prototype"
SUBSAN_LDFLAGS_COMMON="-fsanitize=safe-stack -fuse-ld=lld -no-pie -z muldefs -Wl,-z,muldefs -Wl,-u,__subsan_init"

SUBSAN_CFLAGS="$SUBSAN_CFLAGS_COMMON"
SUBSAN_LDFLAGS="$SUBSAN_LDFLAGS_COMMON"

cd "$SCRIPT_DIR"

echo "=== subsan_shadow test ==="
echo "Compiler : $SUBSAN_C"
echo "Arch     : $ARCH"
echo "CFLAGS   : $SUBSAN_CFLAGS"
echo "LDFLAGS  : $SUBSAN_LDFLAGS"
echo

echo "Compiling test programs..."
$SUBSAN_C buffer_overflow.c  $SUBSAN_CFLAGS -o buffer_overflow  $SUBSAN_LDFLAGS
$SUBSAN_C buffer_underflow.c $SUBSAN_CFLAGS -o buffer_underflow $SUBSAN_LDFLAGS
$SUBSAN_C use_after_free.c   $SUBSAN_CFLAGS -o use_after_free   $SUBSAN_LDFLAGS
echo "Done."
echo

run_test() {
  local label="$1"; shift
  echo "--- $label ---"
  "$@"
  echo "Exit: $?"
  echo
}

run_test "[buffer-overflow]  valid index (expected: normal exit)"    ./buffer_overflow 40
run_test "[buffer-overflow]  invalid index (expected: subsan trap)"  ./buffer_overflow 64
run_test "[buffer-underflow] (expected: subsan trap)"                ./buffer_underflow 4
run_test "[use-after-free]   (expected: subsan trap)"                ./use_after_free
