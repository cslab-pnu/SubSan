# SubSan Ablation Study Guide

Date: 2026-05-21
Stable baseline: `*.stable_pre_ablation.2026-05-21` 백업 (3개 파일).

## 목적

SubSan 의 다섯 가지 컴포넌트별 runtime overhead 를 SPEC CPU2017 에서
측정하기 위한 toggle 인프라.

| # | 컴포넌트 | Toggle 방식 | Knob |
|---|---|---|---|
| 1 | heap metadata set / clear | runtime env | `SUBSAN_ABLATE_HEAP_META=1` |
| 2 | stack metadata set / clear | compile flag | `-mllvm -subsan-ablate-stack-meta=true` |
| 3 | global metadata set | runtime env | `SUBSAN_ABLATE_GLOBAL_META=1` |
| 4 | stdlib wrapper check | runtime env | `SUBSAN_ABLATE_STDLIB_CHECK=1` |
| 5 | inline runtime check | compile flag | `-mllvm -subsan-ablate-check=true` |

#1·#3·#4 는 **하나의 binary** 로 측정 가능 (env var toggle).
#2·#5 는 IR emit 결정이라 **별도 binary 빌드** 필요.

## 영향 0 보장 (gate off 시)

- **Runtime env (heap/global/stdlib)**: `__builtin_expect(g_*, 0)` 분기 1개 추가만.
  분기 predictor 가 always-not-taken 으로 학습 → fast path overhead 사실상 0.
  Static bool 은 `__subsan_init` 에서 1회 init, 이후 단순 load.
- **Compile flag (stack/check)**: `cl::init(false)` 기본값. emit site 에서
  `if (ClSubsanAblate*) skip` 분기 — flag 가 false 면 기존 IR emit 경로 그대로,
  생성된 binary 는 백업본과 byte-identical 해야 함 (검증 단계 참조).

## 측정 절차

### Build matrix (3종)

```bash
# Build A: baseline (모두 ON, 기존 빌드)
# - 이미 빌드되어 있음. 그대로 사용.

# Build B: -stack-meta (stack tag emit 제거)
EXTRA_CFLAGS  += -mllvm -subsan-ablate-stack-meta=true
EXTRA_LDFLAGS += -Wl,-mllvm,-subsan-ablate-stack-meta=true

# Build C: -check (inline check 제거)
EXTRA_CFLAGS  += -mllvm -subsan-ablate-check=true
EXTRA_LDFLAGS += -Wl,-mllvm,-subsan-ablate-check=true
```

### Run matrix (5종 측정 + baseline)

| 측정 | Build | env |
|---|---|---|
| baseline | A | (none) |
| -heap | A | `SUBSAN_ABLATE_HEAP_META=1` |
| -global | A | `SUBSAN_ABLATE_GLOBAL_META=1` |
| -stdlib | A | `SUBSAN_ABLATE_STDLIB_CHECK=1` |
| -stack | B | (none) |
| -check | C | (none) |

각 env var 은 독립. 조합도 가능 (예: 셋 다 켜서 "no-runtime" 측정).

## 백업 / 복원

```bash
PROJ=/home/kbhetrr/workspace/rangesanitizer/llvm-project-16
BUILD=/home/kbhetrr/workspace/rangesanitizer/llvm-build
TAG=stable_pre_ablation.2026-05-21

# 복원
cp $PROJ/llvm/lib/CodeGen/SafeStack.cpp.$TAG                              $PROJ/llvm/lib/CodeGen/SafeStack.cpp
cp $PROJ/compiler-rt/lib/safestack/subsan_runtime.cpp.$TAG                $PROJ/compiler-rt/lib/safestack/subsan_runtime.cpp
cp $BUILD/lib/clang/16/lib/aarch64-unknown-linux-gnu/libclang_rt.safestack.a.$TAG  $BUILD/lib/clang/16/lib/aarch64-unknown-linux-gnu/libclang_rt.safestack.a

# 재빌드 (libclang_rt 백업이 있으면 보통 불필요, .cpp 만 복원해도 OK)
cd $BUILD && ninja safestack
```

## 영향 0 검증

빌드 직후, 단순 C 파일로 IR diff 검증 (gate off 일 때 백업과 동일해야 함):

```bash
cat > /tmp/abl_test.c <<'EOF'
#include <string.h>
int main(int argc, char **argv) {
  char buf[64];
  memcpy(buf, argv[0], 32);
  return strlen(buf);
}
EOF

# 백업 binary 와 새 binary 의 disassembly 비교 (gate off 시 동일해야 함)
NEW_CC=/home/kbhetrr/workspace/rangesanitizer/llvm-build/bin/clang
$NEW_CC -O2 -no-pie -fsanitize=safe-stack /tmp/abl_test.c -o /tmp/abl_new
# 백업 archive 로 link 한 buton 와 동일한지: 비교는 SafeStack pass 가 emit 한
# IR 단계에서 (-S -emit-llvm) — 동일해야 함.
$NEW_CC -O2 -fsanitize=safe-stack -S -emit-llvm /tmp/abl_test.c -o /tmp/abl_new.ll
diff /tmp/abl_new.ll /tmp/abl_baseline.ll  # 새 빌드 vs 베이스라인 IR
```
