//===-- subsan_runtime.cpp -------------------------------------*- C++ -*-===//

#include "subsan_runtime.h"

#include <dlfcn.h>
#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <unistd.h>

#ifndef PR_SET_TAGGED_ADDR_CTRL
#define PR_SET_TAGGED_ADDR_CTRL 55
#endif
#ifndef PR_TAGGED_ADDR_ENABLE
#define PR_TAGGED_ADDR_ENABLE (1UL << 0)
#endif
// Linux x86 arch_prctl codes (asm/prctl.h, kernel 6.4+).
#ifndef ARCH_ENABLE_TAGGED_ADDR
#define ARCH_ENABLE_TAGGED_ADDR 0x4002
#endif
#ifndef ARCH_GET_MAX_TAG_BITS
#define ARCH_GET_MAX_TAG_BITS 0x4003
#endif
#if defined(__x86_64__)
#include <sys/syscall.h>
#include <asm/unistd.h>
#endif

namespace {

constexpr uintptr_t kShadowAlignment = 16;  // granule
constexpr uintptr_t kShadowGranuleBits = 4;  // log2(16)
constexpr uintptr_t kMetaWordSize = 8;  // sizeof(u64)

inline uintptr_t MemToShadow(uintptr_t p) {
  return __hwasan_shadow_memory_dynamic_address + ((p >> 4) << 3);
}

inline uintptr_t MemToShadowSize(uintptr_t size) {
  return ((size + 15) >> 4) << 3;  // ceil(size/16) * 8
}

bool IsAligned(uintptr_t p, uintptr_t align) {
  return (p & (align - 1)) == 0;
}

uintptr_t RoundUpTo(uintptr_t p, uintptr_t align) {
  return (p + align - 1) & ~(align - 1);
}

uintptr_t RoundDownTo(uintptr_t p, uintptr_t align) {
  return p & ~(align - 1);
}

#if defined(__x86_64__)
constexpr uintptr_t kSubsanDefaultUserBits = 47;  // 128 TB user (4-level)
#elif defined(__aarch64__)
constexpr uintptr_t kSubsanDefaultUserBits = 48;  // 256 TB
#endif
constexpr uintptr_t kSubsanHighMemEnd = (1ULL << kSubsanDefaultUserBits) - 1;
constexpr uintptr_t kSubsanShadowSize = kSubsanHighMemEnd >> 1;

bool InitShadowImpl() {
  constexpr uintptr_t kHints[] = {
      0x0000'0000'1000'0000ULL,  // 256 MB
      0x0000'0000'4000'0000ULL,  // 1 GB
      0x0000'0002'0000'0000ULL,  // 8 GB
      0x0000'0008'0000'0000ULL,  // 32 GB
      0x0000'0020'0000'0000ULL,  // 128 GB
      0x0000'0100'0000'0000ULL,  // 1 TB
      0x0000'0400'0000'0000ULL,  // 4 TB
  };

  uintptr_t shadow_base = 0;
  for (uintptr_t hint : kHints) {
#ifdef MAP_FIXED_NOREPLACE
    void *res = mmap((void *)hint, kSubsanShadowSize, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE |
                         MAP_FIXED_NOREPLACE,
                     -1, 0);
    if (res != MAP_FAILED && (uintptr_t)res == hint) {
      shadow_base = hint;
      break;
    }
    if (res != MAP_FAILED)
      munmap(res, kSubsanShadowSize);
#else
    void *res = mmap((void *)hint, kSubsanShadowSize, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (res != MAP_FAILED) {
      shadow_base = (uintptr_t)res;
      break;
    }
#endif
  }

  if (shadow_base == 0) {
    fprintf(stderr,
            "[subsan] FATAL: all shadow mmap hints failed (size=0x%lx)\n",
            kSubsanShadowSize);
    return false;
  }

#ifdef MADV_HUGEPAGE
  madvise((void *)shadow_base, kSubsanShadowSize, MADV_HUGEPAGE);
#endif

  __hwasan_shadow_memory_dynamic_address = shadow_base;
  if (const char *v = getenv("SUBSAN_VERBOSE"); v && v[0] == '1')
    fprintf(stderr, "[subsan] shadow_base = 0x%lx, size = 0x%lx (cover ptr up to 0x%lx)\n",
            shadow_base, kSubsanShadowSize, kSubsanHighMemEnd);
  return true;
}

#if defined(__x86_64__)
constexpr uint8_t kHeapTagLo = 43;
constexpr uint8_t kHeapTagHi = 63;
#elif defined(__aarch64__)
constexpr uint8_t kHeapTagLo = 171;
constexpr uint8_t kHeapTagHi = 255;
#else
#error "subsan runtime: unsupported architecture (need __x86_64__ or __aarch64__)"
#endif

__attribute__((tls_model("local-exec")))
__thread uint8_t tls_tag_counter = kHeapTagLo;  // heap range start.

uint8_t GenerateTagImpl() {
  uint8_t t = tls_tag_counter;
  tls_tag_counter =
      (tls_tag_counter >= kHeapTagHi) ? kHeapTagLo : (tls_tag_counter + 1);
  return t;
}

using malloc_fn_t = void *(*)(size_t);
using calloc_fn_t = void *(*)(size_t, size_t);
using realloc_fn_t = void *(*)(void *, size_t);
using free_fn_t = void (*)(void *);
using malloc_usable_size_fn_t = size_t (*)(void *);

malloc_fn_t real_malloc = nullptr;
calloc_fn_t real_calloc = nullptr;
realloc_fn_t real_realloc = nullptr;
free_fn_t real_free = nullptr;
malloc_usable_size_fn_t real_malloc_usable_size = nullptr;

void InitLibcSymbols() {
  real_malloc = (malloc_fn_t)dlsym(RTLD_NEXT, "malloc");
  real_calloc = (calloc_fn_t)dlsym(RTLD_NEXT, "calloc");
  real_realloc = (realloc_fn_t)dlsym(RTLD_NEXT, "realloc");
  real_free = (free_fn_t)dlsym(RTLD_NEXT, "free");
  real_malloc_usable_size =
      (malloc_usable_size_fn_t)dlsym(RTLD_NEXT, "malloc_usable_size");
}

// ============================================================================
// Pointer tag bit manipulation
//   AArch64 TBI       : tag in bits 63:56 (top byte, 8 bits)
//   x86_64 LAM_U57    : tag in bits 62:57 (6 bits), bit 63 reserved (must be
//                       0 in user-space canonical addresses)
// ============================================================================
#if defined(__x86_64__)
// LAM_U57: 6-bit tag at bits 62:57. Bit 63 must remain 0.
constexpr int kTagShift = 57;
constexpr uintptr_t kTagMask = 0x3FULL << kTagShift;  // 0x7E00_0000_0000_0000
// Fast-path mask for `(meta - ub) & mask != 0`: bits 63:57 cover tag-mismatch
// 와 OOB-induced borrow propagation. (OOB → bits 62:57 all 1, borrow into bit
// 63 sets it 1 — 7 bits 전부 high.)
constexpr uintptr_t kFastPathMask = 0xFE00000000000000ULL;
constexpr uintptr_t kGranuleIdxMask = 0x001FFFFFFFFFFFFFULL;
// Untag mask for debug print (bits 56:0).
constexpr uintptr_t kAddrMask = 0x01FFFFFFFFFFFFFFULL;
#elif defined(__aarch64__)
// TBI: 8-bit tag at bits 63:56.
constexpr int kTagShift = 56;
constexpr uintptr_t kTagMask = 0xFFULL << kTagShift;
constexpr uintptr_t kFastPathMask = 0xFFFF000000000000ULL;
constexpr uintptr_t kGranuleIdxMask = 0x000FFFFFFFFFFFFFULL;
constexpr uintptr_t kAddrMask = 0x00FFFFFFFFFFFFFFULL;
#endif

inline uintptr_t UntagPtr(uintptr_t p) { return p & ~kTagMask; }
inline uintptr_t AddTagToPointer(uintptr_t p, uint8_t tag) {
  return UntagPtr(p) | (uintptr_t(tag) << kTagShift);
}

}  // namespace

// ============================================================================
// Public API
// ============================================================================
extern "C" {

uintptr_t __hwasan_shadow_memory_dynamic_address = 0;

__attribute__((visibility("default")))
void __subsan_init(void) {
  static bool initialized = false;
  if (initialized) return;
  initialized = true;

#ifndef PR_SET_THP_DISABLE
#define PR_SET_THP_DISABLE 41
#endif
  (void)prctl(PR_SET_THP_DISABLE, 0, 0, 0, 0);

#if defined(__x86_64__)
  long pr_rc = syscall(SYS_arch_prctl, ARCH_ENABLE_TAGGED_ADDR, 6UL);
  if (pr_rc != 0) {
    if (const char *v = getenv("SUBSAN_VERBOSE"); v && v[0] == '1')
      fprintf(stderr,
              "[subsan] WARN: arch_prctl(ARCH_ENABLE_TAGGED_ADDR, 6) failed: "
              "%s — kernel needs LAM support (Linux 6.4+).\n",
              strerror(errno));
  }
#elif defined(__aarch64__)
  if (prctl(PR_SET_TAGGED_ADDR_CTRL, PR_TAGGED_ADDR_ENABLE, 0, 0, 0) != 0) {
    if (const char *v = getenv("SUBSAN_VERBOSE"); v && v[0] == '1')
      fprintf(stderr, "[subsan] WARN: prctl(PR_SET_TAGGED_ADDR_CTRL) failed: %s\n",
              strerror(errno));
  }
#endif

  InitLibcSymbols();
  if (!InitShadowImpl()) {
    fprintf(stderr, "[subsan] FATAL: shadow init failed\n");
    abort();
  }
}

__attribute__((constructor(0))) void __subsan_init_ctor(void) {
  __subsan_init();
}

#if defined(__linux__)
extern "C" {
__attribute__((section(".preinit_array"), used))
void (*__subsan_preinit)(void) = __subsan_init_ctor;
}
#endif

__attribute__((visibility("default")))
__attribute__((target("avx2")))
void __subsan_set_meta(uintptr_t p, uintptr_t size, uint64_t meta) {
  uintptr_t shadow_start = MemToShadow(p);
  uintptr_t shadow_size = MemToShadowSize(size);

  if (__builtin_expect(meta == 0, 0)) {
    __builtin_memset((void *)shadow_start, 0, shadow_size);
    return;
  }

  uint64_t *shadow_words = reinterpret_cast<uint64_t *>(shadow_start);
  uintptr_t num_words = shadow_size >> 3;  // / sizeof(uint64_t)
  for (uintptr_t i = 0; i < num_words; i++)
    shadow_words[i] = meta;
}

__attribute__((visibility("default")))
uintptr_t __subsan_tag_memory(uintptr_t p, uintptr_t size, uint8_t tag) {
  uintptr_t aligned_size = RoundUpTo(size, kShadowAlignment);
  uint64_t meta = (tag == 0) ? 0
                             : ((uint64_t(p + size - 1)) |
                                (uint64_t(tag) << kTagShift));
  __subsan_set_meta(p, aligned_size, meta);
  return AddTagToPointer(p, tag);
}

__attribute__((visibility("default")))
void __subsan_init_global(uintptr_t p, uintptr_t size, uint8_t tag) {
  __subsan_tag_memory(p, size, tag);
}

__attribute__((visibility("default")))
uint8_t __subsan_generate_tag(void) {
  return GenerateTagImpl();
}

__attribute__((visibility("default"), noreturn))
void __subsan_trap_ptr(uintptr_t ptr, uintptr_t access_size) {
  void *ret0 = __builtin_return_address(0);
  fprintf(stderr,
          "[subsan] OOB / UAF detected (LTO-debug)\n"
          "  ptr       = 0x%lx (tag=0x%02x, untag=0x%lx)\n"
          "  size      = %lu\n"
          "  caller[0] = %p\n",
          ptr, (unsigned)((ptr >> kTagShift) & (kTagMask >> kTagShift)),
          ptr & kAddrMask,
          (unsigned long)access_size,
          ret0);
  _exit(134);
}

__attribute__((visibility("default"), noreturn))
void __subsan_trap(void) {
  void *ret0 = __builtin_return_address(0);
  fprintf(stderr,
          "[subsan] OOB / UAF detected\n"
          "  caller[0] = %p\n",
          ret0);
  _exit(134);
}

__attribute__((visibility("default")))
void *__subsan_malloc(size_t size) {
  if (!real_malloc) InitLibcSymbols();
  if (size == 0) return nullptr;
  // 16B align 보장 — libc malloc 결과가 보통 16B align 이지만 명시.
  size_t aligned_size = RoundUpTo(size, kShadowAlignment);
  void *raw = real_malloc(aligned_size);
  if (!raw) return nullptr;
  uint8_t tag = __subsan_generate_tag();
  uintptr_t tagged = __subsan_tag_memory(uintptr_t(raw), size, tag);
  return reinterpret_cast<void *>(tagged);
}

__attribute__((visibility("default")))
void *__subsan_calloc(size_t nmemb, size_t size) {
  if (nmemb && size && nmemb > SIZE_MAX / size) {
    errno = ENOMEM;
    return nullptr;
  }
  size_t total = nmemb * size;
  if (total == 0) return nullptr;

  if (total >= 131072) {
    if (!real_calloc) InitLibcSymbols();
    size_t aligned_total = RoundUpTo(total, kShadowAlignment);
    void *raw = real_calloc(1, aligned_total);
    if (!raw) return nullptr;
    uint8_t tag = __subsan_generate_tag();
    uintptr_t tagged = __subsan_tag_memory(uintptr_t(raw), total, tag);
    return reinterpret_cast<void *>(tagged);
  }

  void *p = __subsan_malloc(total);
  if (p) memset(reinterpret_cast<void *>(UntagPtr(uintptr_t(p))), 0, total);
  return p;
}

__attribute__((visibility("default")))
void *__subsan_realloc(void *ptr, size_t size) {
  if (!ptr) return __subsan_malloc(size);
  if (size == 0) {
    __subsan_free(ptr);
    return nullptr;
  }
  uintptr_t raw = UntagPtr(uintptr_t(ptr));
  size_t old_size = real_malloc_usable_size ? real_malloc_usable_size((void *)raw) : 0;

  if (old_size && size <= old_size && size >= old_size / 2) {
    uint8_t cur_tag = (uint8_t)((uintptr_t(ptr) >> kTagShift) & (kTagMask >> kTagShift));
    if (cur_tag != 0) {
      uintptr_t aligned_size = RoundUpTo(size, kShadowAlignment);
      uint64_t meta = ((uint64_t)(raw + size - 1)) | ((uint64_t)cur_tag << kTagShift);
      __subsan_set_meta(raw, aligned_size, meta);
    }
    return ptr;
  }

  void *new_p = __subsan_malloc(size);
  if (!new_p) return nullptr;
  size_t copy_size = (old_size && old_size < size) ? old_size : size;
  memcpy(reinterpret_cast<void *>(UntagPtr(uintptr_t(new_p))), (void *)raw, copy_size);
  __subsan_free(ptr);
  return new_p;
}

static inline void __subsan_check_free_validity(void *ptr) {
  uintptr_t p = uintptr_t(ptr);
  uint8_t ptr_tag = (uint8_t)((p >> kTagShift) & (kTagMask >> kTagShift));
  uintptr_t raw = UntagPtr(p);

  if (ptr_tag == 0) {
    return;  // skip our checks, let real_free handle
  }

  // 2. shadow meta tag 일치 검증.
  uintptr_t granule_idx = (raw >> 4) & kGranuleIdxMask;
  uint64_t *meta_ptr = reinterpret_cast<uint64_t *>(
      __hwasan_shadow_memory_dynamic_address + granule_idx * 8);
  uint64_t meta_cur = *meta_ptr;
  uint8_t meta_tag_cur =
      (uint8_t)((meta_cur >> kTagShift) & (kTagMask >> kTagShift));
  if (meta_tag_cur != ptr_tag) {
    fprintf(stderr,
            "[subsan] misuse-of-free: shadow tag mismatch "
            "(ptr=0x%lx tag=0x%02x, shadow_tag=0x%02x)\n",
            p, ptr_tag, meta_tag_cur);
    __subsan_trap_ptr(p, 0);
  }

  if (raw >= 16) {
    uintptr_t prev_granule_idx = ((raw - 16) >> 4) & kGranuleIdxMask;
    uint64_t *prev_meta_ptr = reinterpret_cast<uint64_t *>(
        __hwasan_shadow_memory_dynamic_address + prev_granule_idx * 8);
    uint64_t meta_prev = *prev_meta_ptr;
    uint8_t meta_tag_prev =
        (uint8_t)((meta_prev >> kTagShift) & (kTagMask >> kTagShift));
    if (meta_tag_prev == ptr_tag) {
      uintptr_t prev_obj_end_minus_1 = meta_prev & kAddrMask;
      if (prev_obj_end_minus_1 >= raw) {
        fprintf(stderr,
                "[subsan] misuse-of-free: ptr is middle of object "
                "(ptr=0x%lx tag=0x%02x, prev_obj_end=0x%lx)\n",
                p, ptr_tag, prev_obj_end_minus_1);
        __subsan_trap_ptr(p, 0);
      }
      // else: tag collision between different objects — OK
    }
  }
}

__attribute__((visibility("default")))
void __subsan_free(void *ptr) {
  if (!ptr) return;
  if (!real_free) InitLibcSymbols();
  __subsan_check_free_validity(ptr);
  uintptr_t raw = UntagPtr(uintptr_t(ptr));
  // meta clear (tag=0 → trap on subsequent access = UAF detect).
  size_t actual_size =
      real_malloc_usable_size ? real_malloc_usable_size((void *)raw) : 0;
  if (actual_size > 0) {
    size_t aligned_size = RoundUpTo(actual_size, kShadowAlignment);
    __subsan_set_meta(raw, aligned_size, 0);
  }
  real_free((void *)raw);
}

__attribute__((visibility("default")))
void *__noinstrument_dyn_alloc(uintptr_t size, uintptr_t align) {
  (void)align;
  return __subsan_malloc(size);
}

__attribute__((visibility("default")))
void __noinstrument_dyn_free(void *ptr) {
  __subsan_free(ptr);
}

__attribute__((visibility("default")))
void __noinstrument_dyn_free_optional(void *ptr) {
  if (ptr) __subsan_free(ptr);
}

__attribute__((visibility("default")))
void free(void *ptr) {
  __subsan_free(ptr);
}
__attribute__((visibility("default"), alias("free")))
void cfree(void *ptr);

}  // extern "C"

#include <stdarg.h>

static inline void __subsan_check_n_impl(void *target, size_t n) {
  if (n == 0) return;
  uintptr_t p = uintptr_t(target);
  uint8_t tag = (uint8_t)((p >> kTagShift) & (kTagMask >> kTagShift));
  uintptr_t granule_idx = (p >> 4) & kGranuleIdxMask;
  uint64_t *meta_ptr =
      reinterpret_cast<uint64_t *>(__hwasan_shadow_memory_dynamic_address +
                                    granule_idx * 8);
  uint64_t meta = *meta_ptr;
  uintptr_t ub = p + n - 1;
  uint64_t diff = meta - ub;
  // fast path: top tag-window bits 0 → no trap.
  //   TBI : 16-bit window (bits 63:48)
  //   LAM : 7-bit window  (bits 63:57; bit 63 may be set by OOB borrow)
  if ((diff & kFastPathMask) != 0) {
    if (tag != 0)
      __subsan_trap_ptr(p, n);
  }
}

extern "C" {

__attribute__((visibility("default")))
void __subsan_check_n(void *target, size_t n) {
  __subsan_check_n_impl(target, n);
}

// ------------ mem* family ------------
__attribute__((visibility("default")))
void *__subsan_memcpy(void *dst, const void *src, size_t n) {
  if (n == 0) return dst;
  __subsan_check_n_impl((void *)src, n);
  __subsan_check_n_impl(dst, n);
  return memcpy(dst, src, n);
}
__attribute__((visibility("default")))
void *__subsan_memcpy_src_only(void *dst, const void *src, size_t n) {
  if (n == 0) return dst;
  __subsan_check_n_impl((void *)src, n);
  return memcpy(dst, src, n);
}
__attribute__((visibility("default")))
void *__subsan_memcpy_dst_only(void *dst, const void *src, size_t n) {
  if (n == 0) return dst;
  __subsan_check_n_impl(dst, n);
  return memcpy(dst, src, n);
}
__attribute__((visibility("default")))
void *__subsan_memmove(void *dst, const void *src, size_t n) {
  if (n == 0) return dst;
  __subsan_check_n_impl((void *)src, n);
  __subsan_check_n_impl(dst, n);
  return memmove(dst, src, n);
}
__attribute__((visibility("default")))
void *__subsan_memmove_src_only(void *dst, const void *src, size_t n) {
  if (n == 0) return dst;
  __subsan_check_n_impl((void *)src, n);
  return memmove(dst, src, n);
}
__attribute__((visibility("default")))
void *__subsan_memmove_dst_only(void *dst, const void *src, size_t n) {
  if (n == 0) return dst;
  __subsan_check_n_impl(dst, n);
  return memmove(dst, src, n);
}
__attribute__((visibility("default")))
void *__subsan_memset(void *s, int c, size_t n) {
  if (n == 0) return s;
  __subsan_check_n_impl(s, n);
  return memset(s, c, n);
}

// ------------ str* family ------------
__attribute__((visibility("default")))
int __subsan_strcmp(const char *s1, const char *s2) {
  unsigned char c1, c2;
  size_t i;
  for (i = 0;; i++) {
    c1 = (unsigned char)s1[i];
    c2 = (unsigned char)s2[i];
    if (c1 != c2 || c1 == '\0') break;
  }
  __subsan_check_n_impl((void *)s1, i + 1);
  __subsan_check_n_impl((void *)s2, i + 1);
  return (c1 == c2) ? 0 : (c1 < c2) ? -1 : 1;
}
__attribute__((visibility("default")))
int __subsan_strncmp(const char *s1, const char *s2, size_t n) {
  unsigned char c1 = 0, c2 = 0;
  size_t i;
  for (i = 0; i < n; i++) {
    c1 = (unsigned char)s1[i];
    c2 = (unsigned char)s2[i];
    if (c1 != c2 || c1 == '\0') break;
  }
  size_t offset = (i + 1 < n) ? (i + 1) : n;
  __subsan_check_n_impl((void *)s1, offset);
  __subsan_check_n_impl((void *)s2, offset);
  return (c1 == c2) ? 0 : (c1 < c2) ? -1 : 1;
}
__attribute__((visibility("default")))
int __subsan_memcmp(const void *s1, const void *s2, size_t n) {
  __subsan_check_n_impl((void *)s1, n);
  __subsan_check_n_impl((void *)s2, n);
  return memcmp(s1, s2, n);
}
__attribute__((visibility("default")))
size_t __subsan_strlen(const char *s) {
  size_t result = strlen(s);
  __subsan_check_n_impl((void *)s, result + 1);
  return result;
}
__attribute__((visibility("default")))
size_t __subsan_strnlen(const char *s, size_t n) {
  size_t result = strnlen(s, n);
  __subsan_check_n_impl((void *)s, (result + 1 < n) ? (result + 1) : n);
  return result;
}
__attribute__((visibility("default")))
char *__subsan_strcat(char *dst, const char *src) {
  size_t dlen = strlen(dst);
  size_t slen = strlen(src);
  __subsan_check_n_impl((void *)src, slen + 1);
  __subsan_check_n_impl((void *)dst, dlen + slen + 1);
  return strcat(dst, src);
}
__attribute__((visibility("default")))
char *__subsan_strncat(char *dst, const char *src, size_t n) {
  size_t dlen = strlen(dst);
  size_t slen = strnlen(src, n);
  __subsan_check_n_impl((void *)src, slen);
  __subsan_check_n_impl((void *)dst, dlen + slen + 1);
  return strncat(dst, src, n);
}
__attribute__((visibility("default")))
char *__subsan_strcpy(char *dst, const char *src) {
  size_t slen = strlen(src) + 1;
  __subsan_check_n_impl((void *)src, slen);
  __subsan_check_n_impl((void *)dst, slen);
  return strcpy(dst, src);
}
__attribute__((visibility("default")))
char *__subsan_strncpy(char *dst, const char *src, size_t n) {
  __subsan_check_n_impl((void *)dst, n);
  size_t slen = strnlen(src, n);
  __subsan_check_n_impl((void *)src, slen);
  return strncpy(dst, src, n);
}
__attribute__((visibility("default")))
char *__subsan_strdup(const char *s) {
  size_t slen = strlen(s) + 1;
  __subsan_check_n_impl((void *)s, slen);
  void *p = __subsan_malloc(slen);
  if (!p) return nullptr;
  uintptr_t untagged = UntagPtr(uintptr_t(p));
  memcpy((void *)untagged, s, slen);
  return (char *)p;
}

// ------------ int parse family ------------
__attribute__((visibility("default")))
int __subsan_atoi(const char *s) {
  __subsan_check_n_impl((void *)s, strlen(s) + 1);
  return atoi(s);
}
__attribute__((visibility("default")))
long __subsan_atol(const char *s) {
  __subsan_check_n_impl((void *)s, strlen(s) + 1);
  return atol(s);
}
__attribute__((visibility("default")))
long long __subsan_atoll(const char *s) {
  __subsan_check_n_impl((void *)s, strlen(s) + 1);
  return atoll(s);
}
__attribute__((visibility("default")))
long __subsan_strtol(const char *s, char **endptr, int base) {
  __subsan_check_n_impl((void *)s, strlen(s) + 1);
  return strtol(s, endptr, base);
}
__attribute__((visibility("default")))
long long __subsan_strtoll(const char *s, char **endptr, int base) {
  __subsan_check_n_impl((void *)s, strlen(s) + 1);
  return strtoll(s, endptr, base);
}

// ------------ stdio family ------------
__attribute__((visibility("default")))
int __subsan_puts(const char *s) {
  __subsan_check_n_impl((void *)s, strlen(s) + 1);
  return puts(s);
}
__attribute__((visibility("default")))
int __subsan_printf(const char *fmt, ...) {
  __subsan_check_n_impl((void *)fmt, strlen(fmt) + 1);
  va_list ap;
  va_start(ap, fmt);
  int r = vprintf(fmt, ap);
  va_end(ap);
  return r;
}
__attribute__((visibility("default")))
int __subsan_snprintf(char *str, size_t size, const char *fmt, ...) {
  __subsan_check_n_impl((void *)fmt, strlen(fmt) + 1);
  __subsan_check_n_impl(str, size);
  va_list ap;
  va_start(ap, fmt);
  int r = vsnprintf(str, size, fmt, ap);
  va_end(ap);
  return r;
}

__attribute__((visibility("default")))
void *__subsan_mmap(void *addr, size_t length, int prot, int flags, int fd,
                     long offset) {
  return mmap(addr, length, prot, flags, fd, offset);
}
__attribute__((visibility("default")))
int __subsan_munmap(void *addr, size_t length) {
  return munmap(reinterpret_cast<void *>(UntagPtr(uintptr_t(addr))), length);
}

__attribute__((visibility("default")))
char **__subsan_move_argv_to_heap(int argc, char **argv) {
  if (argc < 0 || argv == nullptr) return argv;
  size_t arr_bytes = ((size_t)argc + 1) * sizeof(char *);
  char **new_argv = (char **)__subsan_malloc(arr_bytes);
  if (!new_argv) return argv;  // OOM fallback
  char **new_argv_raw = (char **)UntagPtr((uintptr_t)new_argv);
  for (int i = 0; i < argc; i++) {
    char *src = argv[i];
    if (!src) { new_argv_raw[i] = nullptr; continue; }
    size_t slen = strlen(src) + 1;
    char *dst = (char *)__subsan_malloc(slen);
    if (!dst) { new_argv_raw[i] = src; continue; }  // OOM: fallback untagged
    char *dst_raw = (char *)UntagPtr((uintptr_t)dst);
    memcpy(dst_raw, src, slen);
    new_argv_raw[i] = dst;  // tagged ptr stored
  }
  new_argv_raw[argc] = nullptr;
  return new_argv;
}

}  // extern "C"
