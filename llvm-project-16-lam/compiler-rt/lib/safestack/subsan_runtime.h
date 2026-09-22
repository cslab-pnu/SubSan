//===-- subsan_runtime.h ---------------------------------------*- C++ -*-===//
//
//   - 16:8 shadow ratio (16 byte granule → 8 byte meta word)
//   - meta = obj_end_minus_1 | (tag << kTagShift)
//       AArch64 TBI    : kTagShift = 56 (8-bit tag in bits 63:56)
//       x86_64 LAM_U57 : kTagShift = 57 (6-bit tag in bits 62:57, bit 63 = 0)
//   - inline check: meta_ptr = shadow_base + (ptr >> 4)
//                   meta     = *meta_ptr
//                   diff     = meta - (ptr + size - 1)
//                   trap if (diff & kFastPathMask) != 0
//       TBI fast-path mask: 0xFFFF000000000000 (top 16 bits)
//       LAM fast-path mask: 0xFE00000000000000 (top 7 bits: bits 63:57)
//
//===---------------------------------------------------------------------===//

#ifndef SUBSAN_RUNTIME_H
#define SUBSAN_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

extern "C" {
extern uintptr_t __hwasan_shadow_memory_dynamic_address;

void __subsan_init(void);
void __subsan_set_meta(uintptr_t p, uintptr_t size, uint64_t meta);

uintptr_t __subsan_tag_memory(uintptr_t p, uintptr_t size, uint8_t tag);

void __subsan_init_global(uintptr_t p, uintptr_t size, uint8_t tag);

uint8_t __subsan_generate_tag(void);

void __subsan_trap(void) __attribute__((noreturn));

void *__subsan_malloc(size_t size);
void *__subsan_calloc(size_t nmemb, size_t size);
void *__subsan_realloc(void *ptr, size_t size);
void __subsan_free(void *ptr);

}  // extern "C"

#endif  // SUBSAN_RUNTIME_H
