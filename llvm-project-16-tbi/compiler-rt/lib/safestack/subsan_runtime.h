//===-- subsan_runtime.h ---------------------------------------*- C++ -*-===//
//
//
//   - 16:8 shadow ratio (16 byte granule → 8 byte meta word)
//   - meta = obj_end_minus_1 | (tag << 56)
//   - inline check: meta_ptr = shadow_base + (ptr >> 4)
//                   meta     = *meta_ptr
//                   diff     = meta - (ptr + size - 1)
//                   trap if (diff >> 48) != 0
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
