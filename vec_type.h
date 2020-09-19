#ifndef _VEC_TYPE_H_
#define _VEC_TYPE_H_

#include <stdint.h>

#ifdef __clang__
#define DECL_VEC(type, len) \
    typedef type type##len __attribute__((ext_vector_type(len),  aligned(1)))
#else
#define DECL_VEC(type, len) \
    typedef type type##len __attribute__((vector_size(sizeof(type) * len), aligned(1)))
#endif

typedef char s8_;
typedef short s16_;
typedef int s32_;
typedef unsigned char u8_;
typedef unsigned short u16_;
typedef unsigned int u32_;

DECL_VEC(s8_, 4);
DECL_VEC(s8_, 8);
DECL_VEC(s8_, 16);
DECL_VEC(s8_, 32);
DECL_VEC(u8_, 4);
DECL_VEC(u8_, 8);
DECL_VEC(u8_, 16);
DECL_VEC(u8_, 32);
DECL_VEC(s16_, 4);
DECL_VEC(s16_, 8);
DECL_VEC(s16_, 16);
DECL_VEC(u16_, 4);
DECL_VEC(u16_, 8);
DECL_VEC(u16_, 16);
DECL_VEC(s32_, 4);
DECL_VEC(s32_, 8);
DECL_VEC(s32_, 16);
DECL_VEC(u32_, 4);
DECL_VEC(u32_, 8);
DECL_VEC(u32_, 16);

#endif
