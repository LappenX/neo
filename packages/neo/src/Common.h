#ifndef NEO_COMMON_H
#define NEO_COMMON_H

#include <stdint.h>
#include <stddef.h>

#ifndef __CUDACC__

#define __host__
#define __device__

#define IS_ON_HOST true
#define IS_ON_DEVICE false

#else

#include <cuda_runtime.h>

#ifdef __CUDA_ARCH__
#define IS_ON_HOST false
#define IS_ON_DEVICE true
#else
#define IS_ON_HOST true
#define IS_ON_DEVICE false
#endif

#endif

#endif // NEO_COMMON_H
