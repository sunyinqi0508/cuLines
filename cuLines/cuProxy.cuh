#ifndef _H_CUPROXY
#define _H_CUPROXY

#include "cuProxy.h"
#ifdef __INTELLISENSE__
#define __CUDACC__
#endif
#include "cuda_runtime.h"
#include <device_atomic_functions.h>
#include "device_functions.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include "curand_kernel.h"

#ifdef __INTELLISENSE__
#undef __global__
#undef __device__
#undef __host__
#undef __inline__
#undef __forceinline__
#undef __restrict__
#undef __shared__

#define __global__
#define __device__
#define __host__
#define __inline__
#define __forceinline__
#define __restrict__
#define __shared__
#endif
#endif
