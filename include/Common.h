#ifndef _COMMON_H
#define _COMMON_H

#define __macro_min(a,b) ((a)<(b)?(a):(b))
#define __macro_max(a,b) ((a)>(b)?(a):(b))
#define __macro_bound(x, a, b) ((x) > (a) ? ((x) < (b) ? (x):(b)):(a))

#define __macro_TIMING_PREPARATION \
	LARGE_INTEGER begin, end, freq;\
	QueryPerformanceFrequency(&freq);\

#define __macro_TIME(EXPRESSION)\
	QueryPerformanceCounter(&begin);\
		EXPRESSION\
	QueryPerformanceCounter(&end);\
	printf("Time: %f\n", (float)(end.QuadPart - begin.QuadPart)/ (float) freq.QuadPart);\

#define _out_ 
#define _in_ 

enum LSH_Application {
	LSH_None, LSH_Contraction,  LSH_Alpha, LSH_Selection, LSH_Clustering, LSH_Seeding
};
struct Communicator {
	const char* filename;
	float **f_streamlines = 0;
	float *results = 0;
	int *colors = 0;
	void *AdditionalParas = 0;
	int n_streamlines = -1;
	int *sizes = 0;
	float *alpha = 0;
	int n_points;
	LSH_Application application = LSH_None;
	float lsh_radius = 0;
};

template<typename T>
inline T constexpr cubic(const T v) noexcept { return v*v*v; }
template<typename T>
#ifdef __CUDA_ARCH__
__device__ __host__
#endif
__forceinline T constexpr square(const T v) noexcept { return v * v; }
#include <random>
extern std::mt19937_64 engine;
extern float gaussianDist(float x);
#endif
