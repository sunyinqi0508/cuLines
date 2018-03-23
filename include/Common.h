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

struct Communicator {
	const char* filename;
	float **f_streamlines;
	float *results;
	void *AdditionalParas;
	int n_streamlines;
	int *sizes;
	int n_points;
};

template<typename T>
inline T constexpr cubic(const T v) noexcept { return v*v*v; }

#include <random>
extern std::mt19937_64 engine;

#endif