#ifndef _CU_STUBS
#define _CU_STUBS
#include <stdint.h>
#include "Parameters.h"
#include "Common.h"
#include <device_launch_parameters.h>

__device__ __host__ 
struct GPU_Lsh_Func {
	float _a[3];
	float _b, _w;
	int n_buckets;
	
	__device__ __host__ 
	constexpr int operator()(const float* v) const {
		return static_cast<int>(
			__macro_bound((v[0] * _a[0] + v[1] * _a[1] + v[2] * _a[2] + _b) / _w, 0, n_buckets)
			);
	}

};

__device__ __host__
struct GPU_HashTable {
	
	int64_t r1[K];
	int64_t r2[K];
	int LSHFuncs[K];
	int table_offsets[TABLESIZE + 1];

};

__device__ __host__
struct Table_Contents {
	int64_t fingerprint2;
	int segment;
};

__device__ __host__
struct GPU_Segments {

	float centroid[3];
	int line;

};

__device__ __host__ 
struct GPU_SegmentsLv2 {

	int point_num_offset, bucket_pos_offset;
	int line;
	int length; 

	float origin[3], projector[3];
	float width;

};

namespace cudadevice_variables {
	extern GPU_SegmentsLv2* segslv2;  //Lv.2 hash projector 
	extern GPU_Segments* segs;//centroid + line No. for Lv.1 LSH
	extern float* l2buckets;
	extern GPU_HashTable *d_hash;
	extern GPU_Lsh_Func *d_funcs;

	extern float* d_streamlines; // FileIO:: f_streamlines[0]
	extern int* d_lineoffsets;// Streamline::sizes;
}

#endif