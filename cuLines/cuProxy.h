#ifndef _CU_STUBS
#define _CU_STUBS

#include <stdint.h>
#include "Vector.h"
#include "Parameters.h"
#include "Common.h"

struct GPU_Lsh_Func {
	Vector3 _a;
	float _b, _w;
	float offset;
	int n_buckets;
#ifdef __CUDACC__
	__device__ 
#endif
	int operator()(const Vector3& v) const {
		const float res = (v.dot(_a) + _b) / _w - offset;
		return static_cast<int>(
			__macro_bound(res, 0, n_buckets - 1)
			);
	}

};

struct GPU_HashTable {
	
	int64_t r1[K];
	int64_t r2[K];
	int LSHFuncs[K];
	int table_offsets[TABLESIZE + 1];

};

struct Table_Contents {
	int64_t fingerprint2;
	int segment;
};

struct GPU_Segments {

	Vector3 centroid;
	int line;

};

struct GPU_SegmentsLv2 {

	int point_num_offset, bucket_pos_offset;
	int line;
	int length; 

	Vector3 origin, projector;
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


void cudaInit();

#endif
