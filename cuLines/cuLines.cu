#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <stdint.h>
#include "cuStubs.cuh"
#include "Vector.h"


template<typename T>
__device__
inline T constexpr pow2(const T v) noexcept { return v * v; }

__global__
void cuLSH(

	const GPU_Lsh_Func *funcs, const GPU_HashTable *hashtable, const Table_Contents *table_contents, 
	const GPU_Segments *segments, int *__restrict__ results,  float* temp, 
	const float *f_stramlines, const int* lineoffsets, const int n_lines, const int n_pts

) {
	temp += (blockIdx.x * blockDim.x + threadIdx.x)*n_lines;
	
	bool sgn = false;
	for (int i = blockIdx.x; i < n_lines; i += gridDim.x) {
		const float* streamline = f_stramlines + lineoffsets[i];
		for (int j = threadIdx.x; j < lineoffsets[i + 1] - lineoffsets[i]; j += blockDim.x) {
			int cnt_result = 0;
			int ptnum = lineoffsets[i] + j;
			for (int t = 0; t < TABLESIZE; t++)
			{
				int64_t fingerprint1 = 0, fingerprint2 = 0;
				for (int f = 0; f < K; f++) {

					const GPU_Lsh_Func curr_func = funcs[hashtable[t].LSHFuncs[f]];
					const int n_buckets = curr_func.n_buckets;
					const int func_val = curr_func(streamline + j*3);
					int64_t tmp_fp1 = hashtable[t].r1[f] * func_val;
					int64_t tmp_fp2 = hashtable[t].r1[f] * func_val;
					tmp_fp1 = 5 * (tmp_fp1 >> 32) + (tmp_fp1 & 0xffffffff);
					tmp_fp2 = 5 * (tmp_fp2 >> 32) + (tmp_fp2 & 0xffffffff);

					fingerprint1 += (tmp_fp1 >> 32) ? (tmp_fp1 - Prime) : tmp_fp1;
					fingerprint2 += (tmp_fp2 >> 32) ? (tmp_fp2 - Prime) : tmp_fp2;

					fingerprint1 = (fingerprint1 >> 32) ? (fingerprint1 - Prime) : fingerprint1;
					fingerprint2 = (fingerprint2 >> 32) ? (fingerprint2 - Prime) : fingerprint2;
				}
				fingerprint1 %= TABLESIZE;
				fingerprint2 %= Prime;
				
				int k = hashtable[t].table_offsets[fingerprint1];
				for (; k < hashtable[t].table_offsets[fingerprint1 + 1]; k++) {
					if (table_contents[k].fingerprint2 == fingerprint2) //optimize: group tables accroding to fingerprint2;
					{
						const int segment = table_contents[k].segment;
						const int line = segments[segment].line;
						if(temp[line] > 0 && sgn)
							results[ptnum*maxNN + cnt_result++] = segment;
						else {
							const float dist = temp[line];
							float this_dist = 0;
#pragma unroll
							for (int _dim = 0; _dim < 3; _dim++)
								this_dist += pow2(segments[segment].centroid[_dim] - streamline[j * 3 + _dim]);
							if (this_dist > dist) {
								const int res_pos = *reinterpret_cast<int*>(temp + line) & 0x1f;
								int *i_dist = reinterpret_cast<int*> (&this_dist);
								*i_dist = *i_dist & 0xffffffe0 + cnt_result;
								temp[line] = sgn ? -this_dist : this_dist;
								results[ptnum*maxNN + res_pos] = segment;
							}
						}
					}
					if (cnt_result >= maxNN) {
						goto finalize;
					}
				}
			}
		finalize:
			if (cnt_result < maxNN)
				results[ptnum*maxNN + cnt_result] = -1;
			sgn = !sgn;
		}

	}
}


__global__
void cuLineHashing(

	int *__restrict__ results,
	const GPU_SegmentsLv2 *segs_lv2, const float *f_stremlines, const int* lineoffsets,
	const short* lv2_buckets, const int n_lines, const int n_pts

) {

	int i = threadIdx.x;
	for (; i < n_lines; i+= blockDim.x) {
		int j = blockIdx.x;
		for (; j < lineoffsets[i + 1]; j += gridDim.x) {
			
			int ptnum = lineoffsets[i] + j;

			for (int k = 0; k < maxNN; k++) 
				if (results[ptnum*maxNN + k] > 0)
				{
					const int this_seg = results[ptnum * maxNN + k];
					const int ptoffset = segs_lv2[this_seg].bucket_pos_offset;
					float projection = 0;
#pragma unroll 
					for (int _dim = 0; _dim < 3; _dim++) 
						projection += 
							(f_stremlines[ptnum * 3 + _dim] - segs_lv2[this_seg].origin[_dim]) * segs_lv2[this_seg].projector[_dim];
					
					int bucket = std::floor(projection);
					if (projection < 0)
						bucket = 0;
					else if (projection > segs_lv2[this_seg].width - 1)
						bucket = segs_lv2[this_seg].width - 1;

					results[ptnum * maxNN + k] = ptoffset + lv2_buckets[segs_lv2[this_seg].bucket_pos_offset] + bucket;
				}
				else
					break;


		}
	}
}


__global__
void cuSimilarity(
	_out_ float *variation, _out_ float* distances, _out_ int*points,
	_in_ const float *f_streamlines, _in_ const int * lineoffsets, _in_ const int* results,
	const int n_lines, const int n_pts
) {

	int i = threadIdx.x;
	for (; i < n_lines; i += blockDim.x) {
		int j = blockIdx.x;
		for (; j < lineoffsets[i + 1]; j += gridDim.x) {
			
		}
	}

}

void cudaLauncher();

namespace cudadevice_variables {
	extern GPU_SegmentsLv2* segslv2;  //Lv2. hash projector 
	extern GPU_Segments* segs;//centroid + line No. for Lv1. LSH
	extern float* l2buckets;
	extern GPU_HashTable *d_hash;
	extern GPU_Lsh_Func *d_funcs;
}
