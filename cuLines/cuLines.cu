#include <stdint.h>
#include "cuProxy.cuh"
#include "Vector.h"


template<typename T>
__device__
inline T constexpr pow2(const T v) noexcept { return v * v; }

template<class _Ty>
__device__ __host__ __forceinline__
int binarySearch(const _Ty* __restrict__ orderedList, int lowerbound, int upperbound, const _Ty& key) {
	while (upperbound > lowerbound) {
		int mid = (lowerbound + upperbound) >> 1;
		if (mid == lowerbound) {
			return orderedList[mid] == key ? lowerbound : -lowerbound;
		}
		else {
			if (orderedList[mid] > key)
				upperbound = mid;
			else if (orderedList[mid] < key)
				lowerbound = mid;
			else
				return mid;
		}
	}
	return orderedList[lowerbound] == key ? lowerbound : -lowerbound;
}
__global__
void update_memorypool(int* current_memoryblock, int* current_offset, const int poolsize) {
	*current_memoryblock++;
	*current_offset -= poolsize;
}
__device__ __forceinline__
bool constraints1(const Vector3& point) {

	return true;
}
__device__ __forceinline__
bool constraints2(const Vector3& point) {

	return true;
}
__global__
void cuLSH( 
	/*lv.1 search*/
	const GPU_Lsh_Func *funcs, const GPU_HashTable *hashtable, const int64_t *table_contents, const int* segments_in_table,
	const GPU_Segments *segments, int* temp, const Vector3 *f_streamlines, const int* lineoffsets, const int n_lines,
	const int n_pts, const int set_size, const int set_associative,

	/*lv.2 search*/
	const GPU_SegmentsLv2 *segs_lv2, const short* lv2_buckets,

	/*memory_pool*/
	int** memory_pool, int64_t* memory_pool_index, int* current_memoryblock, int* current_offset, int* _idxs, const int poolsize,
	int* finished

) {
	extern __shared__ unsigned char sh_tmp[];
	const int tmp_block_size = (set_associative - 1) * set_size;
	int _idx = blockIdx.x;
	temp += (_idx) * tmp_block_size;

	int i = _idxs[_idx] ? _idxs[_idx] : _idx;

	constexpr int get_tag = 0xf;
	bool overflew = 0;
	//bool sgn = false;
	for (; i < n_pts; i += gridDim.x) {
			int cnt_result = 0;
			//lv.1 search
#pragma region lv.1 search
			for (int t = 0; t < L; t++)
			{
				int64_t fingerprint1 = 0, fingerprint2 = 0;
				for (int f = 0; f < K; f++) {

					const GPU_Lsh_Func curr_func = funcs[hashtable[t].LSHFuncs[f]];
					const int n_buckets = curr_func.n_buckets;
					const int func_val = curr_func(f_streamlines[i]);
					int64_t tmp_fp1 = hashtable[t].r1[f] * func_val;
					int64_t tmp_fp2 = hashtable[t].r1[f] * func_val;
					tmp_fp1 = 5 * (tmp_fp1 >> 32ll) + (tmp_fp1 & 0xffffffffll);
					tmp_fp2 = 5 * (tmp_fp2 >> 32ll) + (tmp_fp2 & 0xffffffffll);

					fingerprint1 += (tmp_fp1 >> 32ll) ? (tmp_fp1 - Prime) : tmp_fp1;
					fingerprint2 += (tmp_fp2 >> 32ll) ? (tmp_fp2 - Prime) : tmp_fp2;

					fingerprint1 = (fingerprint1 >> 32ll) ? (fingerprint1 - Prime) : fingerprint1;
					fingerprint2 = (fingerprint2 >> 32ll) ? (fingerprint2 - Prime) : fingerprint2;
				}
				fingerprint1 %= TABLESIZE;
				fingerprint2 %= Prime;
				
				const int table_search_begin = hashtable[t].table_offsets[fingerprint1],
					table_search_end = hashtable[t].table_offsets[fingerprint1 + 1];
				int found = binarySearch(table_contents, table_search_begin, table_search_end, fingerprint2);
				if (found > 0) {
					const unsigned line =  segments[found].line;
					const float dist = segments[found].centroid.sqDist(f_streamlines[i]);

					if (dist < 1.f && constraints1(segments[found].centroid))
					{
						const int position = line / set_associative;
						const int tag = line % set_associative;
						const int current_set = (temp[position] &0x7fffffff) >> 27;
						constexpr int set_increment = 1 << 27;



						if (current_set < set_associative)//max of 16 way set-associative; availible slots
						{
							bool exists = false;
							for (int j = 0; j < current_set; j++) {
								const int this_segment = ((temp[position + j * set_size] & 0x07ffffff) >> 4);
								if (temp[position + j * set_size] & get_tag == tag) {
									if (dist < segments[this_segment].centroid.sqDist(f_streamlines[i]))
									{
										temp[position + j * set_size] &= 0xf800000f;
										temp[position + j * set_size] |= (found << 4);
										exists = true;
										break;
									}
								}
							}

							if (!exists) {
								temp[position] += set_increment;// total_sets ++
								temp[position + (current_set + 1) * set_size] = found << 4 | tag;
							}

						}

					}
				}


		}
#pragma endregion

#pragma region lv.2 search
			for (int j = 0; j < tmp_block_size; ++j) {

				if (temp[j] != 0x80000000) {

					const int this_tag = temp[j] & get_tag;
					const int this_segment = ((temp[j] & 0x07ffffff) >> 4);
					const int this_line = j << 4 + this_tag;
					const GPU_SegmentsLv2& this_seglv2 = segs_lv2[this_segment];
					const float projection = (f_streamlines[i] - this_seglv2.origin).project(this_seglv2.projector) / this_seglv2.width;
					int key = projection;
					if (key < 0)
						key = 0;
					else if (key > this_seglv2.length)
						key = this_seglv2.length;

					key += this_seglv2.bucket_pos_offset;
					const int nearest_point = lv2_buckets[key];
					if (!constraints2(f_streamlines[lineoffsets[this_line] + nearest_point]))
						temp[j] = 0x80000000;
					else
						++cnt_result;

				} 

			}
#pragma endregion

#pragma region storing
			int this_offset = atomicAdd(current_offset, cnt_result);
			int this_memoryblock = *current_memoryblock;
			int this_end = this_offset + cnt_result;
			int curr_end, this_count = 0;
			memory_pool_index[i] =
				(this_memoryblock << 54ll) | (this_offset << 27ll) | cnt_result;
			if (this_end > poolsize) {
				if (this_offset > poolsize)
				{
					this_offset -= poolsize;
					curr_end = this_end - poolsize;
					++this_memoryblock;
					overflew = true;
				}
				else {
					curr_end = poolsize;
					overflew = true;
				}
			}
			else
				curr_end = this_end;
			for (int j = 0; j < tmp_block_size; ++j) {

				if (temp[j] != 0x80000000) {
					++this_count;

					memory_pool[this_memoryblock][this_offset++] = temp[j];

					if (this_offset >= curr_end && overflew) {
						if (this_count >= cnt_result) 
							break;

						this_count = 0;
						curr_end = this_end - poolsize;
						++this_memoryblock;
					}


					temp[j] = 0x80000000;
				}

			}

			if (overflew)
				break;
#pragma endregion

	}
	_idxs[_idx] = i;
	atomicAnd(finished, i >= n_pts);
}

__device__
unsigned int findRange(const int* orderedList, int lowerbound, int upperbound, const int key) {
	int mid;

	while (lowerbound + 1 < upperbound) {

		mid = (lowerbound + upperbound) >> 1;
		if (orderedList[mid] < key)
			lowerbound = mid;
		else if (orderedList[mid] > key)
			upperbound = mid;
		else
			break;

	}
	if (orderedList[mid] != key)
		return 0xffffffff;
	int upe = mid, lowe = mid;
	while (lowerbound < lowe - 1) {
		mid = (lowerbound + lowe) >> 1;
		if (orderedList[mid] < key)
			lowerbound = mid;
		else
			lowe = mid;
	}
	while (upperbound > upe + 1) {
		mid = (upperbound + upe) >> 1;
		if (orderedList[mid] > key)
			upperbound = mid;
		else
			upe = mid;
	}
	return lowerbound | ((upperbound - lowerbound)<<20);
}
__global__
void cuLSH_lv1(
	/*lv.1 search*/
	const GPU_Lsh_Func *funcs, const GPU_HashTable *hashtable, const int64_t *table_contents, const unsigned int* segments_in_table,
	const GPU_Segments *segments, int* temp, const Vector3 *f_streamlines, const int* lineoffsets, const int n_lines,
	const int n_pts, const int n_segments, unsigned int** projections

) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	for (; i < n_pts; i += gridDim.x * blockDim.x) {
		for (int t = 0; t < L; t++)
		{
			int64_t fingerprint1 = 0, fingerprint2 = 0;
			for (int f = 0; f < K; f++) {

				const GPU_Lsh_Func curr_func = funcs[hashtable[t].LSHFuncs[f]];
				const int n_buckets = curr_func.n_buckets;
				const int func_val = curr_func(f_streamlines[i]);
				int64_t tmp_fp1 = hashtable[t].r1[f] * func_val;
				int64_t tmp_fp2 = hashtable[t].r1[f] * func_val;
				tmp_fp1 = tmp_fp1 % TABLESIZE;
				tmp_fp2 = tmp_fp2 % Prime;

				fingerprint1 += tmp_fp1;
				fingerprint2 += tmp_fp2;

				fingerprint1 %= TABLESIZE;
				fingerprint2 %= Prime;
			}
			fingerprint1 %= TABLESIZE;
			fingerprint2 %= Prime;

			const int table_search_begin = hashtable[t].table_offsets[fingerprint1],
				table_search_end = hashtable[t].table_offsets[fingerprint1 + 1];
			int found = binarySearch(table_contents + t*n_segments, table_search_begin, table_search_end, fingerprint2);
			if (found == -1)
				projections[t][i] = -1;
			else
				projections[t][i] = segments_in_table[found];//Segments that has the same fingerprints (1&2)
		}

	}
}


__global__
void cuLSH_lv2(
	/*environments*/
	const GPU_Segments *segments, unsigned char* temp, const Vector3 *f_streamlines, const int* lineoffsets, const int n_lines,
	const int n_pts, const int set_size, const int set_associative, const unsigned int** projections,

	/*lv.2 search*/
	const GPU_SegmentsLv2 *segs_lv2, const short* lv2_buckets,

	/*memory_pool*/
	int** memory_pool, int64_t* memory_pool_index, int* current_memoryblock, int* current_offset, int* _idxs, const int poolsize,
	int* finished
	
) {
	extern __shared__ unsigned char sh_tmp[];
	unsigned char* ptr_tmp = sh_tmp;
	const int tmp_block_size = (set_associative - 1) * set_size;
	int _idx = blockIdx.x * blockDim.x + threadIdx.x;
	temp += (_idx)* tmp_block_size;

	int i = _idxs[_idx] ? _idxs[_idx] : _idx;
	constexpr int get_tag = 0xf;
	const unsigned int cache_page_size = 384;//49512 (bytes per block) /128 (blocks) /1 (bytes per segment)

	auto get_cache = [&temp, &ptr_tmp, &cache_page_size](const int _Index) -> unsigned char&/*constexpr*/
		{	return (_Index < cache_page_size) ? ptr_tmp[_Index] : (temp)[_Index - cache_page_size];  };
	bool overflew = 0;
	//bool sgn = false;
	for (; i < n_pts; i += gridDim.x*blockDim.x) {
		int cnt_result = 0;
		
#pragma region lv.2 search
		for (int j = 0; j < tmp_block_size; ++j) {

			if (temp[j] != 0x80000000) {

				const int this_tag = temp[j] & get_tag;
				const int this_segment = ((temp[j] & 0x07ffffff) >> 4);
				const int this_line = j << 4 + this_tag;
				const GPU_SegmentsLv2& this_seglv2 = segs_lv2[this_segment];
				const float projection = (f_streamlines[i] - this_seglv2.origin).project(this_seglv2.projector) / this_seglv2.width;
				int key = projection;
				if (key < 0)
					key = 0;
				else if (key > this_seglv2.length)
					key = this_seglv2.length;

				key += this_seglv2.bucket_pos_offset;
				const int nearest_point = lv2_buckets[key];
				if (!constraints2(f_streamlines[lineoffsets[this_line] + nearest_point]))
					temp[j] = 0x80000000;
				else
				{
					++cnt_result;
				}
			}

		}
#pragma endregion

#pragma region storage
		int this_offset = atomicAdd(current_offset, cnt_result);
		int this_memoryblock = *current_memoryblock;
		int this_end = this_offset + cnt_result;
		int curr_end, this_count = 0;
		memory_pool_index[i] =
			(this_memoryblock << 54ll) | (this_offset << 27ll) | cnt_result;
		if (this_end > poolsize) {
			if (this_offset > poolsize)
			{
				this_offset -= poolsize;
				curr_end = this_end - poolsize;
				++this_memoryblock;
				overflew = true;
			}
			else {
				curr_end = poolsize;
				overflew = true;
			}
		}
		else
			curr_end = this_end;
		for (int j = 0; j < tmp_block_size; ++j) {

			if (temp[j] != 0x80000000) {
				++this_count;

				memory_pool[this_memoryblock][this_offset++] = temp[j];

				if (this_offset >= curr_end && overflew) {
					if (this_count >= cnt_result)
						break;

					this_count = 0;
					curr_end = this_end - poolsize;
					++this_memoryblock;
				}


				temp[j] = 0x80000000;
			}

		}

		if (overflew)
			break;
#pragma endregion

	}
	_idxs[_idx] = i;
	atomicAnd(finished, i >= n_pts);
}

__global__
void parallelized_memory_allocation_test(

	int** memory_pool, int64_t* memory_pool_index, int* current_memoryblock, int* current_offset, int n,
	int* _idxs, const int poolsize, curandStateMRG32k3a_t *state, int* finished

) {

	const int _idx = threadIdx.x + blockIdx.x * blockDim.x;
	int idx = _idxs[_idx] ? _idxs[_idx] : _idx;//resume last work
	bool overflew = 0;
	for (; idx < n; idx += blockDim.x * gridDim.x) {

		const int result_count = curand_uniform(state) * 32.f;
		int this_offset = atomicAdd(current_offset, result_count);
		int this_memoryblock = *current_memoryblock;
		int this_end = this_offset + result_count;

		memory_pool_index[idx] =
			(this_memoryblock << 54ll) | (this_offset << 27ll) | result_count;

		if (this_end > poolsize)
		{
			for (; this_offset < poolsize; ++this_offset) {
				memory_pool[this_memoryblock][this_offset] = idx * 10000 + this_offset;
			}
			this_offset = 0;
			this_end -= poolsize;
			++this_memoryblock;
			overflew = true;
		}
		for (; this_offset < this_end; ++this_offset) {
			memory_pool[this_memoryblock][this_offset] = idx * 10000 + this_offset;
		}
		if (overflew)
			break;
	}
	_idxs[_idx] = idx;
	atomicAnd(finished, idx >= n);
}


//__global__
//void cuLineHashing(//Lv.2 search
//
//	int *__restrict__ results, int n_results,
//	const GPU_SegmentsLv2 *segs_lv2, const float *f_stremlines, const int* lineoffsets,
//	const short* lv2_buckets, const int n_lines, const int n_pts
//
//) {
//
//	int i = threadIdx.x;
//	for (; i < n_lines; i+= blockDim.x) {
//		int j = blockIdx.x;
//		for (; j < lineoffsets[i + 1]; j += gridDim.x) {
//			
//			int ptnum = lineoffsets[i] + j;
//
//			for (int k = 0; k < maxNN; k++) 
//				if (results[ptnum*maxNN + k] > 0)
//				{
//					const int this_seg = results[ptnum * maxNN + k];
//					const int ptoffset = segs_lv2[this_seg].bucket_pos_offset;
//					const int bucket_begin = segs_lv2[this_seg].bucket_pos_offset;
//					float projection = 0;
//#pragma unroll 
//					for (int _dim = 0; _dim < 3; _dim++) 
//						projection += 
//							(f_stremlines[ptnum * 3 + _dim] - (segs_lv2[this_seg].origin)[_dim]) * segs_lv2[this_seg].projector[_dim];
//					
//					int bucket = std::floor(projection);
//					if (projection < 0)
//						bucket = 0;
//					else if (projection > segs_lv2[this_seg].width - 1)
//						bucket = segs_lv2[this_seg].width - 1;
//
//					results[ptnum * maxNN + k] = segs_lv2[this_seg].line << 16 + (ptoffset + lv2_buckets[bucket_begin + bucket]);//n_lines < 65535 && pt_on_line < 65535
//				}
//				else
//					break;
//
//
//		}
//	}
//}


__global__
void cuLineHashing_mp(
	int ** memory_pool
) {

}
__global__ 
void cuHeapify(

	_in_ _out_ float *variation, _in_ _out_ float*distances, _in_ _out_ float *points,
	const int n_lines, const int n_pts

) {



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
			const int ptnum = lineoffsets[i] + j;
			for (int k = 0; k < maxNN; k++)
				if (results[ptnum*maxNN + k] != -1) {

					const unsigned int targetline = ((unsigned)results[ptnum*maxNN + k]) >> 16;
					const unsigned int targetpt_on_line = ((unsigned)results[ptnum*maxNN + k] & 0xffff);
					const unsigned int target_ptnum = lineoffsets[targetline] + targetpt_on_line;
					
					int begin = lineoffsets[i] + (j > similarity_window) ? (j - similarity_window) : 0;
					int end = lineoffsets[i] + j + similarity_window;
					end = (end >= lineoffsets[i + 1]) ? lineoffsets[i + 1] - 1 : end;

					int forward = ptnum - begin;
					int backward = end - ptnum;
					   
					forward = __macro_min(targetpt_on_line, forward);
					backward = __macro_min(lineoffsets[targetline + 1] - lineoffsets[targetline] - targetpt_on_line - 1, backward);

					float center_dist = 0;

#pragma unroll 
					for (int _dim = 0; _dim < 3; _dim++)
						center_dist += pow2(f_streamlines[ptnum*3 + _dim] - f_streamlines[target_ptnum*3 + _dim]);
					center_dist = sqrtf(center_dist);


					float _variation = 0;
					int start_this = ptnum - forward, start_target = target_ptnum - forward;
					for (; start_this < ptnum; start_this++, start_target++) {
						float _dist = 0;
#pragma unroll 
						for (int _dim = 0; _dim < 3; _dim++) 
							_dist += pow2(f_streamlines[start_this * 3 + _dim] - f_streamlines[start_target * 3 + _dim]);
						_variation += pow2(center_dist - sqrtf(_dist));
					}

					for (; start_this < ptnum + backward; start_this++, start_target++) {
						float _dist = 0;
#pragma unroll 
						for (int _dim = 0; _dim < 3; _dim++)
							_dist += pow2(f_streamlines[start_this * 3 + _dim] - f_streamlines[start_target * 3 + _dim]);
						_variation += pow2(center_dist - sqrtf(_dist));
					}

					const int interval = backward + forward - 1;
					if (interval > 0)
						_variation /= interval;
					else
						_variation = 0;

					distances[ptnum*maxNN + k] = center_dist;
					variation[ptnum * maxNN + k] = _variation;
				}

				else break;

		}
	}

}



namespace cudadevice_variables {

	GPU_SegmentsLv2* segslv2;  //Lv.2 hash projector 
	GPU_Segments* segs;//centroid + line No. for Lv.1 LSH
	float* l2buckets;
	GPU_HashTable *d_hash;
	GPU_Lsh_Func *d_funcs; 

	float* d_streamlines; // FileIO:: f_streamlines[0]
	int* d_lineoffsets;// Streamline::sizes;


	//Memory pool
	int *d_cof,//current offset
		*d_cmb,
		*d_fin,
		*d_mp,
		*d_tmp;
	int64_t* d_idxs;
	const int poolsize = 134217728;//128 Words = 512MB 
	const int tmp_size = 4096;
	const int set_associative = 2; // 2 pre allocated sets of tmp pages, 1 dynamically allocated tmp
	//scale variables
	int n_streamlines, n_points, n_segments;
}

//---------------------------------------------------------------------\\
--------------------- NVCC Compiled Host functions. ---------------------


void cudaInit(

	Vector3 *f_streamlines, int* lineoffsets, GPU_Segments* segments,
	GPU_Lsh_Func *lsh_funcs, GPU_HashTable *hash_table,
	GPU_SegmentsLv2 *segslv2, float* l2buckets

	)
{
}

void cudaLaunch() {

}

void cudaFinalize() {

}
