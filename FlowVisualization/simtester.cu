// CUDA runtime �� + CUBLAS ��  
#define HALF_ENABLE_CPP11_CMATH

#include "cuda_runtime.h"  
#include "cublas_v2.h"  
#include<device_launch_parameters.h>
#include<thrust/host_vector.h>
#include<thrust/device_vector.h>

#include "SimTester.h"
#include <iostream>
#ifdef  __INTELLISENSE__
#define __CUDACC__
#include <device_functions.h>
#undef __constant__
#undef __global__
#undef __shared__
#undef __device__
#define __global__
#define __shared__
#define __device__
#define __constant__
#endif //  __INTELLISENSE__
//#include <cuda_fp16.h>
//#include "float.hpp"


//cuda ����
 
#define BUCKET_NUM 100 
#define STACK_SPACE 96
#define SAVED_TREES 128
#define HEAP_SIZE 64
#define LEAF_OFFSET 32
#define RIGHT_MASK 0
#define LEFT_MASK 1
#define BOTH_MASK 2
#define NEITHER_MASK 3
#define HALF_SAMPLE 8/*distance based->curvature based*/
#define CROP_LAST6 0xffc0
#define GET_LAST6 0x3f
#undef max
#undef min


//#define r2 dslf*dslf

extern 	size_t pitch;
__constant__ float d_hash[49];
__constant__ float epsilon[1];


void constant_cpy(float *linearhash, float _epsilon) {
	cudaMemcpyToSymbol(d_hash, linearhash, 48 * sizeof(float));
	cudaMemcpyToSymbol(epsilon, &_epsilon, sizeof(float));
}
//template <bool add = false>
//change heap to unsigned int;
__global__ void cuDeterminingAvaillines(int *avail, float *val, int n)
{
	int i = threadIdx.x + blockIdx.x *blockDim.x;
	if (i < n) {
		val += i*HEAP_SIZE;
		if (*val > 0)
		{
			int j;
			for (j = 0; j < 64; j++)
				if (val[j] <= 0)
					break;
			avail[i] = j;
		}
		else
			avail[i] = 0;
	}
}
__global__ void  CudaSimilarWithLines(

	int n, //n of pts

	int *lineinfo, float *h_all,  //line information

	int *buckets, float *segments, //LSH

	int *heap, float *val, float *variation, //outputs

	int *searched, //massive tmp

	int *lkd, int*id, //Kdtree

	bool *pt_availablility

)
{
	int pidx = blockIdx.x*blockDim.x + threadIdx.x;
	if (pidx >= n)
		return;
	if (!pt_availablility[pidx])
		return;
	int *lineinfo_ptr = lineinfo + (pidx << 1);

	if ((pidx - *lineinfo_ptr) < HALF_SAMPLE || *(lineinfo_ptr + 1) - pidx + *lineinfo_ptr - 1 < HALF_SAMPLE)
		return;



	float d_tmp;
	float  *si, *siSampled;
	float *sjSampled;


	short tmp[8];
	val += pidx*HEAP_SIZE - 1;
	heap += pidx*HEAP_SIZE - 1;
	variation += pidx * HEAP_SIZE - 1;

	searched += pidx*SAVED_TREES;
	si = h_all + 3 * *lineinfo_ptr;

	siSampled = si + 3 * ((pidx - *lineinfo_ptr) - HALF_SAMPLE);
	int l[32];
	unsigned int stack[16];
	int *leaf = l - 1;
	float pt[3];//= { siSampled[3 * HALF_SAMPLE] ,siSampled[3 * HALF_SAMPLE + 1] ,siSampled[3 * HALF_SAMPLE + 2] };
	pt[0] = siSampled[15];
	pt[1] = siSampled[16];//siSampled[3 * HALF_SAMPLE];// siSampled[3 * HALF_SAMPLE + 1];
	pt[2] = siSampled[17];//siSampled[3 * HALF_SAMPLE];// siSampled[3 * HALF_SAMPLE + 2];


	int i_tmp = 0;
#pragma unroll
	for (int i = 0; i < 8; i++, i_tmp += 6)
	{
		d_tmp = pt[0] * d_hash[i_tmp];
		d_tmp += pt[1] * d_hash[i_tmp + 1];
		d_tmp += pt[2] * d_hash[i_tmp + 2];
		d_tmp += d_hash[i_tmp + 3];
		d_tmp -= d_hash[i_tmp + 4];
		d_tmp /= d_hash[i_tmp + 5];
		tmp[i] = d_tmp;
		if (tmp[i] < 0)
			tmp[i] = 0;
		else if (tmp[i] >= BUCKET_NUM)
			tmp[i] = BUCKET_NUM - 1;

	}
	int size = 0;
	bool dir = true;
	int dither = 0;
	float x, y, z, min, a, v_tmp;
	int ptr = 1, lfptr = 1;
	int lob = 0, rob = 0, toi;
	int index, end, idx, i_tmp2;
	while (size < STACK_SPACE && dither < BUCKET_NUM) //infinite loop when size > remaining segmemts;
	{
#pragma region LSH
		toi = 1;
#pragma unroll
		for (int i = 0; i < 8; i++)
		{
			//int i = 0;
			toi <<= 1;
			index = tmp[i] + dir ? dither : -dither;
			if (index < 0)
			{
				if ((lob&toi) && !dir)
					goto finalize;

				index = 0;
				lob |= toi;
			}
			else if (index >= BUCKET_NUM)
			{
				if ((rob&toi) && dir)
					goto finalize;

				index = BUCKET_NUM - 1;
				rob |= toi;
			}//dithering


			index += 100 * i;
			end = buckets[index + 1];
			index = buckets[index];
			if (index < 0)
				goto finalize;// blank bucket - attention needed on linearization
		found:			while (index < end)
		{
			if (buckets[index] < 0)
			{
				index += buckets[index + 1] + 2;
				goto found;
			}
			for (int j = 0; j < size; j++)
				if (buckets[index] == searched[j])
				{
					index += buckets[index + 1] + 2;
					goto found;
				}
			searched[size++] = buckets[index];
			if (buckets[index + 1] > 1)
			{
				min = INT_MAX;
				i_tmp = index + buckets[index + 1] + 2;
				for (int j = index + 2; j < i_tmp; j++)
				{
					i_tmp2 = buckets[j] * 7;
					x = pt[0] - segments[i_tmp2];
					y = pt[1] - segments[i_tmp2 + 1];
					z = pt[2] - segments[i_tmp2 + 2];

					d_tmp = 0;
					d_tmp += x*segments[i_tmp2 + 3];
					d_tmp = 0;
					d_tmp += y*segments[i_tmp2 + 4];
					d_tmp = 0;
					d_tmp += z*segments[i_tmp2 + 5];

					d_tmp /= segments[i_tmp2 + 6];//padding 
					if (d_tmp <= 0)
						d_tmp = x*x + y*y + z*z;
					else if (d_tmp >= 1)
						d_tmp = pow(x + segments[i_tmp2], 2) + pow(y + segments[i_tmp2 + 1], 2) + pow(z + segments[i_tmp2 + 2], 2);//d_tmp = segments[j * 8 + 7];
					else
						d_tmp = pow(x + d_tmp*segments[i_tmp2], 2) + pow(y + d_tmp*segments[i_tmp2 + 1], 2) + pow(z + d_tmp*segments[i_tmp2 + 2], 2);
					if (d_tmp < min)
					{
						min = d_tmp;
						idx = buckets[j];
					}
				}
			}
			else//blank bucket?
				idx = buckets[index + 2];
#pragma endregion
#pragma region KD-Tree

			int *linearKD = lkd + id[idx];


			int next = 0;
			int dim = 1, stackidx = 0;
			float currdata;
			float dss = INT_MAX;
			float ds;
			int ptidx = -1;

			while (next != -1) {//using mask to reduce memory usage 


				currdata = pt[dim];
				ds = pow(pt[0] - linearKD[next + 1], 2) + pow(pt[1] - linearKD[next + 2], 2)
					+ pow(pt[2] - linearKD[next + 3], 2);
				if (ds < dss)
				{
					dss = ds;
					ptidx = linearKD[next];
				}
				if (linearKD[next + dim] < currdata)
				{
					if (linearKD[next + 4] != -1 && linearKD[next + 11] != -1)
						stack[stackidx++] = (linearKD[next + 11] << 2) + (linearKD[linearKD[next + 11] + 4] < 0 ?
							NEITHER_MASK : linearKD[linearKD[next + 11] + 11] < 0 ? LEFT_MASK : BOTH_MASK);
					next = linearKD[next + 4];
				}
				else
				{
					if (linearKD[next + 4] != -1)
					{
						stack[stackidx++] = (linearKD[next + 4] << 2) + (linearKD[linearKD[next + 4] + 4] < 0 ?
							NEITHER_MASK : linearKD[linearKD[next + 4] + 11] < 0 ? LEFT_MASK : BOTH_MASK);
						next = linearKD[next + 11];
					}
					else
						break;
				}

				dim = (dim++ - 3) ? dim : dim - 3;
			}
			/*faster implementation:
			1: no pruning;
			2: calc the minimum dist while getting into the point;
			*/


			//backtrace;
			int  r;
			int rt;

		ctn:	while (stackidx > 0) {

			rt = stack[--stackidx];
			r = rt&NEITHER_MASK;
			rt >>= 2;
			ds = pow(pt[0] - linearKD[rt + 1], 2) + pow(pt[1] - linearKD[rt + 2], 2)
				+ pow(pt[2] - linearKD[rt + 3], 2);
			if (ds < dss)
			{
				dss = ds;
				ptidx = linearKD[rt];
			}
			ds = 0;
			switch (r)
			{
			case 0: rt = rt + 11;
				break;
			case 1: rt = rt + 4;
				break;
			case 3:continue;
			default:
				rt = rt + 4;
				break;
			}
			if (pt[0] < linearKD[rt + 1])
				ds += pow(pt[0] - linearKD[rt + 1], 2);
			else if (pt[0] > linearKD[rt + 4])
				ds += pow(pt[0] - linearKD[rt + 4], 2);
			if (pt[1] < linearKD[rt + 2])
				ds += pow(pt[1] - linearKD[rt + 2], 2);
			else if (pt[1] > linearKD[rt + 5])
				ds += pow(pt[1] - linearKD[rt + 5], 2);
			if (pt[2] < linearKD[rt + 3])
				ds += pow(pt[2] - linearKD[rt + 3], 2);
			else if (pt[2] > linearKD[rt + 6])
				ds += pow(pt[2] - linearKD[rt + 6], 2);
			if (ds < dss&&linearKD[rt]>0)
			{
				stack[stackidx] = linearKD[rt] << 2;
				stack[stackidx++] += linearKD[linearKD[rt] + 4] < 0 ? NEITHER_MASK : linearKD[linearKD[rt] + 11] < 0 ? LEFT_MASK : BOTH_MASK;//BOTH_MASK;

			}
			if (r == 2)
			{
				rt = rt + 7;
				ds = 0;
				if (pt[0] < linearKD[rt + 1])
					ds += pow(pt[0] - linearKD[rt + 1], 2);
				else if (pt[0] > linearKD[rt + 4])
					ds += pow(pt[0] - linearKD[rt + 4], 2);
				if (pt[1] < linearKD[rt + 2])
					ds += pow(pt[1] - linearKD[rt + 2], 2);
				else if (pt[1] > linearKD[rt + 5])
					ds += pow(pt[1] - linearKD[rt + 5], 2);
				if (pt[2] < linearKD[rt + 3])
					ds += pow(pt[2] - linearKD[rt + 3], 2);
				else if (pt[2] > linearKD[rt + 6])
					ds += pow(pt[2] - linearKD[rt + 6], 2);
				if (ds < dss)
				{
					stack[stackidx] = linearKD[rt] << 2;
					stack[stackidx++] += linearKD[linearKD[rt] + 4] < 0 ? NEITHER_MASK : linearKD[linearKD[rt] + 11] < 0 ? LEFT_MASK : BOTH_MASK;
				}
			}
		}

				dss = sqrt(dss);

#pragma endregion
#pragma region AdditionalCalc
				//int t = buckets[index];

				if (ptidx < HALF_SAMPLE || lineinfo[buckets[ptidx + 2] + 1] - ptidx - 1 < HALF_SAMPLE)
				{
					index += buckets[index + 1] + 2;
					continue;
				}
				else
					sjSampled = h_all + (3 * lineinfo[buckets[ptidx + 2]]) + 3 * (ptidx - HALF_SAMPLE);

				a = 0;
				//float *sis = pt - 3 * HALF_SAMPLE;
				//#pragma unroll
				for (int j = 0; j < 2 * HALF_SAMPLE + 1; j++)
					a += sqrt(pow(siSampled[3 * j] - sjSampled[3 * j] - (pt[0] - sjSampled[15]), 2)
						+ pow(siSampled[3 * j + 1] - sjSampled[3 * j + 1] - (pt[1] - sjSampled[16]), 2)
						+ pow(siSampled[3 * j + 2] - sjSampled[3 * j + 2] - (pt[2] - sjSampled[17]), 2));
#pragma endregion
#pragma region Heap
				//heap op
				//*(int*)&val[1]&CROP_LAST6;
				int  j = 0, t;


				if (ptr > HEAP_SIZE)//offset
				{

					if (val[leaf[1]] <= dss)
					{
						index += buckets[index + 1] + 2;
						continue;
					}
					j = leaf[1];
					t = j >> 1;
					d_tmp = val[t];
					i_tmp = heap[t];
					v_tmp = variation[t];
					while (j > 1 && d_tmp > dss)
					{
						val[j] = d_tmp;
						heap[j] = i_tmp;
						variation[j] = v_tmp;
						j = t;
						t >>= 1;
						d_tmp = val[t];
						i_tmp = heap[t];
						v_tmp = variation[t];
					}
					val[j] = dss;
					heap[j] = (buckets[index]);// << 18) + idx;
					variation[j] = 100 * a;
					//leaf-heap operation
					i_tmp2 = leaf[1];
					j = 2;
					i_tmp = val[leaf[2]] > val[leaf[3]] ? leaf[2] : leaf[++j];
					while (val[i_tmp] > dss)
					{
						leaf[j >> 1] = i_tmp;
						if ((j <<= 1) >= LEAF_OFFSET)
							break;
						i_tmp = val[leaf[j]] > val[leaf[j + 1]] ? leaf[j] : leaf[++j];

					}
					leaf[j >> 1] = i_tmp2;
					//end leaf-heap op
				}
				else {

					j = ptr++;


					t = j >> 1;

					d_tmp = val[t];
					i_tmp = heap[t];
					v_tmp = variation[t];
					while (j > 1 && d_tmp > dss)
					{
						heap[j] = i_tmp;
						val[j] = d_tmp;
						variation[j] = v_tmp;
						j = t;
						t >>= 1;
						d_tmp = val[t];
						i_tmp = heap[t];
						v_tmp = variation[t];
					}

					val[j] = dss;
					heap[j] = (buckets[index]);// << 18) + idx;//seg
					variation[j] = 100 * a;
					//leaf_op
					if (ptr > LEAF_OFFSET)
					{
						j = lfptr++;
						dss = val[ptr - 1];
						i_tmp = leaf[j >> 1];

						while (j > 1 && val[i_tmp] < dss)
						{
							leaf[j] = i_tmp;
							j >>= 1;
							i_tmp = leaf[j >> 1];
						}
						leaf[j] = ptr - 1;
					}
					//end leaf_op
				}
#pragma endregion
				//return heap-as-{size}-nearst-pts;
				index += buckets[index + 1] + 2;
				if (size > STACK_SPACE)
					return;


		}

					finalize:
						if (!dir)
							dither++;
						dir = !dir;
		}
	}
	if (ptr <= HEAP_SIZE)
		val[ptr] = -1;

}


union reused_t {
	float    fp;
	uint32_t uint;
};


#define THREADS_PER_BLOCK 128
#define LSH_SEARCH_BUFFER 128 /*__test__:2 */


#define MODULO_8_CROPPER 0x7
#define LSB_EXTRACTOR 0x1 /*extract lsb from binary number by a logic and*/
#define UCHAR_MAX 0xff

//Macro functions
#define pt(i) shared[blockDim.x *(i) + threadIdx.x]
#define stack(i) i_shared[blockDim.x *((i) + 3) + threadIdx.x]
#define check(i) ((searched[n * ((i)>>3) + ptidx])>>((i)&MODULO_8_CROPPER))&LSB_EXTRACTOR
#define mark(i) (searched[n * ((i)>>3) + ptidx]) &= (unsigned char)(UCHAR_MAX - (1<<((i)&MODULO_8_CROPPER)))
#define availibility_check(i) (ptavail[((i)>>3)]>>((i)&MODULO_8_CROPPER))&LSB_EXTRACTOR/*12 I32Add-equivalents for bit calculation*/
#define ispt(i) (i)<0
#define linearKD(i)  __uint_as_float(linearKD[(i)])


__global__
void LSH_determining_bucket(

	//unsigned char* searched, int *streamlineoffsets,

	uchar4 *bucketsforpt, float *ptinfo, int *output, int n

)
{

	int ptidx = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned char tmp[8]; //max of 256 buckets;
	uchar4 *parts = reinterpret_cast<uchar4 *> (tmp);

	float pt[3];
	if (ptidx < n) {
		
#pragma unroll
		for (int i = 0; i < 3; i++)
			pt[i] = ptinfo[n * i + ptidx];
		pt[0] = fabs(pt[0]);
		float d_tmp;//float
		int i_tmp = 0;
#pragma unroll
		for (int i = 0; i < 8; i++, i_tmp += 6)
		{
			d_tmp = pt[0] * d_hash[i_tmp];
			d_tmp += pt[1] * d_hash[i_tmp + 1];
			d_tmp += pt[2] * d_hash[i_tmp + 2];
			d_tmp += d_hash[i_tmp + 3];
			d_tmp -= d_hash[i_tmp + 4];
			d_tmp /= d_hash[i_tmp + 5];

			d_tmp = d_tmp < 100 ? d_tmp : 100;
			tmp[i] =(unsigned char) d_tmp > 0 ? __float2uint_rn(d_tmp) : 0;
			//printf("%d %d %f \n", __float2uint_rn(d_tmp), (unsigned int)(d_tmp), d_tmp);

		}
		bucketsforpt[ptidx] = parts[0];
		bucketsforpt[n + ptidx] = parts[1];
#pragma unroll
		for (int i = 0; i < LSH_SEARCH_BUFFER; i++)
			output[n*i + ptidx] = INT32_MIN;
	}
}//preproc;


__global__ 
void LSHSearch( //`output` need to be zeroed before first use;

	unsigned char *searched, unsigned char *ptavail,

	int *buckets, float *segments, float *ptinfo,

	uchar4 *bucketsforpt, int *output,

	int n, int slotsize

) {

	int ptidx = blockIdx.x*blockDim.x + threadIdx.x;
	if (ptidx < n) {
		__shared__ float shared[THREADS_PER_BLOCK * 3];
		unsigned char tmp[8]; //max of 256 buckets;
#pragma unroll
		for (int i = 0; i < 3; i++)
			pt(i) = ptinfo[ptidx + i * n];
		pt(0) = fabs(pt(0));
		if (ptidx < n && availibility_check(ptidx)) {

			int size = 0;

			float x, y, z, min, a, v_tmp;
			int ptr = 1, lfptr = 1;
			//int lob = 0, rob = 0, toi;
			int index, end, idx, i_tmp2, i_tmp;
			float d_tmp;

			reinterpret_cast<uchar4 *> (tmp)[0] = bucketsforpt[ptidx];
			reinterpret_cast<uchar4 *> (tmp)[1] = bucketsforpt[ptidx + n];
			int dither = 0;
			bool dir = true;
			int size1 = -1;
			int size2 = -2;
			while (size < LSH_SEARCH_BUFFER && dither <= BUCKET_NUM) {

//#pragma unroll
				for (int i = 0; i < 8; i++) //infinite loop when size > remaining segmemts;
				{


					index = tmp[i] + (dir ? dither : -dither);
					if (index < 0)
					{
						continue;
					}

					else if (index >= BUCKET_NUM)
					{
						continue;
					}//dithering


					index += 100 * i;
					end = buckets[index + 1];
					index = buckets[index];
					if (index < 0)
						continue;// blank bucket - attention needed on linearization

					while (index < end)
					{
						if (buckets[index] < 0 || !(check(buckets[index])))
						{
							index += buckets[index + 1] + 2;
							continue;
						}

						mark(buckets[index]);
						idx = -1;
						if (buckets[index + 1] > 1)
						{
							min = INT_MAX;
							i_tmp = index + buckets[index + 1] + 2;
							for (int j = index + 2; j < i_tmp; j++)
							{
								i_tmp2 = buckets[j] * 7;
								x = pt(0) - segments[i_tmp2];
								y = pt(1) - segments[i_tmp2 + 1];
								z = pt(2) - segments[i_tmp2 + 2];

								d_tmp = 0;
								d_tmp += x*segments[i_tmp2 + 3];
								d_tmp = 0;
								d_tmp += y*segments[i_tmp2 + 4];
								d_tmp = 0;
								d_tmp += z*segments[i_tmp2 + 5];

								d_tmp /= segments[i_tmp2 + 6];//padding 
								if (d_tmp <= 0)
									d_tmp = x*x + y*y + z*z;
								else if (d_tmp >= 1)
									d_tmp = pow(x + segments[i_tmp2], 2) + pow(y + segments[i_tmp2 + 1], 2) + pow(z + segments[i_tmp2 + 2], 2);//d_tmp = segments[j * 8 + 7];
								else
									d_tmp = pow(x + d_tmp*segments[i_tmp2], 2) + pow(y + d_tmp*segments[i_tmp2 + 1], 2) + pow(z + d_tmp*segments[i_tmp2 + 2], 2);
								if (d_tmp < min)
								{
									min = d_tmp;
									idx = buckets[j];
								}
							}
						}
						else//blank bucket?
							idx = buckets[index + 2];
						while (
							ispt(output[size*n + ptidx])
							&& output[size*n + ptidx] > INT32_MIN
							&& availibility_check(-output[size*n + ptidx])
							) {
							size++;
						}
						if (idx == -1)
						{
							index += buckets[index + 1] + 2;
							continue;
						}
						output[size++ * n + ptidx] = (idx << 13) + buckets[index];//H19 seg L13 line
						//index += 
						index += buckets[index + 1] + 2;
						if (size >= LSH_SEARCH_BUFFER)
							return;
					}

				}
			finalize:
				if (dir)
					dither++;
				dir = !dir;
			}
			if (size < LSH_SEARCH_BUFFER)
				output[size *n + ptidx] = INT32_MIN;
		}
	}
}

__global__ 
void KDSearch(

	int *lkd, int*id,

	int *outputs, unsigned short *variation,

	float *lineinfo, unsigned char* ptavail, //ignore negative values in lineinfo.x(s) 

	int n

) {
	int kernelIdx = threadIdx.x + blockIdx.x * blockDim.x;
	if (kernelIdx < n && availibility_check(kernelIdx))
	{
		__shared__ float shared[16 * THREADS_PER_BLOCK];
		int *i_shared = (int *)shared;
		for (int i = 0; i < 3; i++)
			pt(i) = lineinfo[kernelIdx + i * n];
		pt(0) = fabs(pt(0));
		//pt(3) = pt(1) * pt(1) + pt(0) * pt(0) + pt(2) * pt(2);
		bool finished = false;
		for (int i = 0; i < LSH_SEARCH_BUFFER; i++) { 
			float dss = INT_MAX;
			int ptidx = -1;
			int opti = outputs[n * i + kernelIdx];
			if (ispt(opti) || finished)//point or segment?
			{
				return;
				if (opti <= INT32_MIN)
					continue;
				else
					opti = -opti;

				if (availibility_check(opti))
				{
					dss = sqrt(pow(pt(0) - fabs(lineinfo[opti]), 2)
						+ pow(pt(1) - lineinfo[opti + n], 2)
						+ pow(pt(2) - lineinfo[opti + 2 * n], 2));
				}
				else
				{
					finished = true;
					continue;
				}
			}
			else {
				int *linearKD = lkd + id[(opti >> 13)];//kd-search for segment idx
				int next = 0, dim = 1, stackidx = 0;
				int rt = 3;
				float currdata, ds;

				while (next != -1) {//using mask to reduce memory usage 

					currdata = pt((dim - 1));
					ds = pow(pt(0) - linearKD(next + 1), 2) + pow(pt(1) - linearKD(next + 2), 2)
						+ pow(pt(2) - linearKD(next + 3), 2);
					if (ds < dss)
					{
						dss = ds;
						ptidx = linearKD[next];
					}


					if (linearKD(next + dim) < currdata)
					{
						if (linearKD[next + 4] != -1 && linearKD[next + 11] != -1)
						{
							stack(stackidx) = (linearKD[next + 11] << 2) + (linearKD[linearKD[next + 11] + 4] == -1 ?
								NEITHER_MASK : linearKD[linearKD[next + 11] + 11] == -1 ? LEFT_MASK : BOTH_MASK);//lv 2 opt
							stackidx++;
						}
						next = linearKD[next + 4];
					}
					else
					{
						if (linearKD[next + 4] != -1)
						{
							stack(stackidx) = (linearKD[next + 4] << 2) + (linearKD[linearKD[next + 4] + 4] == -1 ?
								NEITHER_MASK : linearKD[linearKD[next + 4] + 11] == -1 ? LEFT_MASK : BOTH_MASK);
							stackidx++;
							next = linearKD[next + 11];
						}
						else
							break;
					}
					dim = (dim++ - 3) ? dim : dim - 3;
				}
				/*better implementation:
				Half precision for boxing. (16 bytes aligning, 1~2 g.mem fetches)
				0  1 2 3  4 5   5   6   6   7   7    8 9   9   10  10  11  11
				id x y z (l lmx lmy lmz lMx lMy lMz (r rmx rmy rmz rMx rMy rMz))
				*/
				int  r;// backtrace
			ctn:
				while (stackidx > 0) {
					stackidx--;
					rt = stack(stackidx);

					r = rt&NEITHER_MASK;
					rt >>= 2;
					ds = pow(pt(0) - linearKD[rt + 1], 2) + pow(pt(1) - linearKD[rt + 2], 2)
						+ pow(pt(2) - linearKD[rt + 3], 2);
					if (ds < dss)
					{
						dss = ds;
						ptidx = linearKD[rt];
					}
					ds = 0;
					switch (r)
					{
					case 0: rt = rt + 11;
						break;
					case 1: rt = rt + 4;
						break;
					case 3:continue;
					default:
						rt = rt + 4;
						break;
					}


					/*r = linearKD[rt] & 0x3;//rt&NEITHER_MASK;
					//rt >>= 2;
					ds = pow(pt(0) - linearKD(rt + 1), 2) + pow(pt(1) - linearKD(rt + 2), 2)
					+ pow(pt(2) - linearKD(rt + 3), 2);
					if (ds < dss)
					{
					dss = ds;
					ptidx = linearKD[rt];
					}
					ds = 0;
					switch (r)
					{
					case 0: continue;// rt = rt + 11;
					break;
					case 1: rt = rt + 4;
					break;
					case 3:
					rt = rt + 4;
					break;// continue;
					default:
					printf("error!");
					continue;
					rt = rt + 4;
					break;
					}*/
					if (pt(0) < linearKD(rt + 1))
						ds += pow(pt(0) - linearKD(rt + 1), 2);
					else if (pt(0) > linearKD(rt + 4))
						ds += pow(pt(0) - linearKD(rt + 4), 2);
					if (pt(1) < linearKD(rt + 2))
						ds += pow(pt(1) - linearKD(rt + 2), 2);
					else if (pt(1) > linearKD(rt + 5))
						ds += pow(pt(1) - linearKD(rt + 5), 2);
					if (pt(2) < linearKD(rt + 3))
						ds += pow(pt(2) - linearKD(rt + 3), 2);
					else if (pt(2) > linearKD(rt + 6))
						ds += pow(pt(2) - linearKD(rt + 6), 2);
					if (ds < dss&&linearKD[rt]>0)
					{
						stack(stackidx) = linearKD[rt] << 2;
						stack(stackidx) += linearKD[linearKD[rt] + 4] == -1 ? NEITHER_MASK : linearKD[linearKD[rt] + 11] == -1 ? LEFT_MASK : BOTH_MASK;//BOTH_MASK;
						stackidx++;
					}
					if (r == 3)
					{
						rt = rt + 7;
						ds = 0;
						if (pt(0) < linearKD(rt + 1))
							ds += pow(pt(0) - linearKD(rt + 1), 2);
						else if (pt(0) > linearKD(rt + 4))
							ds += pow(pt(0) - linearKD(rt + 4), 2);
						if (pt(1) < linearKD(rt + 2))
							ds += pow(pt(1) - linearKD(rt + 2), 2);
						else if (pt(1) > linearKD(rt + 5))
							ds += pow(pt(1) - linearKD(rt + 5), 2);
						if (pt(2) < linearKD(rt + 3))
							ds += pow(pt(2) - linearKD(rt + 3), 2);
						else if (pt(2) > linearKD(rt + 6))
							ds += pow(pt(2) - linearKD(rt + 6), 2);
						if (ds < dss)
						{
							stack(stackidx) = linearKD[rt] << 2;
							stack(stackidx) += linearKD[linearKD[rt] + 4] == -1 ? NEITHER_MASK : linearKD[linearKD[rt] + 11] == -1 ? LEFT_MASK : BOTH_MASK;
							stackidx++;
						}
					}

				}
				outputs[n * i + kernelIdx] = -(ptidx);//>>2
				variation[kernelIdx * LSH_SEARCH_BUFFER + i] = opti & 0x1fff;//get L13, line
			}
#pragma endregion
		}
	}

}
//1.8.23->1.5.10/6.10
__device__ float& hf2float(const short& hf) {
	int sf = ((0x8000 & hf) << 16) + ((0x7c00 & hf) << 13) + ((0x03ff & hf) << 13);
	return *(float*)&sf;
}
__device__ float& uhf2float(const short& uhf) {
	int sf = ((0xfc00 & uhf) << 13) + ((0x03ff & uhf) << 13);
	return *(float*)&sf;
}
union _4bit{
	float f;
	int i;
};
__global__
void VectorizedHashing(

	int *linearsegs, short* offsets,

	int *outputs, unsigned short *variation,

	float *linfo, /* linfo optimized for random access */
	
	unsigned char* ptavail, //ignore negative values in lineinfo.x(s) 

	int n

) {
	int kernelIdx = threadIdx.x + blockIdx.x * blockDim.x;
	if (kernelIdx < n && availibility_check(kernelIdx))
	{
		__shared__ float shared[16 * THREADS_PER_BLOCK];
		int *i_shared = (int *)shared;
		for (int i = 0; i < 3; i++)
			pt(i) = linfo[kernelIdx * 3 + i];
		pt(0) = fabs(pt(0));
		//pt(3) = pt(1) * pt(1) + pt(0) * pt(0) + pt(2) * pt(2);
		bool finished = false;
		for (int i = 0; i < LSH_SEARCH_BUFFER; i++) {
			float dss = INT_MAX;
			int ptidx = -1;
			int opti = outputs[n * i + kernelIdx];
			if (ispt(opti) || finished)//point or segment?
			{
				return;
				if (opti <= INT32_MIN)
					continue;
				else
					opti = -opti;

				if (availibility_check(opti))
				{
					dss = sqrt(pow(pt(0) - fabs(linfo[opti *3]), 2)
						+ pow(pt(1) - linfo[opti * 3 + 1], 2)
						+ pow(pt(2) - linfo[opti *3+ 2], 2));
				}
				else
				{
					finished = true;
					continue;
				}
			}
			else {

				int *vecs = linearsegs + ((opti >> 13)*10); //(opti >> 13 <<3)
				float *fvecs = (float *)vecs;

				int base = vecs[6];

				float length = 0;
				int bucket;
				/*	Record format:
				*	0	1	2	3	4	5	6	7 
				* base vec  -   -  fp step len offset 
				*   Pipeline:
				*	bucket = ((pt - [base])*vec - fp)/step
 				*/
#pragma unroll
				for (int i = 0; i < 3; i++) 
					length += (pt(i) - fvecs[i]) * fvecs[i + 3];

				if (length < fvecs[7])
					ptidx = base;
				else {
					short* this_offsets = offsets + vecs[9];
					unsigned short n_bucket = *(unsigned short*)this_offsets;
					int bucket = (length - fvecs[7]) / fvecs[8];
				
					if (bucket >= n_bucket)
						ptidx = base + this_offsets[this_offsets[n_bucket] + n_bucket];
					else
					{
					
						const int bias = ((float)(this_offsets[bucket + 1] - (bucket == 0 ? 0 : this_offsets[bucket])))
							*(length - bucket * fvecs[8]) / fvecs[8];
						
						ptidx = base + this_offsets[(bucket == 0?0:this_offsets[bucket]) + n_bucket + 1 + bias];
					
					}
				}
				outputs[n * i + kernelIdx] = -(ptidx);//>>2
				variation[kernelIdx * LSH_SEARCH_BUFFER + i] = opti & 0x1fff;//get L13, line
			}
#pragma endregion
		}
	}

}


__global__ 
void CoupledHeapsFiltration(

	float *lineinfo, unsigned char *ptavail,

	int *heap_, float *val_, float *variation_,

	int *outputs, int n

) {//max lsh_search_buffer = 255

	int kernelIdx = threadIdx.x + blockDim.x * blockIdx.x;
	if (kernelIdx < n  && availibility_check(kernelIdx)) {

		__shared__ unsigned char shared[32 * THREADS_PER_BLOCK];// 32 bytes per thread 2K bytes/block
		int *heap = heap_ + kernelIdx * HEAP_SIZE - 1;
		float *val = val_ + kernelIdx * HEAP_SIZE - 1;
		float *variation = variation_ + kernelIdx * HEAP_SIZE - 1;
		unsigned short *lines = reinterpret_cast<unsigned short*>(variation_);
		unsigned char *leaf = shared + threadIdx.x * 32 - 1;
		int ptr = 1, j, t, i_tmp, i_tmp2, lfptr = 1;
		float pt[3];
		float d_tmp;
#pragma unroll
		for (int i = 0; i < 3; i++)
		{
			pt[i] = lineinfo[kernelIdx + i * n];
		}
		pt[0] = fabs(pt[0]);

//#pragma unroll
		for (int i = 0; i < LSH_SEARCH_BUFFER; i++)
		{
			int currpt = outputs[i * n + kernelIdx];
			if (currpt < 0)
			{
				if (currpt <= INT32_MIN)
					continue;
				float dss = 0;
				currpt = -currpt;
#pragma unroll
				for (int k = 0; k < 3; k++)
					dss += pow(pt[k] - fabs(lineinfo[currpt + k * n]), 2);
				if (kernelIdx == currpt)
					continue;
				dss = sqrt(dss);

				if (ptr > HEAP_SIZE)//offset
				{

					if (val[leaf[1]] <= dss)
						continue;
					j = leaf[1];
					t = j >> 1;
					while (j > 1)
					{
						d_tmp = val[t];
						i_tmp = heap[t];
						if (d_tmp <= dss)
							break;
						val[j] = d_tmp;
						heap[j] = i_tmp;
						j = t;
						t >>= 1;

					}
					val[j] = dss;
					heap[j] = (((unsigned)lines[kernelIdx * LSH_SEARCH_BUFFER + i]) << 8) + i;// << 18) + idx;
																							  //leaf-heap operation
					i_tmp2 = leaf[1];
					j = 2;
					i_tmp = val[leaf[2]] > val[leaf[3]] ? leaf[2] : leaf[++j];
					while (val[i_tmp] > dss)
					{
						leaf[j >> 1] = i_tmp;
						if ((j <<= 1) >= LEAF_OFFSET)
							break;
						i_tmp = val[leaf[j]] > val[leaf[j + 1]] ? leaf[j] : leaf[++j];

					}
					leaf[j >> 1] = i_tmp2;
					//end leaf-heap op
				}
				else {

					j = ptr++;
					t = j >> 1;

					while (j > 1)
					{
						d_tmp = val[t];
						i_tmp = heap[t];
						if (d_tmp <= dss)
							break;
						heap[j] = i_tmp;
						val[j] = d_tmp;
						j = t;
						t >>= 1;
					}

					val[j] = dss;
					heap[j] = (((unsigned)(lines[kernelIdx * LSH_SEARCH_BUFFER + i])) << 8) + i;// << 18) + idx;//seg
																								//leaf_op
					if (ptr > LEAF_OFFSET + 1)
					{
						j = lfptr++;
						dss = val[ptr - 1];
						//i_tmp = j>1?leaf[j >> 1]:0;

						while (j > 1)
						{
							i_tmp = leaf[j >> 1];
							if (val[i_tmp] >= dss)
								break;
							leaf[j] = i_tmp;
							j >>= 1;
						}
						//if (j <= 32)
						leaf[j] = ptr - 1;
					}
					//end leaf_op
				}
			}
		}
		if (ptr <= HEAP_SIZE&&ptr >= 1)
			val[ptr] = heap[ptr] = -1;
		ptr--;
#pragma region variation
		//bool signi = pt[0] > 0, signj;	
		int starti = 0, endi = 0;

		while (starti < HALF_SAMPLE + 1 && (kernelIdx - starti > 0) && lineinfo[kernelIdx - starti] > 0)
			starti++;
		while (endi < HALF_SAMPLE + 1 && kernelIdx + endi<n&& lineinfo[kernelIdx + endi] > 0)
			endi++;
//#pragma unroll
		for (int i = 1; i <= ptr; i++) {
			int currptj = -outputs[n*(heap[i] & 0xff) + kernelIdx];
			heap[i] >>= 8;//corrected
			float current_variation = 0;
			if (currptj < 0)
				break;
			int startj = 1, endj = 1;

			while (startj < starti
				&&currptj - startj >0 && lineinfo[currptj - startj] > 0)
			{
				float di = 0;
				di += pow(fabs(lineinfo[kernelIdx - startj]) - fabs(lineinfo[currptj - startj]) - (pt[0] - fabs(lineinfo[currptj])), 2);
#pragma unroll
				for (int i = 1; i < 3; i++)
					di += pow(lineinfo[kernelIdx - startj + i *n] - lineinfo[currptj - startj + i * n] - (pt[i] - lineinfo[currptj + i * n]), 2);
				current_variation += sqrt(di);//pow(di, 2);
				startj++;
			}

			//endj = 1;
			while (endj < endi
				&& currptj + endj<n&& lineinfo[currptj + endj]>0)
			{
				float di = 0;
				di += pow(fabs(lineinfo[kernelIdx + endj]) - fabs(lineinfo[currptj + endj]) - (pt[0] - fabs(lineinfo[currptj])), 2);
#pragma unroll
				for (int i = 1; i < 3; i++)
					di += pow(lineinfo[kernelIdx + endj + i *n] - lineinfo[currptj + endj + i * n] - (pt[i] - lineinfo[currptj + i * n]), 2);
				//di = sqrt(di);
				//current_variation += sqrt(di);// pow(di - dist, 2);
				endj++;
			}
			/*if (endj + startj)
			current_variation /= (float)(endj + startj);
			else
			current_variation = 1;
			*/
			//current_variation += .1 * (2 * HALF_SAMPLE - (endj + startj) + 2) ;//__test__:1 短线补偿策略
			if(!(endj+startj))
				current_variation /= (float)(endj + startj);

			variation[i] = current_variation;// -val[i];// *(val[i] + 100);
		}
#pragma endregion

	}
}

__global__ void UpdateLsh(int *buckets, int target) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < 800) {
		int end = buckets[i + 1] - buckets[i];
		buckets += buckets[i];
		for (int j = 0; j < end;)
		{
			if (buckets[j] == target)
				buckets[j] = -buckets[j];

			j += buckets[j + 1] + 2;
		}
	}
}


__global__ void RollbackLsh(int *buckets) {
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	if (i < 800) {
		int end = buckets[i + 1];
		buckets += buckets[i];
		for (int j = 0; j < end; j++)
			if (buckets[j]<0)
				buckets[j] = -buckets[j];
	}
}

__global__ void Avg(float *similarity, int n, float *avg_back) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;//(2,64),256
	double sum = 0;
	__shared__ double reduced[32];
	int wid = threadIdx.y >> 5;
	int lane = threadIdx.y - (wid << 5);

	for (; i < n; i += blockDim.x * gridDim.x)
	{
		__syncthreads();

		sum = similarity[HEAP_SIZE * i + threadIdx.y];
		for (int j = 16; j >= 1; j >>= 1) {
			sum += __shfl_down(sum, j);
		}

		if (lane == 0) {
			reduced[wid + threadIdx.x * 2] = sum;
		}

		__syncthreads();

		if (threadIdx.y ==  0) {
			atomicAdd(avg_back, (reduced[threadIdx.x * 2] + reduced[threadIdx.x * 2 + 1]));
		}
	}

}


__global__ void saliency_1(float *similarity, float *distance, float *output, int n, float c = 3.f) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;//(2,64),256
	float sum = 0;
	__shared__ float reduced[2];
	int wid = threadIdx.y >> 5;
	int lane = threadIdx.y - (wid << 5);
	sum = 0;
	float sum_divisor = 0;
	for (; i < n; i += blockDim.x * gridDim.x)
	{
		sum = similarity[HEAP_SIZE*i+threadIdx.x]/((1.f+.5*distance[HEAP_SIZE*i + threadIdx.x]));
		for (int j = 16; j >= 1; j >>= 1) {
			sum += __shfl_down(sum, j);
		}
		if (lane == 0 && wid) {
			reduced[threadIdx.x * 2] = sum;
		}
		__syncthreads();
		if (threadIdx.y == 0) {
			sum += reduced[threadIdx.x * 2];
			if (isnan(sum) || isinf(sum))
				sum = 0;
			output[i] =  1 - exp(-sum / 64.);
		}
	}
	sum = 0;
}

#define _SUM_INV_X_2 1.62918636078388701094
__global__ void AlphaCalc(float *similarity, float *distance, float *output, int n, float avg, float alpha, float _min)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;//(2,64),256
	float sum = 0;
	float range = 1 - _min;
	__shared__ float reduced[4];
	__shared__ float reduced_divisor[4];
	int wid = threadIdx.y >> 5;
	int lane = threadIdx.y - (wid << 5);
	sum = 0;
	float sum_divisor = 0;
	for (; i < n; i += blockDim.x * gridDim.x)
	{
		if (avg != 0) {
			if (distance[HEAP_SIZE*i + threadIdx.y] == 0)
				sum_divisor = 0;
			else
				sum_divisor = 1 / (distance[HEAP_SIZE*i + threadIdx.y] *distance[HEAP_SIZE*i + threadIdx.y]);
			
			sum = 1 - avg*alpha*(pow(2.718281828f, -pow(similarity[HEAP_SIZE * i + threadIdx.y], 2.f) / 2.f) - _min) / range;
			//sum = sum > 0 ? sum : 0;

			sum *= sum_divisor;
			for (int j = 16; j >= 1; j >>= 1) {
				sum += __shfl_down(sum, j);
				sum_divisor += __shfl_down(sum_divisor, j);
			}

			if (lane == 0) {
				reduced[wid + threadIdx.x * 2] = sum;
				reduced_divisor[wid + threadIdx.x * 2] = sum_divisor;
			}
			__syncthreads();
			if (threadIdx.y == 0) {
				sum_divisor = reduced_divisor[threadIdx.x * 2] + reduced_divisor[threadIdx.x * 2 + 1];
				output[i] = (reduced[threadIdx.x * 2] + reduced[threadIdx.x * 2 + 1]) / sum_divisor;
			}
		}
	}
	sum = 0;
}

__global__ void cuMax(float *similarity, int n, unsigned int *max) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	//natural padding with width of 64
	float _max = 0;
	__shared__ float shared[32];
	int warp = threadIdx.x << 5;
	int lane = threadIdx.x - warp >> 5;

	for (; i < n * 64; i += blockDim.x*gridDim.x) {
		_max = similarity[i];
		for (int offset = 16; offset >= 1; offset >>= 1) {
			float tmp = __shfl_down(_max, offset);
			_max = tmp > _max ? tmp : _max;
		}
		if (lane == 0) {
			shared[warp] = _max;
		}
		__syncthreads();
		if (warp == 0) {
			_max = shared[lane];
			for (int offset = 16; offset >= 1; offset >>= 1) {
				float tmp = __shfl_down(_max, offset);
				_max = tmp > _max ? tmp : _max;
			}
			if (threadIdx.x == 0) {
				atomicMax(max, __float_as_uint(_max));
			}
		}
	}
}
void cavg(float *similarity, int n, float *avg_back, float *max = 0) {
	Avg << < 256, dim3(2, 64) >> >(similarity, n, avg_back);
	if (max) {
		cuMax << <32, 1024 >> > (similarity, n, (unsigned *)max);
	}
}
__global__
void simple_simlarity(float *output, float * variation, float * distances, int N) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	for (; i < N; i += blockDim.x * gridDim.x) {
		output[i] = variation[i*HEAP_SIZE];
	}
}
void cuda(float *similarity, float *distance, float *output, int n, float avg = -1, float alpha = -1, float min = -1) {
	//saliency_1 << <256, dim3(2, 64) >> >(similarity, distance, output, n, alpha);

	//AlphaCalc << <256, dim3(2, 64) >> >(similarity, distance, output, n, avg, alpha, min);
	simple_simlarity << <256, 256 >> > (output, similarity, distance, n);
}
__device__ int pos;
//double buffering 2-way
__global__ void deletion(float *val, float *variation, int *heap, unsigned char *avail, int *idx, int *p2seg, int n, int p, unsigned char* next_avail, int *next_idx) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i >= n)
		return;
	int pt = idx[i];//n of pt;
	if (p2seg[pt] == p)
		return;

	unsigned char availible = avail[i];
	val += pt * HEAP_SIZE;
	variation += pt*HEAP_SIZE;
	heap += pt*HEAP_SIZE;
	for (int i = 0; i <= availible; i++) {
		if (p2seg[heap[i]] == p) {
			while (p2seg[heap[--availible]] == p&&availible > 0);

			int k;
			if (heap[i] > heap[availible])
			{
				k = i;
				int   j = i << 1;

				while (heap[j] < heap[availible] && j<availible)
				{
					heap[j] = heap[k];//2b verified
					val[j] = val[k];
					variation[j] = variation[k];
					k = j;
					j <<= 1;
				}

				//up
			}
			else
			{
				k = i;
				int  j = i >> 1;

				while (heap[j] > heap[availible] && k>0)
				{
					heap[j] = heap[k];//2b verified
					val[j] = val[k];
					variation[j] = variation[k];
					k = j;
					j >>= 1;
				}
				//down
			}
			variation[k] = variation[availible];
			val[k] = val[availible];
			heap[k] = heap[availible];
		}
	}
	if (availible < 0)
		return;
	int next = atomicAdd(&pos, 1);
	next_idx[next] = pt;
	next_avail[next] = availible + 1;
}

__global__ void local_sim(float *sel_pts, int *othersegs, float *out, int *lkd, float *vari_data, int prob_n, int n) {
	int thread = threadIdx.x + blockIdx.x * blockDim.x;
	if (thread < prob_n)
	{
		static __shared__ float shared[16 * THREADS_PER_BLOCK];
		int *i_shared = (int *)shared;
		//float *stack = shared + threadIdx.x * 16;
		//	float *pt = shared + threadIdx.x * 16;
		//	float *stack = pt  + 4;
		for (int i = 0; i < 3; i++)
			pt(i) = sel_pts[thread * 3 + i];
		/*ushort3 upts;
		upts.x = (__float_as_uint(pt(0))>>12);
		*/

		pt(3) = pt(1) * pt(1) + pt(0) * pt(0) + pt(2) * pt(2);
		for (int i = 0; i < n; i++) {
			int *linearKD = lkd + othersegs[i];

			//kd-search for segment idx

			int next = 0;
			int dim = 1, stackidx = 0;
			float currdata;
			float dss = INT_MAX;
			float ds;

			int ptidx = -1;

			while (next != -1) {//using mask to reduce memory usage 


				currdata = pt(dim);
				ds = pow(pt(0) - linearKD[next + 1], 2) + pow(pt(1) - linearKD[next + 2], 2)
					+ pow(pt(2) - linearKD[next + 3], 2);
				if (ds < dss)
				{
					dss = ds;
					ptidx = linearKD[next];
				}
				if (linearKD[next + dim] < currdata)
				{
					if (linearKD[next + 4] != -1 && linearKD[next + 11] != -1)
						stack(stackidx++) = (linearKD[next + 11] << 2) + (linearKD[linearKD[next + 11] + 4] < 0 ?
							NEITHER_MASK : linearKD[linearKD[next + 11] + 11] < 0 ? LEFT_MASK : BOTH_MASK);
					next = linearKD[next + 4];
				}
				else
				{
					if (linearKD[next + 4] != -1)
					{
						stack(stackidx++) = (linearKD[next + 4] << 2) + (linearKD[linearKD[next + 4] + 4] < 0 ?
							NEITHER_MASK : linearKD[linearKD[next + 4] + 11] < 0 ? LEFT_MASK : BOTH_MASK);
						next = linearKD[next + 11];
					}
					else
						break;
				}

				dim = (dim++ - 3) ? dim : dim - 3;
			}
			/*faster implementation:
			1: no pruning;
			2: calc the minimum dist while getting into the point;
			*/


			//backtrace;
			int  r;
			int rt;

		ctn:	while (stackidx > 0) {

			rt = stack(--stackidx);
			r = rt&NEITHER_MASK;
			rt >>= 2;
			ds = pow(pt(0) - linearKD[rt + 1], 2) + pow(pt(1) - linearKD[rt + 2], 2)
				+ pow(pt(2) - linearKD[rt + 3], 2);
			if (ds < dss)
			{
				dss = ds;
				ptidx = linearKD[rt];
			}
			ds = 0;
			switch (r)
			{
			case 0: rt = rt + 11;
				break;
			case 1: rt = rt + 4;
				break;
			case 3:continue;
			default:
				rt = rt + 4;
				break;
			}
			if (pt(0) < linearKD[rt + 1])
				ds += pow(pt(0) - linearKD[rt + 1], 2);
			else if (pt(0) > linearKD[rt + 4])
				ds += pow(pt(0) - linearKD[rt + 4], 2);
			if (pt(1) < linearKD[rt + 2])
				ds += pow(pt(1) - linearKD[rt + 2], 2);
			else if (pt(1) > linearKD[rt + 5])
				ds += pow(pt(1) - linearKD[rt + 5], 2);
			if (pt(2) < linearKD[rt + 3])
				ds += pow(pt(2) - linearKD[rt + 3], 2);
			else if (pt(2) > linearKD[rt + 6])
				ds += pow(pt(2) - linearKD[rt + 6], 2);
			if (ds < dss&&linearKD[rt]>0)
			{
				stack(stackidx) = linearKD[rt] << 2;
				stack(stackidx++) += linearKD[linearKD[rt] + 4] < 0 ? NEITHER_MASK : linearKD[linearKD[rt] + 11] < 0 ? LEFT_MASK : BOTH_MASK;//BOTH_MASK;

			}
			if (r == 2)
			{
				rt = rt + 7;
				ds = 0;
				if (pt(0) < linearKD[rt + 1])
					ds += pow(pt(0) - linearKD[rt + 1], 2);
				else if (pt(0) > linearKD[rt + 4])
					ds += pow(pt(0) - linearKD[rt + 4], 2);
				if (pt(1) < linearKD[rt + 2])
					ds += pow(pt(1) - linearKD[rt + 2], 2);
				else if (pt(1) > linearKD[rt + 5])
					ds += pow(pt(1) - linearKD[rt + 5], 2);
				if (pt(2) < linearKD[rt + 3])
					ds += pow(pt(2) - linearKD[rt + 3], 2);
				else if (pt(2) > linearKD[rt + 6])
					ds += pow(pt(2) - linearKD[rt + 6], 2);
				if (ds < dss)
				{
					stack(stackidx) = linearKD[rt] << 2;
					stack(stackidx++) += linearKD[linearKD[rt] + 4] < 0 ? NEITHER_MASK : linearKD[linearKD[rt] + 11] < 0 ? LEFT_MASK : BOTH_MASK;
				}
			}
		}

				//dss = sqrt(dss);

#pragma endregion
#pragma region AdditionalCalc
				if (ptidx != -1)
				{
					dss += vari_data[ptidx * 4] - pt(0) * vari_data[ptidx * 4 + 1] - pt(1) * vari_data[ptidx * 4 + 2] - pt(2) * vari_data[ptidx * 4 + 3];
					atomicAdd(/*(unsigned int *)*/out + i, /*__float_as_uint(*/dss/*)*/);
				}
#pragma endregion
				//int t = buckets[index];
		}
	}
}

bool SimTester::isSimilarWithSelf(std::deque<vec3> &si, int siIdx)
{
	float lineLength = g_param.w;
	int   nHalfSample = g_param.nHalfSample;
	vector<vec3> siSampled;
	int          siPos;
	if (!sampleLine(si, siIdx, lineLength, nHalfSample, siSampled, siPos))
		return false;

	vec3 p = siSampled[nHalfSample];
	int lowID, highID;
	findIdxRange(si, p, g_param.dMin, lowID, highID);

	deque<vec3>& sj = si;

	vec3 q;

	int sjIdx = -1;


	float min_dist = FLT_MAX;


	for (int j = 0; j< sj.size(); j++)
	{
		if (j >= lowID && j <= highID)
			continue;
		float l = length(p - sj[j]);
		if (l < min_dist)
		{
			q = sj[j];
			min_dist = l;
			sjIdx = j;
		}
	}


	if (min_dist >= g_param.dSelfsep || sjIdx == -1)
		return false;




	// sample line
	vector<vec3>	sjSampled;
	int				sjPos;
	if (!sampleLine(sj, sjIdx, lineLength, nHalfSample, sjSampled, sjPos))
		return false;

	// enough points to compare
	float term1 = (siSampled[nHalfSample] - sjSampled[nHalfSample]).length();//min_dist;
	float term2 = 0.0f;
	for (int i = 0; i < siSampled.size(); ++i)
	{
		float a = length(siSampled[i] - sjSampled[i]);
		term2 += abs(a - term1);
	}


	float alpha = 5;

	term2 = alpha * term2 / siSampled.size();

	if ((term1 + term2) < g_param.dSelfsep)
		return true;

	return false;
}





bool SimTester::self_line_similarty(std::vector<vec3> &si_tmp, int id)
{
	vec3 p = si_tmp[id];
	vec3 q = si_tmp[0];
	int compare_id = -1;
	float min_dist = 100000;;
	for (int j = 0; j<si_tmp.size() - g_param.w / 2.0f; j++)
		if (min_dist > length(p - si_tmp[j]) && length(p - si_tmp[j]) > g_param.dMin)
		{
			min_dist = length(p - si_tmp[j]);
			q = si_tmp[j];
			compare_id = j;
		}

	if (compare_id == -1)
		return false;

	if (compare_id < g_param.w / 2 || compare_id > si_tmp.size() - g_param.w / 2)
		return false;

	std::vector<vec3> si;
	std::vector<vec3> sj;
	for (int i = id - g_param.w / 2.0f; i<id + g_param.w / 2.0f; i++)
		si.push_back(si_tmp[i]);

	for (int i = compare_id - g_param.w / 2.0f; i<compare_id + g_param.w / 2.0f; i++)
		sj.push_back(si_tmp[i]);

	float term1 = length(p - q);
	float term2 = 0.0f;
	float a;
	for (int k = 0; k<si.size(); k++)
	{
		a = length(si[k] - sj[k]);
		term2 += abs(a - term1);
	}
	term2 = g_param.alpha * term2 / si.size();
	if ((term1 + term2) > g_param.dSelfsep)
		return true;

	return false;
}



bool SimTester::sampleLine(const std::deque<vec3>& line, int idx,
	float lineLength, int nHalfSample,
	vector<vec3>& result, int& idxPos)
{
	if (idx<0 || idx >= line.size())
		return false;

	float segmentLength = lineLength / (nHalfSample * 2);

	vector<vec3> buffer[2];
	float totLength[2] = { 0, 0 };
	int   idxDir[2] = { 1, -1 };
	int   idxBound[2] = { line.size() - 1, 0 };

	for (int ithDir = 0; ithDir<2; ++ithDir)
	{
		buffer[ithDir].reserve(nHalfSample * 2 + 1);
		if (idx != idxBound[ithDir])
		{
			int thisIdx = idx, nextIdx = idx + idxDir[ithDir];
			vec3 curPnt = line[thisIdx];
			vec3 curDir = line[nextIdx] - curPnt;
			float allocateLength = curDir.length();
			curDir /= allocateLength;

			while (buffer[ithDir].size() < nHalfSample * 2 + 1)
			{
				if (totLength[ithDir] > allocateLength)
				{
					nextIdx += idxDir[ithDir];
					thisIdx += idxDir[ithDir];
					if (nextIdx >= line.size() || nextIdx < 0)
						break;

					vec3  delta = line[nextIdx] - line[thisIdx];
					float deltaLength = delta.length();
					float remainLength = totLength[ithDir] - allocateLength;
					allocateLength += deltaLength;
					curDir = delta / deltaLength;
					curPnt = line[thisIdx] + curDir * remainLength;
				}
				else
				{
					buffer[ithDir].push_back(curPnt);
					curPnt += curDir * segmentLength;
					totLength[ithDir] += segmentLength;
				}
			}
			totLength[ithDir] -= segmentLength;
		}
		else
			buffer[ithDir].push_back(line[idx]);
	}

	// line is too short
	if (buffer[0].size() + buffer[1].size() < nHalfSample * 2 + 2)
		return false;
	int nSample;
	int validData[2] = { nHalfSample, nHalfSample };
	for (int i = 0; i < 2; ++i)
	{
		nSample = buffer[i].size() - 1;
		if (nSample < nHalfSample)
		{
			validData[i] = nSample;
			validData[1 - i] += nHalfSample - nSample;
		}
	}

	result.clear();
	result.reserve(nHalfSample * 2 + 1);
	for (int i = validData[1]; i > 0; i--)
		result.push_back(buffer[1][i]);
	idxPos = result.size();
	for (int i = 0; i <= validData[0]; i++)
		result.push_back(buffer[0][i]);
	return true;
}



bool SimTester::MysampleLine(const std::deque<vec3>& line, int idx, int nHalfSample, vector<vec3>& result)
{
	if (idx < nHalfSample || line.size() - idx - 1 < nHalfSample)
		return false;

	result.resize(nHalfSample * 2 + 1);
	for (int i = 0; i < 2 * nHalfSample + 1; ++i)
		result[i] = line[i + idx - nHalfSample];

	return true;
}


bool SimTester::findIdxRange(const std::deque<vec3>&line, const vec3& centerPnt, float radius, int& lowID, int& highID)
{
	lowID = 0;
	highID = line.size();

	int i;
	int centerID[2] = { 0, line.size() - 1 };
	float initDist[2] = { 0, 0 };
	float minDist = FLT_MAX;
	for (i = 0; i < line.size() - 1; ++i)
	{
		vec3 d1 = line[i + 1] - line[i];
		vec3 d2 = (centerPnt - line[i]);
		float t = d2.dot(d1) / d1.dot(d1);
		t = min(1.0, max(0.0, t));
		vec3 td1 = t * d1;
		float dist = (d2 - td1).length();
		if (dist < minDist)
		{
			minDist = dist;
			centerID[0] = i;
			centerID[1] = i + 1;
			initDist[0] = td1.length();
			initDist[1] = d1.length() - initDist[0];
		}
	}

	for (i = centerID[0] - 1; i > 0; --i)
	{
		initDist[0] += (line[i] - line[i + 1]).length();
		if (initDist[0] >= radius)
		{
			lowID = i;
			break;
		}
	}

	for (i = centerID[1] + 1; i < line.size(); ++i)
	{
		initDist[1] += (line[i] - line[i - 1]).length();
		if (initDist[1] >= radius)
		{
			highID = i;
			break;
		}
	}
	return true;
}

