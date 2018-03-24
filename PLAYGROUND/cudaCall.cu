#include "common.h"

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include<stdio.h>
__device__ __host__
struct Struct {
	int a, b;
};

__global__ void test2(Struct *a) {
	a[threadIdx.x].a = blockIdx.x;
	a[threadIdx.x].b = threadIdx.x;
	printf("a\n");
}


void cudacall() {


	int *d_a, *a = new int[64];
	Struct *s = new Struct[4], *d_s;
	cudaMalloc(&d_s, 4 * sizeof(Struct));
	test2 << <4, 4 >> > (d_s); 
	cudaMemcpy(s, d_s, 4 * sizeof(Struct), cudaMemcpyDeviceToHost);
	for(int i = 0;i<4;i++)
		printf("%d %d \n", s[i].a, s[i].b);

}