#include "common.h"

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include<stdio.h>
__global__ void test2(int *a=0) {
	a[threadIdx.x] = blockIdx.x;
	printf("a\n");
}


void cudacall() {


	int *d_a, *a = new int[64];
	cudaMalloc(&d_a, 64);
	test2 << <4, 4 >> > (d_a); 
	cudaMemcpy(a, d_a, 16, cudaMemcpyDeviceToHost);
	printf("%d", a[3]);

}