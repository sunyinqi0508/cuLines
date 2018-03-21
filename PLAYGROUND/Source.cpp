#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <Windows.h>
#include "common.h"
using namespace std;
#define OPERATION(OPERAND) int _minus(int a, int b){\
	return a OPERAND b;\
}



extern "C"{
	int __declspec(dllexport) _minus(int a, int b) {
		return a + b;
	}
}

__global__ __host__
__align__ (8)
struct test {
	int val;
	char align_test;
};
__global__ void testfunc(int *c)
{
	c[2] = 9;
	//printf("Ha! Global func on device %d\n");
};
	//OPERATION(+)

int main()
{
	//printf("hello");
	LARGE_INTEGER begin, end, freq;
	QueryPerformanceCounter(&begin);
	HINSTANCE dllInstance = LoadLibrary("playground.dll");
	//printf("hello %x %d\n",dllInstance, GetLastError());
	int(_stdcall *__minus)(int, int) = (int(*)(int, int))GetProcAddress(dllInstance, "_minus");
	//printf("hello %x %d\n", *_minus, GetLastError());
	cout<<__minus(1, 2)<<endl;
	QueryPerformanceCounter(&freq);

	QueryPerformanceCounter(&end);

	printf("%g\n", (float)(end.QuadPart - begin.QuadPart));
	void *testMem;
	cudaMalloc(&testMem, 16);
	cudaMemcpy(testMem, __minus, 8, cudaMemcpyHostToDevice);
	printf("%s\n", cudaGetErrorName(cudaGetLastError()));
	test tt;
	tt.val = 2;
	tt.align_test = 'f';
	//testfunc();
	printf("%d %d\n", tt.val, sizeof(tt));
	cudacall();
	return 0;
}