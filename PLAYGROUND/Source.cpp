#include <iostream>
#include <Windows.h>
using namespace std;
#define OPERATION(OPERAND) int _minus(int a, int b){\
	return a OPERAND b;\
}



extern "C"{
	int __declspec(dllexport) _minus(int a, int b) {
		return a + b;
	}
}

	
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

	return 0;
}