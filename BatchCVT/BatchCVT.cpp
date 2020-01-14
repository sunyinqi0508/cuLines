
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
#include <stdio.h>
#include <vector>
#include <Windows.h>
#undef max
#undef min
//#include "Vector.h"
#include "FileIO.h"
#include <Psapi.h>
#include <filesystem>
using namespace std;
using namespace FileIO;

void segmentaion() {

}


int main()
{
	LARGE_INTEGER freq, begin, end;
	//LoadWaveFrontObject("d:/flow_data/wall-mounted-10k.obj");
	//OutputBSL();
	namespace fs = std::filesystem;
	string directory = "d:/flow_data/";
	string destination = directory + "BSLDataNormalized/";

	QueryPerformanceCounter(&begin);
	FILE *fp;
	
	freopen_s(&fp, (destination + "BSLConverter.log").c_str(), "w", stdout);
	for (auto file : fs::directory_iterator(directory)) {
		if (file.status().type() == fs::file_type::regular)
		{
			string extension = file.path().filename().extension().string();
			if (extension != ".obj")
				continue;
			string filename = file.path().filename().string();

			printf("data %s ", filename.c_str());
			
			LoadWaveFrontObject((directory + filename).c_str());

			auto normalized = normalize();

			printf("scale: %f\n", normalized.multiplier);

			OutputBSL((destination + filename.substr(0, filename.size() - 3) + "bsl").c_str());
			//ReadBSL(file.path().string().c_str());
		}
	}
	//ReadBSL("d:/flow_data/BSLData/test.bsl");
	QueryPerformanceCounter(&end);
	QueryPerformanceFrequency(&freq);

	auto myHandle = GetCurrentProcess();
	PROCESS_MEMORY_COUNTERS pmc;
	if (GetProcessMemoryInfo(myHandle, &pmc, sizeof(pmc)))
		printf("Time elapsed: %f\nMemory consumption: %fMB\n", (double)(end.QuadPart - begin.QuadPart) / (double)freq.QuadPart, (pmc.WorkingSetSize / 1024.) / 1024.);

}
