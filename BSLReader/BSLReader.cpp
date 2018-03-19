#include "FileIO.h"
#include <string>
#include <iostream>
using namespace std;
using namespace FILEIO;

#define max(a, b) ((a) > (b)? (a):(b))
#define min(a, b) ((a) < (b)? (a):(b))
#define bound(x) ((x)>fn_len? 0:(x)) 

int main(int argc, char** argv) {

	if (argc >= 2) {

		LoadWaveFrontObject(argv[1]);
		string str_file = string(argv[1]);
		size_t fn_len = str_file.size();
		size_t fn_start = max(bound(str_file.find_last_of('/')), bound(str_file.find_last_of('\\')));
		size_t fn_end = str_file.find_last_of('.');

		if (fn_end <= fn_start)
			fn_end = str_file.size();
		OutputBSL(((str_file.substr(fn_start + 1, fn_end - fn_start - 1)) + ".bsl").c_str());

	}

}