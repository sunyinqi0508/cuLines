#include "Common.h"
#include "FileIO.h"
#include "../cuLines/LSH.h"  


extern "C" {
	void __declspec(dllexport) transfer(Communicator* pointer) {
		_in_ pointer->filename;
		_in_ pointer->AdditionalParas;

		_out_ pointer->f_streamlines;
		_out_ pointer->results;
	}
}