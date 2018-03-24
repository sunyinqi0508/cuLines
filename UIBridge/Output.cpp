#include "Common.h"
#include "FileIO.h"
#include "../cuLines/LSH.h"  
#include <memory>

extern "C" {
	void __declspec(dllexport) transfer(Communicator* pointer) {

		_in_ pointer->filename;
		_in_ pointer->AdditionalParas;
		_out_ pointer->f_streamlines;
		_out_ pointer->results;

		initialize(pointer->filename);
		pointer->f_streamlines = reinterpret_cast<float**>(FileIO::f_streamlines);
		pointer->n_streamlines = FileIO::n_streamlines;
		pointer->sizes = FileIO::Streamline::sizes;
		pointer->n_points = FileIO::n_points;
		//TODO: assign alpha array here;
		pointer->alpha = alpha;
		pointer->colors = 0;// new int[FileIO::n_points];
	}
}
