#define _CRT_SECURE_NO_WARNINGS
#include "Common.h"
#include "Parameters.h"
#include "FileIO.h"
#include "../cuLines/LSH.h"
#include "../cuLines/Segmentation.h"
#include <memory>
#include <Windows.h>
int getIntColor(const Vector3 c) {
	int ret = 0xff;
	ret = ret + (((int)c.x) << 24) + (((int)c.y) << 16) + (((int)c.z) << 8);
	return ret;
}
using namespace FileIO;
extern "C" {
	void __declspec(dllexport) transfer(Communicator* pointer) {

		_in_ pointer->filename;
		_in_ pointer->AdditionalParas;
		_out_ pointer->f_streamlines;
		_out_ pointer->results;

		if(pointer->lsh_radius > 0)
			initialize(pointer->filename, pointer->n_streamlines, pointer->application, pointer->lsh_radius);
		else
			initialize(pointer->filename, pointer->n_streamlines, pointer->application);

		//doCriticalPointQuery("E:\\flow_data\\cp\\5cp.cp");


		pointer->f_streamlines = reinterpret_cast<float**>(FileIO::f_streamlines);
		pointer->n_streamlines = FileIO::n_streamlines;
		pointer->sizes = FileIO::Streamline::sizes;
		pointer->n_points = FileIO::n_points;
		
		//TODO: assign alpha array here;
#ifdef APPLICATION_ALPHA
		pointer->alpha = alpha;
#else
		pointer->alpha = 0;
#endif
#ifdef APPLICATION_SALIENCY
		Vector3 color0 = { 0x49,0xb9,0xf9 }, colordiff{ 0xa9,-0x7e,-0xad };
		pointer->colors = new int[FileIO::n_points];
		for (int i = 0; i < FileIO::n_points; i++)
			if (alpha[i] != -1)
				pointer->colors[i] = getIntColor(color0 + colordiff * alpha[i]);
			else
				pointer->colors[i] = getIntColor( Vector3(0x89,0x89,0x89) );
#else
		pointer->colors = 0;
#endif

		//pointer->colors = new int[FileIO::n_points];
		//int cnt_seg = 0;
		//for (const auto seg : segments) {
		//	if(cnt_seg++%2)
		//		std::fill(pointer->colors+Streamline::offsets[seg.line] + seg.begin, pointer->colors + Streamline::offsets[seg.line] + seg.end, 0x49b9f9ff);
		//	else
		//		std::fill(pointer->colors + Streamline::offsets[seg.line] + seg.begin, pointer->colors + Streamline::offsets[seg.line] + seg.end, 0xb64606ff);
		//}
		//pointer->colors = 0;// new int[FileIO::n_points];
	}

}

