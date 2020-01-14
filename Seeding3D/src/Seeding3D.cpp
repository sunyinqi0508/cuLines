#include <iostream>
#include <stdio.h>      /* printf, scanf, puts, NULL */
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <algorithm>
#include <deque>
#include <fstream>
#include <assert.h>
#include "../cuLines/LSH.h"
#include "streamline.h"

#include "Field2D.h"
#include "Integrator.h"
#include "SimTester.h"
using namespace std;
using namespace FileIO;

Vector3 m_minDims;
Vector3 m_maxDims;
StreamlineParam g_param;

#define  STREAM_LINE
//#define POINT_SAMPLE

#ifdef STREAM_LINE
std::vector<LshFunc> create_seeding(int n_funcs, VectorField& field)
{
	std::vector<LshFunc> funcs;
	float minimum_value = field.m_dimensions.l1Norm();
	for (int i = 0; i < n_funcs; ++i) {
		funcs.emplace_back(LshFunc());
		funcs.back().a_ = { gauss_dist(engine), gauss_dist( engine ) ,gauss_dist( engine ) };
		funcs.back().b_ = uni_dist(engine);
		funcs.back().w_ = 4.f;
		funcs.back().lsh_func_offset = -minimum_value;
		funcs.back()._n_buckets = std::numeric_limits<int>::max();
	}
	return funcs;
}
void main()
{

	// read field
	VectorField field;
	string filename = "d:/flow_fields/SquareCylinder/flow_t4048.am";
	
	FileIO::FieldfromAmiramesh(&field, filename.c_str());
	// init params
	int dim_x = field.getNCols();			// dimension of flow field
	int dim_y = field.getNRows();
	int dim_z = field.getNLayers();

	g_param.alpha = 2.0f/10.f;
	g_param.dSep = 0.007 * std::min(std::min(dim_x, dim_y), dim_z);
	g_param.w = 3.0f * g_param.dSep;
	g_param.dSelfsep = g_param.dSep/10.f;
	g_param.dMin = 10.0f* g_param.dSelfsep;
	g_param.minLen = 2*g_param.w;
	g_param.maxLen = 6000 * g_param.minLen;
	g_param.maxSize = 5000;
	g_param.nHalfSample = 10; 
	g_param.print();

	// init seeds
	int num_seed = 5000;					// maximal streamlines / number of seedpoints	
	std::vector<vec3> seeds;
	srand (time(NULL));
	for(int i=0; i<num_seed; i++)
	{
		//@ Yunhai: change this to your seeding
		vec3 s = vec3(rand()%dim_x, rand()%dim_y, rand()%dim_z);
		seeds.push_back(s);
	}
	vector<LshFunc> lshfuncs = create_seeding(4, field);
	vector<int> selections = { 0,1,2,3 };
	HashTable ht(selections, &lshfuncs, TABLESIZE);
	
	RK45Integrator integrator(&field);		// RK45
	//std::vector<Line> streamlines;			// Resulting streamlines
	Vector3* currentline = new Vector3[g_param.maxSize * 2];
	float* curvatures = new float[g_param.maxSize + 1];
	int *pagetable = new int[num_seed];
	std::fill(pagetable, pagetable + num_seed, -1);
	vector<Vector3> v_centroids, v_startpoints, v_endpoints;
	vector<Vector3*> v_fstreamlines;
	vector<int> seg2line;
	getchar();

	for(int i=0; i<num_seed; i++)
	{
		if((i*100) % num_seed == 0)
			printf("\r%d%c", (i*100)/num_seed, '%');
		// current streamline
		currentline[g_param.maxSize - 1] = seeds[i];
		int ptr_front = g_param.maxSize - 2, ptr_back = g_param.maxSize;
		int current_size = ptr_back - ptr_front - 1;


		// grow current_line
		double stepSize[2]   = {0.05,0.05};
		bool   isContinue[2] = {true, true};
		bool stop_forward = false, stop_backward = false;

		while((isContinue[0] || isContinue[1]) && current_size < g_param.maxSize)
		{
			if (isContinue[0])
			{
				// integrate FORWARD over the flow field
				// @Yunhai insert here your integration algorithm (runge-kutta or similar)

				isContinue[0] &= integrator.integrate(currentline[ptr_back - 1], currentline[ptr_back], stepSize[0], stepSize, true);
				if (isContinue[0])
				{
					
					bool stop_forward_self  = false;
					int idx = ptr_back++;
					assert(ptr_back <= g_param.maxSize*2);
					stop_forward = SimTester::isSimilarWithLines(streamlines, pagetable, seg2line.data(), ht, currentline, ptr_front, ptr_back, idx);
					//stop_forward = SimTester::isSimilarWithLines(streamlines, curPath, idx);
					//stop_forward_self = SimTester::isSimilarWithSelf(curPath, idx);

					if (stop_forward || stop_forward_self) 
					{
						ptr_back--;
						isContinue[0] = false;
					}
				}
			}
			
			if (isContinue[1])
			{
				// integrate BACKWARD over the flow field
				// @Yunhai insert here your integration algorithm (runge-kutta or similar)
				isContinue[1] &= integrator.integrate(currentline[ptr_front + 1], currentline[ptr_front], stepSize[1], stepSize + 1, false);

				if (isContinue[1])
				{

					bool stop_backward_self  = false;
					int idx = ptr_front--;
					assert(ptr_front >= -1);
					//stop_backward = SimTester::isSimilarWithLines(streamlines, curPath, idx);
					//stop_backward_self = SimTester::isSimilarWithSelf(curPath, idx);
					stop_backward = SimTester::isSimilarWithLines(streamlines, pagetable, seg2line.data(), ht, currentline, ptr_front, ptr_back, idx);

					if (stop_backward || stop_backward_self) 
					{
						ptr_front++;
						isContinue[1] = false;
					}
				}
				
			}
			current_size = ptr_back - ptr_front - 1;
		}
		
		// add current line to stream line set
		if (/*!(stop_forward || stop_backward) && */Line::getLength(currentline, ptr_back, ptr_front) > (g_param.minLen))
		{
			streamlines.emplace_back(vector<Vector3>());
			streamlines.back().resize(current_size);
			v_fstreamlines.emplace_back(streamlines.back().data());
			f_streamlines = v_fstreamlines.data();

			std::memcpy(streamlines.back().data(), currentline[ptr_front + 1], sizeof(Vector3)*current_size);
			auto new_segments = decomposeByCurvature_perline(M_PI, 10.f, streamlines.back().size(), streamlines.size() - 1, streamlines.back().data(), curvatures);

			for (int j = new_segments.first; j < new_segments.second; ++j)
			{
				seg2line.emplace_back(segments[j].line);
				v_startpoints.emplace_back(streamlines[segments[j].line][segments[j].begin]);
				v_endpoints.emplace_back(streamlines[segments[j].line][segments[j].end - 1]);
				Vector3 _centroid = 0;
				for (int k = segments[j].begin; k < segments[j].end; ++k)
					_centroid += streamlines[segments[j].line][k];
				_centroid /= (float)(segments[j].end - segments[j].begin);
				v_centroids.emplace_back(_centroid / 10.f);

				ht.append(_centroid / 10.f, j);
				start_points = v_startpoints.data();
				end_points = v_endpoints.data();
				second_level.emplace_back(j);
			}


			centroids = v_centroids.data();
			//printf("\tstream added, length:%d\n\tstep[0]=%lf, step[1]=%lf\n", current_size, stepSize[0], stepSize[1]);
		}
		else
			;//printf("\tfailed.\n");
	}

	char buf[256];
	std::cin.getline(buf, 255);
	string reply(buf);
	for (auto& ch : reply)
		if (ch <= 'Z' && ch >= 'A')
			ch += 32;
	if (reply != "no")
	{
		if (reply.size() <= 0)
		{
			int fn_begin = (filename).find_last_of('/');
			fn_begin = fn_begin < 0 ? 0 : fn_begin;
			int fn_end = filename.find_last_of('.');
			fn_end = fn_end < 0 ? filename.size() : fn_end;
			reply = filename.substr(fn_begin, fn_end) + ".bsl";
			FileIO::normalize(1.f, true, Format::STREAMLINE_VECTOR);
			FileIO::OutputBSL(("d:/flow_data/bsldata/" + reply).c_str());
		}
		else {
			const int last_dot = reply.find_last_of('.');
			if (last_dot == -1)
				reply.append(".bsl");
			else
				if (reply.substr(last_dot) == ".obj")
					FileIO::OutputOBJ(("d:/flow_data/" + reply).c_str());
				else
				{
					FileIO::normalize(1.f, true, Format::STREAMLINE_VECTOR);
					FileIO::OutputBSL(("d:/flow_data/bsldata/" + reply).c_str());
				}
		}
		
	}
	
}



#endif
