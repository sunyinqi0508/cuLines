#ifndef _CULINES_H
#define _CULINES_H
#include <vcruntime.h>
#include "Vector.h"
#include "FileIO.h"
#include <vector>

class Segment {
public:
	size_t line;
	int begin, end, cnt = end - begin;
	Vector3 centroid = 0;
	Vector3 start_point, end_point;
	Segment(size_t line, int begin, int end) :
		line(line), begin(begin), end(end) {
		centroid = 0;
		for (int i = begin; i < end; i++) {
			centroid += FILEIO::f_streamlines[line][i];
		}
		centroid /= cnt;
		start_point = f_streamlines[line][begin];
		end_point = f_streamlines[line][end - 1];

		Vector3 move = centroid - (end_point + start_point)/2.f;

		start_point += move;
		end_point += move;
	}
	operator Vector3() {
		return centroid;
	}
};

std::vector<Segment> segments;
#endif