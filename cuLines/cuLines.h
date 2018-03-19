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
	Segment(size_t line, int begin, int end) :
		line(line), begin(begin), end(end) {
		centroid = 0;
		for (int i = begin; i < end; i++) {
			centroid += FILEIO::f_streamlines[line][i];
		}
		centroid /= cnt;
	}

};

std::vector<Segment> segments;
#endif