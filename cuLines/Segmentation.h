#ifndef _H_SEGMENTATION
#define _H_SEGMENTATION


#include "Vector.h"
#include <vector>

class Segment {
public:
	const size_t line;
	const int begin, end, cnt;
	Vector3 centroid = 0;
	Vector3 start_point, end_point;
	Segment(size_t line, int begin, int end);

	operator Vector3();
};

extern void segGlobal(float penalty = 0);
extern void decomposeByCurvature(float, float);

extern std::vector<Segment> segments;


#endif