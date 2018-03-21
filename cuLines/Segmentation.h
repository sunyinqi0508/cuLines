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

class SegmentPointLookupTable {
    Segment *seg_;
    Vector3 target_;
    float min_, max_, width_;
    int *slots_, n_slots_;
public:
    SegmentPointLookupTable(int seg_idx);

    ~SegmentPointLookupTable() {
        delete[] slots_;
    }

    int nearest(const Vector3 &v) const;
};

extern void segGlobal(float penalty = 0);
extern void decomposeByCurvature(float, float);

extern std::vector<Segment> segments;
extern std::vector<SegmentPointLookupTable> second_level;


#endif