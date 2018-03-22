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

	operator const Vector3&();
};

class SegmentPointLookupTable {
    const Segment &seg_;
    Vector3 target_;
    float min_, max_, width_;
    int *slots_ = 0, n_slots_;
public:
    explicit SegmentPointLookupTable(int seg_idx);
	//SegmentPointLookupTable(SegmentPointLookupTable &&) = delete;
    ~SegmentPointLookupTable() {
		//delete[] slots_;
    }

    int nearest(const Vector3 &v) const;
};

extern void segGlobal(float penalty = 0);
extern void decomposeByCurvature(float, float);
extern void initializeSecondLevel();

extern std::vector<Segment> segments;
extern std::vector<SegmentPointLookupTable> second_level;


#endif