#ifndef _H_SEGMENTATION
#define _H_SEGMENTATION


#include "Vector.h"
#include <vector>
class Segment {
public:
	int line;
	int id;
	int begin, end, cnt;
//	Vector3 centroid;
	//float len;
	Segment() = default;
	Segment(int line, int begin, int end, int _id);

	operator const Vector3&();
	Segment& operator=(const Segment& _s);

	static Vector3*& _dir;
	static Vector3*& _centroids;
	static Vector3*& _start_points;
	static Vector3*& _end_points;

	static void seg_init(const float tac, const float th_len);
private:

};
//#pragma optimize("", off)
class SegmentPointLookupTable {
    const Segment seg_;
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

//#pragma optimize("", on)

extern void segGlobal(float penalty = 0);
extern __forceinline std::pair<int, int> decomposeByCurvature_perline(
	const float crv_thresh, const float len_thresh, int len_streamline, int idx_streamline, const Vector3 *this_streamline, float *curvature);
extern void decomposeByCurvature(const float, const float);
extern void initializeSecondLevel();
extern Vector3* dir;
extern Vector3* centroids;
extern Vector3* start_points;
extern Vector3* end_points;
extern int* pt_to_segs;
extern std::vector<Segment> segments;
extern std::vector<SegmentPointLookupTable> second_level;


#endif
