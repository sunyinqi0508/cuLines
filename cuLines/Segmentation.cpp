#include "Segmentation.h"
#include "FileIO.h"

#include <algorithm>

using namespace std;
using namespace FileIO;

Vector3* centroids = 0;
Vector3* start_points = 0;
Vector3* end_points = 0;
Vector3* dir = 0;
 
Vector3 *&Segment::_centroids = centroids;
Vector3 *&Segment::_start_points = start_points;
Vector3 *&Segment::_end_points = end_points;
Vector3 *&Segment::_dir = dir;


//vector<Vector3> vec_centroids;

Segment::Segment(int _line, int _begin, int _end, int _id):
	line(_line), begin(_begin), end(_end), id(_id), cnt(end - begin)
{
	while (end - 1 > begin) {
		if (f_streamlines[line][end - 1] - f_streamlines[line][begin] < 1e-14)
			end--;
		else break;
	}
	if (begin >= end - 1)
	{
		this->line = -1;
		return;
	}

}

Segment::operator const Vector3&(){
	return centroids[id];
}
Segment& Segment::operator=(const Segment& _s) {

	line = _s.line;

	begin = _s.begin;
	end = _s.end;
	cnt = _s.cnt;
	 
	id = _s.id;

	return *this;
}
void Segment::seg_init(const float tac, const float th_len)
{
	segments.clear();
	decomposeByCurvature(tac, th_len);
	if (centroids&& start_points&&end_points&&dir)
	{
		delete[] centroids;
		delete[] start_points;
		delete[] end_points;
		delete[] dir; 
	}
	centroids = new Vector3[segments.size()];
	start_points = new Vector3[segments.size()];
	end_points = new Vector3[segments.size()];
	dir = new Vector3[segments.size()];
	for (int i = 0; i < segments.size(); i++) {
		if (segments[i].line == -1)
			printf("err");
		segments[i].id = i;
	}
	for (const auto& seg : segments) {

		centroids[seg.id] = 0;// (_start_points[seg.id] + _end_points[seg.id]) / 2.f;
		for (int i = seg.begin; i < seg.end; i++) {
			centroids[seg.id] += f_streamlines[seg.line][i];
			pt_to_segs[Streamline::offsets[seg.line] + i] = seg.id;
		}
		centroids[seg.id] /= seg.cnt;

		start_points[seg.id] = f_streamlines[seg.line][seg.begin];
		end_points[seg.id] = f_streamlines[seg.line][seg.end - 1];

		Vector3 move = centroids[seg.id] - (start_points[seg.id]+ end_points[seg.id]) / 2.f;

		start_points[seg.id] += move;
		end_points[seg.id] += move;

		dir[seg.id] = (end_points[seg.id] - start_points[seg.id]).normalized();
	}

}
std::vector<Segment> segments;

void segGlobal(float penalty) {
	printf("%d\n", n_points);

	float *f = new float[Streamline::max_size()];
	int *ll_f = new int[Streamline::max_size()];
	for (int i = 0; i < n_streamlines; i++) {
		f[0] = 0;
		ll_f[0] = 0;
		for (int j = 1; j < Streamline::size(i); j++) {
			f[j] = numeric_limits<float>::max();
			for (int k = 0; k < j; k++) {
				Vector3 mean = 0;
				float distqrs = 0;

				for (int l = k + 1; l <= j; l++)
					mean += f_streamlines[i][j];
				mean /= (j - k);
				for (int l = k + 1; l <= j; l++)
					distqrs += (f_streamlines[i][l] - mean).sq();
				distqrs /= (float)(j - k);
				if (f[k] + distqrs + penalty < f[j])
				{
					f[j] = f[k] + distqrs + penalty;
					ll_f[j] = k;
				}

			}
		}

		vector<Segment> currsegs;
		int j = Streamline::size(i) - 1;

		while (j > 0) {
			currsegs.push_back(Segment(i, ll_f[j], j, -1));
			j = ll_f[j];
		}

		for (Segment& seg : currsegs)
		{
			seg.id = segments.size();
			segments.push_back(seg);
		}
	}
	delete[] f;
	delete[] ll_f;
}
//#pragma optimize("", on)
__forceinline pair<int, int> decomposeByCurvature_perline(
	const float crv_thresh, const float len_thresh, int len_streamline, int idx_streamline, const Vector3 *this_streamline, float *curvature = 0) {
	if (!curvature)
		curvature = new float[len_streamline];

	size_t begin = 0;
	float tac_sum = 0.f, len_sum = 0.f;
	const int segment_begin_idx = segments.size();
	if (len_streamline < 5)
	{
		segments.emplace_back(idx_streamline, 0, len_streamline, segments.size());
		if (segments.back().line == -1)
			segments.pop_back();
		return make_pair(segment_begin_idx, segments.size());
	}
	const Vector<double> _delta_x_2 = (this_streamline[1] - this_streamline[0]);
	Vector<double> _delta_x_1 = (this_streamline[2] - this_streamline[1]);
	Vector<double> _delta_x1 = (this_streamline[3] - this_streamline[2]);
	Vector<double> _delta_x2 = (this_streamline[4] - this_streamline[3]);

	Vector<double> _dx_1 = (_delta_x_2 + _delta_x_1) / (_delta_x_2.length() + _delta_x_1.length()),
		_dx = (_delta_x_1 + _delta_x1) / (_delta_x_1.length() + _delta_x1.length()),
		_dx1 = (_delta_x1 + _delta_x2) / (_delta_x1.length() + _delta_x2.length());//grad(x(curve), y(curve), z(curve)) by curve
	Vector<double> _ddx = (_dx1 - _dx_1) / (_delta_x_1.length() + _delta_x1.length());
	//_ddx = (_ddx / _dx.x)*_dx.x + (_ddx) + ( _ddx);
	int j = 0;
	for (; j < 3; j++)
		curvature[j] = (_ddx).length();
	for (; j < len_streamline - 2; j++) {
		_dx_1 = _dx;
		_dx = _dx1;
		_delta_x_1 = _delta_x1;
		_delta_x1 = _delta_x2;
		_delta_x2 = (this_streamline[j + 2] - this_streamline[j + 1]);
		_dx1 = (_delta_x2 + _delta_x1) / (_delta_x2.length() + _delta_x1.length());

		_ddx = (_dx1 - _dx_1) / (_delta_x_1.length() + _delta_x1.length());
		//_ddx = (_ddx / _dx.x)*(_dx.x) + (_ddx / _dx.y)*_dx.y + (_ddx / _dx.z)*_dx.z;

		curvature[j] = (_ddx).length();// / cubic(_dx.length());
	}
	for (; j < len_streamline; j++)
		curvature[j] = curvature[j - 1];

	for (size_t end = 1; end < len_streamline - 1; end++) {
		float len = (this_streamline[end + 1] - this_streamline[end - 1]) / 2.f;
		float tac = curvature[end] * len;
		if ((tac_sum + tac > crv_thresh || len_sum + len > len_thresh)) {
			Segment seg = Segment(idx_streamline, begin, end + 1, segments.size());
			segments.emplace_back(seg);
			tac_sum = len_sum = 0.f;
			begin = end;
		}
		else {
			tac_sum += tac;
			len_sum += len;
		}
	}
	// finalize the last segment (if unfinished)
	if (segments.empty() || segments.back().line != idx_streamline || segments.back().end != len_streamline) {
		segments.emplace_back(idx_streamline, begin, len_streamline, segments.size());
	}

	Segment* _data = segments.data();
	int segment_end_idx = segment_begin_idx;
	for (int i = segment_begin_idx; i < segments.size(); ++i)
		if (_data[i].line != -1 && segment_end_idx++ != i)
			_data[segment_end_idx] = (_data[i]);

	segments.resize(segment_end_idx);
	return std::make_pair(segment_begin_idx, segment_end_idx);
}
void decomposeByCurvature(const float crv_thresh, const float len_thresh) {

	if (pt_to_segs) {
		delete[] pt_to_segs;
	}
	pt_to_segs = new int[n_points];
	if (curvatures)
	{
		delete[] curvatures[0];
		delete[] curvatures;
		curvatures = 0;
	}

	if (n_streamlines > 0 && n_points > 0) {
		curvatures = new float*[n_streamlines];
		curvatures[0] = new float[n_points];
	}
	for (int i = 1; i < n_streamlines; ++i)
		curvatures[i] = curvatures[i - 1] + Streamline::size(i - 1);
	

	float *curvature;
	//curvature = new float[Streamline::max_size()];

	for (int i = 0; i < n_streamlines; i++) {
		decomposeByCurvature_perline(crv_thresh, len_thresh, Streamline::sizes[i], i, f_streamlines[i], curvatures[i]);
		//curvature = curvatures[i];
		//size_t begin = 0;
		//float tac_sum = 0.f, len_sum = 0.f;

		//int _size = Streamline::size(i);
		//if (_size < 5)
		//{
		//	segments.emplace_back(i, 0, _size, segments.size());
		//	continue;
		//}
		//const Vector<double> _delta_x_2 = (f_streamlines[i][1] - f_streamlines[i][0]);
		//Vector<double> _delta_x_1 = (f_streamlines[i][2] - f_streamlines[i][1]);
		//Vector<double> _delta_x1 = (f_streamlines[i][3] - f_streamlines[i][2]);
		//Vector<double> _delta_x2 = (f_streamlines[i][4] - f_streamlines[i][3]);

		//Vector<double> _dx_1 = (_delta_x_2 + _delta_x_1) / (_delta_x_2.length() + _delta_x_1.length()),
		//	_dx = (_delta_x_1 + _delta_x1) / (_delta_x_1.length() + _delta_x1.length()),
		//	_dx1 = (_delta_x1 + _delta_x2) / (_delta_x1.length() + _delta_x2.length());//grad(x(curve), y(curve), z(curve)) by curve
		//Vector<double> _ddx = (_dx1 - _dx_1) / (_delta_x_1.length() + _delta_x1.length());
		////_ddx = (_ddx / _dx.x)*_dx.x + (_ddx) + ( _ddx);
		//int j = 0;
		//for (; j < 3; j++)
		//	curvature[j] = (_ddx).length();
		//for (; j < _size - 2; j++) {
		//	_dx_1 = _dx;
		//	_dx = _dx1;
		//	_delta_x_1 = _delta_x1;
		//	_delta_x1 = _delta_x2;
		//	_delta_x2 = (f_streamlines[i][j + 2] - f_streamlines[i][j + 1]);
		//	_dx1 = (_delta_x2 + _delta_x1) / (_delta_x2.length() + _delta_x1.length());

		//	_ddx = (_dx1 - _dx_1) / (_delta_x_1.length() + _delta_x1.length());
		//	//_ddx = (_ddx / _dx.x)*(_dx.x) + (_ddx / _dx.y)*_dx.y + (_ddx / _dx.z)*_dx.z;

		//	curvature[j] = (_ddx).length();// / cubic(_dx.length());
		//}
		//for (; j < _size; j++)
		//	curvature[j] = curvature[j - 1];

		//for (size_t end = 1; end < _size - 1; end++) {
		//	float len = (f_streamlines[i][end + 1] - f_streamlines[i][end - 1]) / 2.f;
		//	float tac = curvature[end] * len;
		//	if ((tac_sum + tac > crv_thresh || len_sum + len > len_thresh)) {
		//		Segment seg = Segment(i, begin, end + 1, segments.size());
		//		segments.emplace_back(seg);
		//		tac_sum = len_sum = 0.f;
		//		begin = end;
		//	}
		//	else {
		//		tac_sum += tac;
		//		len_sum += len;
		//	}
		//}
		//// finalize the last segment (if unfinished)
		//if (segments.empty() || segments.back().line != i || segments.back().end != _size) {
		//	segments.emplace_back(i, begin, _size, segments.size());
		//}

	}

	//vector<Segment>::iterator it;
	//it = segments.begin();
	//while (it != segments.end()) {

	//	if (it->line == -1)
	//		it = segments.erase(it);
	//	else
	//		it++;
	//}

	//delete[] curvature;
}

void initializeSecondLevel() {
    second_level.reserve(segments.size());
    for (std::size_t i = 0; i < segments.size(); i++)
		second_level.emplace_back(i);
}

SegmentPointLookupTable::SegmentPointLookupTable(int seg_idx) :
    seg_{ segments[seg_idx] }
{
//	printf("Init: %x", *this);
    // data member initialization
    target_ = end_points[seg_idx] - start_points[seg_idx];
    // compute projections
    std::vector<std::pair<float, int>> proj_vals;
    proj_vals.reserve(seg_.cnt);
    for (int i = seg_.begin; i < seg_.end; i++) {
        proj_vals.emplace_back((f_streamlines[seg_.line][i] - start_points[seg_.id]).project(target_), i);

		if (isnan(proj_vals.back().first))
			puts("Detected NaN in projection values.");
    }
    std::sort(begin(proj_vals), end(proj_vals));
    // create mapping slots
    n_slots_ = static_cast<int>(proj_vals.size() * 5);
    slots_ = new int[n_slots_];
    // min, max and width
    min_ = proj_vals.front().first;
    max_ = proj_vals.back().first;
    width_ = (max_ - min_) / static_cast<float>(n_slots_);
    // middle points
    int fill_begin = 0;
    for (std::size_t i = 1; i < proj_vals.size(); i++) {
        // `fill_end' is the middle point of projection
        // `i - 1' and projection `i'
        auto mid_proj_val = (proj_vals[i].first + proj_vals[i - 1].first) / 2;
        auto fill_end = static_cast<int>(
            std::floor((mid_proj_val - min_) / width_)
        );
        for (; fill_begin < fill_end; fill_begin++)
            slots_[fill_begin] = proj_vals[i - 1].second;
    }
    // now `fill_begin' is the middle point of last projection
    // and last two projection
    for (; fill_begin < n_slots_; fill_begin++)
        slots_[fill_begin] = proj_vals.back().second;
}

static const float EPSILON = 1e-7;

int SegmentPointLookupTable::nearest(const Vector3 &v) const {
    auto p = (v - start_points[seg_.id]).project(target_);
    if (p - min_ < EPSILON)
        return slots_[0];
    if (max_ - p < width_)
        return slots_[n_slots_ - 1];

    return slots_[static_cast<int>(std::floor((p - min_) / width_))];
}
int* pt_to_segs = 0;
std::vector<SegmentPointLookupTable> second_level;
