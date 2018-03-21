#include "Segmentation.h"
#include "FileIO.h"

#include <algorithm>

using namespace std;
using namespace FILEIO;


Segment::Segment(size_t line, int begin, int end):
	line(line), begin(begin), end(end) 
{
	centroid = 0;
	for (int i = begin; i < end; i++) {
		centroid += f_streamlines[line][i];
	}
	centroid /= cnt;
	start_point = f_streamlines[line][begin];
	end_point = f_streamlines[line][end - 1];

	Vector3 move = centroid - (end_point + start_point) / 2.f;

	start_point += move;
	end_point += move;
}

Segment::operator Vector3(){
	return centroid;
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
				;
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
			currsegs.push_back(Segment(i, ll_f[j], j));
			j = ll_f[j];
		}

		for (Segment seg : currsegs)
			segments.push_back(seg);

	}
	delete[] f;
	delete[] ll_f;
}
//#pragma optimize("", on)

void decomposeByCurvature(float crv_thresh, float len_thresh) {
	// the first derivative
	float *curvature;
	curvature = new float[Streamline::max_size()];


	for (int i = 0; i < n_streamlines; i++) {
		size_t begin = 0;
		float tac_sum = 0.f, len_sum = 0.f;

		int _size = Streamline::size(i);
		if (_size < 5)
		{
			segments.push_back(Segment(i, 0, _size));
			continue;
		}
		const Vector3 _delta_x_2 = (f_streamlines[i][1] - f_streamlines[i][0]);
		Vector3 _delta_x_1 = (f_streamlines[i][2] - f_streamlines[i][1]);
		Vector3 _delta_x1 = (f_streamlines[i][3] - f_streamlines[i][2]);
		Vector3 _delta_x2 = (f_streamlines[i][4] - f_streamlines[i][3]);

		Vector3 _dx_1 = (_delta_x_2 + _delta_x_1) / (_delta_x_2.length() + _delta_x_1.length()),
			_dx = (_delta_x_1 + _delta_x1) / (_delta_x_1.length() + _delta_x1.length()),
			_dx1 = (_delta_x1 + _delta_x2) / (_delta_x1.length() + _delta_x2.length());//grad(x(curve), y(curve), z(curve)) by curve
		Vector3 _ddx = (_dx1 - _dx_1) / (_delta_x_1.length() + _delta_x1.length());
		//_ddx = (_ddx / _dx.x)*_dx.x + (_ddx) + ( _ddx);
		int j = 0;
		for (; j < 3; j++)
			curvature[j] = (_ddx).length();
		for (; j < _size - 2; j++) {
			_dx_1 = _dx;
			_dx = _dx1;
			_delta_x_1 = _delta_x1;
			_delta_x1 = _delta_x2;
			_delta_x2 = (f_streamlines[i][j + 2] - f_streamlines[i][j + 1]);
			_dx1 = (_delta_x2 + _delta_x1) / (_delta_x2.length() + _delta_x1.length());

			_ddx = (_dx1 - _dx_1) / (_delta_x_1.length() + _delta_x1.length());
			//_ddx = (_ddx / _dx.x)*(_dx.x) + (_ddx / _dx.y)*_dx.y + (_ddx / _dx.z)*_dx.z;

			curvature[j] = (_ddx).length();// / cubic(_dx.length());
		}
		for (; j < _size; j++)
			curvature[j] = curvature[j - 1];

		for (size_t end = 1; end < _size - 1; end++) {
			float len = (f_streamlines[i][end + 1] - f_streamlines[i][end - 1]) / 2.f;
			float tac = curvature[end] * len;
			if ((tac_sum + tac > crv_thresh || len_sum + len > len_thresh)) {
				Segment seg = Segment(i, begin, end + 1);
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
		if (segments.empty() || segments.back().end != _size) {
			segments.emplace_back(i, begin, _size);
		}
	}
}

void initializeSecondLevel() {
    for (std::size_t i = 0; i < segments.size(); i++)
        second_level.emplace_back(i);
}

SegmentPointLookupTable::SegmentPointLookupTable(int seg_idx) :
    seg_{ &segments[seg_idx] }
{
    // data member initialization
    target_ = seg_->end_point - seg_->start_point;
    // compute projections
    std::vector<std::pair<float, int>> proj_vals;
    proj_vals.reserve(seg_->cnt);
    for (int i = seg_->begin; i < seg_->end; i++) {
        proj_vals.emplace_back((f_streamlines[seg_->line][i] - seg_->start_point).project(target_), i);
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
        auto fill_end = static_cast<int>(
            std::floor((proj_vals[i].first + proj_vals[i - 1].first) / 2 / width_)
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
    auto p = (v - seg_->start_point).project(target_);
    if (p - min_ < EPSILON)
        return slots_[0];
    if (max_ - p < EPSILON)
        return slots_[n_slots_ - 1];
    return slots_[static_cast<int>(std::floor((p - min_) / width_))];
}

