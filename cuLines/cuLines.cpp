#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuLines.h"
#include "FileIO.h"
#include "Vector.h"
#include "Common.h"
#include <random>
#include <algorithm>
#include <iostream>
#include <future>
using namespace std;
using namespace FILEIO;
typedef Vector3 Vector;


class LshFunc {
	Vector3 a_;
	float b_, w_;
public:

	LshFunc(Vector3 a, float b, float w) : a_{ a }, b_{ b }, w_{ w } {}

	inline float operator() (const Vector3 &x) const {
		return (x.dot(a_) + b_) / w_;
	}

	inline float operator() (Segment *seg) const {
		return (*this)(seg->centroid);
	}

	static std::vector<LshFunc> create(int n, Vector3* points) {
		std::random_device rd{};
		std::mt19937_64 engine{ rd() };
		std::normal_distribution<float> gauss_dist{ 0, 1. };

		Vector3 a(gauss_dist(engine), gauss_dist(engine), gauss_dist(engine));



		std::uniform_real_distribution<float> uni_dist{ 0, w };
		std::vector<LshFunc> fn;
		for (int i = 0; i < n; i++)
			fn.emplace_back( a, uni_dist(engine), w);
		return fn;
	}
};

void segGlobal(float penalty = 0) {
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
				mean /= (j - k );
				for (int l = k + 1; l <= j; l++)
					distqrs += (f_streamlines[i][l] - mean).sq();
				distqrs /= (float)(j - k );
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
		Vector3 _ddx = (_dx1 - _dx_1)/ (_delta_x_1.length() + _delta_x1.length());
		//_ddx = (_ddx / _dx.x)*_dx.x + (_ddx) + ( _ddx);
		const float cur = (_ddx).length();//|r''(curve)|_2
		int j = 0;
		for(; j < 3; j++)
			curvature[j] = cur;
		for (; j < _size - 2; j++) {
			_dx_1 = _dx;
			_dx = _dx1;
			_delta_x_1 = _delta_x1;
			_delta_x1 = _delta_x2;
			_delta_x2 = (f_streamlines[i][j + 2] - f_streamlines[i][j + 1]);
			_dx1 = (_delta_x2 + _delta_x1) / (_delta_x2.length() + _delta_x1.length());
			
			_ddx = (_dx1 - _dx_1)/(_delta_x_1.length() + _delta_x1.length());
			//_ddx = (_ddx / _dx.x)*(_dx.x) + (_ddx / _dx.y)*_dx.y + (_ddx / _dx.z)*_dx.z;

			curvature[j] = (_ddx).length();// / cubic(_dx.length());
		}
		for (; j < _size; j++)
			curvature[j] = curvature[j - 1];

		for (size_t end = 1; end < _size - 1; end++) {
			float len = (f_streamlines[i][end + 1] - f_streamlines[i][end - 1])/2.f;
			float tac = curvature[end] * len;
			if ((tac_sum + tac > crv_thresh || len_sum + len > len_thresh)) {
				Segment seg = Segment(i,begin,end + 1);
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
			Segment seg = Segment(i,begin,_size);
			segments.emplace_back(seg);
		}

	}

}


void arrangement(int n_buckets, int n_tuple, int* buckets) {
	float* x = new float[n_buckets];

}

int main() {

	LoadWaveFrontObject("d:/flow_data/tornado_reduced.obj");
	//FILEIO::normalize();
	FILEIO::toFStreamlines();
	decomposeByCurvature(M_PI, 1000.f);
	vector<LshFunc> funcs = LshFunc::create(32, 4);
	for (const LshFunc& func : funcs) {

	}
	
	//int *a = new int[5], *b = new int[5];
	//for (int i = 0; i< 5; i++) {
	//	a[i] = i;
	//	b[i] = i *(1 + (float)rand() / (float)RAND_MAX);
	//}
	//int *d_a, *d_b;
	//cudaMalloc(&d_a, sizeof(float) * 5);
	//cudaMalloc(&d_b, sizeof(float) * 5);
	//cudaMemcpy(d_b, b, sizeof(float) * 5, cudaMemcpyHostToDevice);
	//cudaMemcpy(d_a, a, sizeof(float) * 5, cudaMemcpyHostToDevice);

	//test << <1, 5 >> > (d_a, d_b);
	//cudaMemcpy(b, d_b, sizeof(float) * 5, cudaMemcpyDeviceToHost);
	//cudaMemcpy(a, d_a, sizeof(float) * 5, cudaMemcpyDeviceToHost);
	//for (int i = 0; i < 5; i++) {
	//	cout << a[i] << endl;
	//}

	return 0;
}