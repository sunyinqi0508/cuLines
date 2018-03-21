//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
#include "LSH.h"
#include "FileIO.h"
#include "Vector.h"
#include "Common.h"
#include "cuStubs.cuh"
#include "Segmentation.h"
#include <stdint.h>

#include <unordered_map>
#include <map>
#include <set>
#include <list>
#include <random>
#include <algorithm>
#include <iostream>
#include <future>
using namespace std;
using namespace FILEIO;
typedef Vector3 Vector;
constexpr int64_t Prime = (1ll << 32) - 5;

class LshFunc {

protected:
	Vector3 a_;
	vector<int32_t> *buckets;
	float b_, w_;
	unsigned char* _h;
	int _n_buckets;
public:

	LshFunc() = default;
	LshFunc(Vector3 a, float b, float w) : a_{ a }, b_{ b }, w_{ w } {}
	inline unsigned char& h(int i) {
		return _h[i];
	}
	inline int operator() (const Vector3 &x) const {
		return static_cast<int>(__macro_bound(
			(x.dot(a_) + b_) / w_,0, _n_buckets - 1
		));
	}

	inline int operator() (Segment *seg) const {
		return (*this)(seg->centroid);
	}

	inline const int& get_n_buckets() const { return _n_buckets; }
	~LshFunc() {
	}
	
	void deinit() {
		delete[] buckets;
		delete[] _h;
	}

	static pair<std::vector<LshFunc>*, int> create(int n, Segment* samples, size_t n_samples, int n_buckets) {


		std::normal_distribution<float> gauss_dist{ 0, 1. };
		
		std::vector<LshFunc>* fn = new vector<LshFunc>;
		
		float *projections = new float[n_samples];
		const float avg_size = (float)n_samples / (float)n_buckets;
		int bestfunc = -1;
		float min_variation = numeric_limits<float>::max();
		
		for (int i = 0; i < n; i++)
		{
			LshFunc curr_func;
			curr_func.buckets = new vector<int32_t>[n_buckets];
			curr_func._h = new unsigned char[n_buckets];
			curr_func._n_buckets = n_buckets;
			fn->push_back(curr_func);
			float variation = 0;
			do {
				variation = 0;
				for (int j = 0; j < n_buckets; j++)
					(*fn)[i].buckets[j].clear();
				Vector3 a(gauss_dist(engine), gauss_dist(engine), gauss_dist(engine));
				(*fn)[i].a_ = a;

				float _min = numeric_limits<float>::max(), _interval = numeric_limits<float> ::min();
				for (int j = 0; j < n_samples; j++) {
					projections[j] = (*fn)[i].a_.dot(samples[j].centroid);
					_min = __macro_min(_min, projections[j]);
					_interval = __macro_max(_interval, projections[j]);
				}

				_interval -= _min;
				const float _div = _interval / n_buckets;
				(*fn)[i].w_ = _div;
				std::uniform_real_distribution<float> uni_dist{ 0, _div };
				(*fn)[i].b_ = uni_dist(engine);
				for (int j = 0; j < n_samples; j++)
					(*fn)[i].buckets
					[(*fn)[i]( samples + j )].
					push_back(j);
				for (int j = 0; j < n_buckets; j++)
					if ((*fn)[i].buckets[j].size() < avg_size)
						variation += avg_size - (*fn)[i].buckets[j].size();//l2 norm might be better choice;
				static int hit = 0;
				hit++;
			} while (variation > (n_samples)/3.f);//Todo: determining efficiency of hash function
			
			if (variation < min_variation) {
				min_variation = variation;
				bestfunc = i;
			}

			for(int j =0;j < n_buckets; j++)
				for (int _sample : (*fn)[i].buckets[j])
					projections[_sample] = j;
		}
		 
		return make_pair(fn, bestfunc);
	}
};

class HashTable {
public:
	
	struct LSHPoint {
		vector<int> ptr_segments;
		LSHPoint(int ptr_segment) :
			ptr_segments() {
			ptr_segments.push_back(ptr_segment);
		}
	};
	
	int tablesize;
	vector<int> LSHFunctions;//indices of lsh functions
	vector<int> r1, r2;
	vector<LshFunc> *function_pool;
	  
	unordered_map<int64_t, LSHPoint *>* lshTable;
	HashTable(vector<int>LSHFunctions, int tablesize, vector<LshFunc> *function_pool, Segment *samples, int n_samples) 
		: LSHFunctions(LSHFunctions), tablesize(tablesize), function_pool(function_pool)
	{
		lshTable = new unordered_map<int64_t, LSHPoint*>[tablesize];
		std::uniform_int_distribution<int> uni_intdist{};
		for (int funcidx : LSHFunctions) {
			r1.push_back(uni_intdist(engine));
			r2.push_back(uni_intdist(engine));
		}
		for (int i = 0; i < n_samples; i++) {//i -> ptr to segment
			
			int64_t fingerprint1 = 0, fingerprint2 = 0;
			for (int j = 0; j < LSHFunctions.size(); j++) {

				const int64_t tmp_fp1 = r1[j] * (*function_pool)[LSHFunctions[j]].h(i);
				const int64_t tmp_fp2 = r2[j] * (*function_pool)[LSHFunctions[j]].h(i);
				
				fingerprint1 += (tmp_fp1 >> 32) ? ((tmp_fp1 >> 32) + 5) : tmp_fp1;
				fingerprint2 += (tmp_fp2 >> 32) ? ((tmp_fp2 >> 32) + 5) : tmp_fp2;

				fingerprint1 = (fingerprint1 >> 32) ? ((fingerprint1 >> 32) + 5) : fingerprint1;
				fingerprint2 = (fingerprint2 >> 32) ? ((fingerprint2 >> 32) + 5) : fingerprint2;

			}

			fingerprint1 %= tablesize;
			fingerprint2 %= Prime;
			if (lshTable[fingerprint1].find(fingerprint2) == lshTable[fingerprint1].end()) {
				lshTable[fingerprint1][fingerprint2] = new LSHPoint(i);
			}
			else {
				lshTable[fingerprint1][fingerprint2]->ptr_segments.push_back(i);
			}

		}
	}

	void Query(vector<int>& results, Vector3 point, bool from_distinct_lines) {

		int64_t fingerprint1 = 0, fingerprint2 = 0;
		unordered_map<int, int> res;

		for (int j = 0; j < LSHFunctions.size(); j++) {
			const LshFunc curr_func = (*function_pool)[LSHFunctions[j]];
			const int n_buckets = curr_func.get_n_buckets();
			const int64_t tmp_fp1 = r1[j] * curr_func(point);
			const int64_t tmp_fp2 = r2[j] * curr_func(point);

			fingerprint1 += (tmp_fp1 >> 32) ? ((tmp_fp1 >> 32) + 5) : tmp_fp1;
			fingerprint2 += (tmp_fp2 >> 32) ? ((tmp_fp2 >> 32) + 5) : tmp_fp2;

			fingerprint1 = (fingerprint1 >> 32) ? ((fingerprint1 >> 32) + 5) : fingerprint1;
			fingerprint2 = (fingerprint2 >> 32) ? ((fingerprint2 >> 32) + 5) : fingerprint2;


		}
		fingerprint1 %= tablesize;
		fingerprint2 %= Prime;
		for (int resultint_pt : lshTable[fingerprint1][fingerprint2]->ptr_segments) {
			const Segment this_seg = segments[resultint_pt];
			unordered_map<int, int>::iterator findings = res.find(this_seg.line);
			
			if (findings != res.end()) {
				int& curr_seg = findings->second;
				if ((point - this_seg.centroid).length() < (point - segments[curr_seg].centroid).length()) {
					curr_seg = resultint_pt;
				}
			}
			else
				res[this_seg.line] = resultint_pt;
		}
		unordered_map<int, int>::iterator it = res.begin();

        if (from_distinct_lines) {
            auto seg_of_lines = new int[n_streamlines];
            std::fill(seg_of_lines, seg_of_lines + n_streamlines, -1);
            while (it != res.end()) {
                auto line_idx = segments[it->second].line;
                if (seg_of_lines[line_idx] == -1)
                    seg_of_lines[line_idx] = it->second;
                else {
                    auto old_dist = (segments[seg_of_lines[line_idx]].centroid - point).length();
                    auto new_dist = (segments[it->second].centroid - point).length();
                    if (new_dist < old_dist)
                        seg_of_lines[line_idx] = it->second;
                }
            }
            for (int i = 0; i < n_streamlines; i++)
                if (seg_of_lines[i] >= 0)
                    results.emplace_back(seg_of_lines[i]);
            delete[] seg_of_lines;
        } else {
            while (it != res.end()) {
                results.push_back(it->second);
                it++;
            }
        }
	}


};

void arrangement(int n_buckets, int n_tuple, int* buckets) {
	float* x = new float[n_buckets];

}
vector<HashTable> hashtables;
constexpr int funcpool_size = 32, L = 5, K = 4, TABLESIZE = 100;
int main() {

	LoadWaveFrontObject("d:/flow_data/tornado.obj");
	//FILEIO::normalize();
	FILEIO::toFStreamlines();
	decomposeByCurvature(2*M_PI, 1000.f);
	pair<vector<LshFunc>*, int> funcs = LshFunc::create(funcpool_size, segments.data(), segments.size(), 5);

	//stocastic table construction
	uniform_int_distribution<int> uni_dist{ 0, funcpool_size - 1};
	for (int l = 0; l < L; l++) {
		
		vector<int> func_for_table;
		for (int k = 0; k < K; k++) 
			func_for_table.push_back(uni_dist(engine));

		hashtables.push_back(HashTable(func_for_table, TABLESIZE, funcs.first, segments.data(), segments.size()));

	}

	return 0;
}