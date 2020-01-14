#ifndef _CULINES_H
#define _CULINES_H
#include "../include/Parameters.h"
#include "../include/Common.h"
#include "../include/Vector.h"
#include "../include/FileIO.h"
#include "../include/Seeding3D.h"
#include "Segmentation.h"
#include <random>
#include <unordered_map>
#include <map>
#define map_t std::unordered_map
#define map_buckets_t std::unordered_map
extern Vector3 *tangents;
extern std::normal_distribution<float> gauss_dist;
extern std::uniform_real_distribution<float> uni_dist;

class VectorField;
class LshFunc {

protected:

	Vector3 a_;
	float b_, w_;
	float lsh_func_offset;
	int _n_buckets;
	//vector<int32_t> *buckets;
	map_buckets_t<int, std::vector<int>> buckets;
	unsigned char* _h;
public:
	friend std::vector<LshFunc> create_seeding(int n_funcs, VectorField& field);
	LshFunc() = default;
	LshFunc(Vector3 a, float b, float w) : a_{ a }, b_{ b }, w_{ w } {}
	inline unsigned char& h(int i) {
		return _h[i];
	}
	inline int operator() (const Vector3 &x) const {
		return static_cast<int>(__macro_bound(
			(x.dot(a_) + b_) / w_ - lsh_func_offset, 0, _n_buckets - 1
		));
	}

	inline int operator() (Segment *seg) const {
		return (*this)(centroids[seg->id]);
	}

	inline const int& get_n_buckets() const { return _n_buckets; }
	~LshFunc() {
	}
	inline static float gen_lshParamsR1() {
		float x1 = 0;
		do {
			x1 = uni_dist(engine);
		} while (x1 == 0);
		return sqrt(-2.0 * log(x1)) * cos(2.0 * M_PI * uni_dist(engine));
	}
	void deinit() {
		//delete[] buckets;
		delete[] _h;
	}

	static std::pair<std::vector<LshFunc>*, int> create(int n, Segment* samples, size_t n_samples, int n_buckets) {




		std::vector<LshFunc>* fn = new std::vector<LshFunc>;

		//float *projections = new float[n_samples];
		int bestfunc = -1;
		float min_variation = std::numeric_limits<float>::max();
		std::uniform_real_distribution<float> uni_dist2{ 0, 4 };

		for (int i = 0; i < n; i++)
		{
			LshFunc curr_func;
			///curr_func.buckets = new vector<int32_t>[n_buckets];
			curr_func._h = new unsigned char[n_samples];
			fn->push_back(curr_func);
			float variation = 0;
			do {
				variation = 0;
				//for (int j = 0; j < n_buckets; j++)
				(*fn)[i].buckets.clear();
				(*fn)[i].a_ = Vector3(gen_lshParamsR1(), gen_lshParamsR1(), gen_lshParamsR1());
				(*fn)[i].w_ = 4;
				(*fn)[i].b_ = uni_dist2(engine);

				float _min = std::numeric_limits<float>::max(), _interval = std::numeric_limits<float> ::min();
				for (int j = 0; j < n_samples; j++) {
					//projections[j]
					float projectionj = ((*fn)[i].a_.dot(centroids[samples[j].id]) + (*fn)[i].b_) / (*fn)[i].w_;
					_min = __macro_min(_min, projectionj);
					_interval = __macro_max(_interval, projectionj);
				}

				_interval -= _min;
				(*fn)[i]._n_buckets = _interval + 1;// / (*fn)[i].w_;
				(*fn)[i].lsh_func_offset = _min;
				//const float _div = _interval / n_buckets;

				for (int j = 0; j < n_samples; j++)
					(*fn)[i].buckets[(*fn)[i](samples + j)].push_back(j);
				const float avg_size = (float)n_samples / (float)(*fn)[i]._n_buckets;
				for (int j = 0; j < (*fn)[i]._n_buckets; j++)
					if ((*fn)[i].buckets[j].size() < avg_size)
						variation += avg_size - (*fn)[i].buckets[j].size();//l2 norm might be better choice;
				static int hit = 0;
				hit++;
			} while (0);// variation > (n_samples) / 3.f );//Todo: determining efficiency of hash function

			if (variation < min_variation) {
				min_variation = variation;
				bestfunc = i;
			}

			for (int j = 0; j < (*fn)[i]._n_buckets; j++)
				for (int _sample : ((*fn)[i].buckets[j]))
				{
					//projections[_sample] = j;
					(*fn)[i]._h[_sample] = j;
				}
		}

		//delete[]projections;
		return make_pair(fn, bestfunc);
	}
};
//constexpr bool _contraction = false;
class HashTable {
public:

	struct LSHPoint {
		std::vector<int> ptr_segments;
		LSHPoint(int ptr_segment) :
			ptr_segments() {
			ptr_segments.push_back(ptr_segment);
		}
	};

	int tablesize;
	std::vector<int> LSHFunctions;//indices of lsh functions
	std::vector<int> r1, r2;
	std::vector<LshFunc> *function_pool;

	map_buckets_t<std::int64_t, LSHPoint *>* lshTable = 0;
	HashTable(std::vector<int> LSHFunctions, std::vector<LshFunc> *function_pool, int tablesize)
		: LSHFunctions (LSHFunctions), tablesize(tablesize), function_pool(function_pool){
		lshTable = new map_buckets_t<std::int64_t, LSHPoint*>[tablesize];

		std::uniform_int_distribution<int> uni_intdist{ 0, (1 << 29) };
		for (int funcidx : LSHFunctions) {
			r1.push_back(uni_intdist(engine));
			r2.push_back(uni_intdist(engine));
		}

	}
	HashTable(std::vector<int>LSHFunctions, int tablesize, std::vector<LshFunc> *function_pool, Segment *samples, int n_samples)
		: LSHFunctions(LSHFunctions), tablesize(tablesize), function_pool(function_pool)
	{
		
		lshTable = new map_buckets_t<std::int64_t, LSHPoint*>[tablesize];
		std::uniform_int_distribution<int> uni_intdist{ 0, (1 << 29) };
		for (int funcidx : LSHFunctions) {
			r1.push_back(uni_intdist(engine));
			r2.push_back(uni_intdist(engine));
		}
		for (int i = 0; i < n_samples; i++) {//i -> ptr to segment

			std::int64_t fingerprint1 = 0, fingerprint2 = 0;
			for (int j = 0; j < LSHFunctions.size(); j++) {

				std::int64_t tmp_fp1 = r1[j] * (std::int64_t)(*function_pool)[LSHFunctions[j]].h(i);
				std::int64_t tmp_fp2 = r2[j] * (std::int64_t)(*function_pool)[LSHFunctions[j]].h(i);
				tmp_fp1 = tmp_fp1 % tablesize;// 5 * (tmp_fp1 >> 32) + (tmp_fp1 & 0xffffffff);
				tmp_fp2 = tmp_fp2 % Prime;// 5 * (tmp_fp2 >> 32) + (tmp_fp2 & 0xffffffff);

				fingerprint1 += tmp_fp1;// (tmp_fp1 >> 32) ? (tmp_fp1 - Prime) : tmp_fp1;
				fingerprint2 += tmp_fp2;// (tmp_fp2 >> 32) ? (tmp_fp2 - Prime) : tmp_fp2;

				fingerprint1 %= tablesize;// (fingerprint1 >> 32) ? (fingerprint1 - Prime) : fingerprint1;
				fingerprint2 %= Prime;//(fingerprint2 >> 32) ? (fingerprint2 - Prime) : fingerprint2;

			}

			fingerprint1 %= tablesize;
			fingerprint2 %= Prime;
			if (lshTable[fingerprint1].find(fingerprint2) == lshTable[fingerprint1].end()) {
				lshTable[fingerprint1][fingerprint2] = new LSHPoint(i);
			}
			else {
				lshTable[fingerprint1][fingerprint2]->ptr_segments.emplace_back(i);
			}

		}
	}
	void append(Vector3 sample, int identifier) {
		std::int64_t fingerprint1 = 0, fingerprint2 = 0;
		for (int j = 0; j < LSHFunctions.size(); j++) {

			std::int64_t tmp_fp1 = r1[j] * (std::int64_t)(*function_pool)[LSHFunctions[j]](sample);
			std::int64_t tmp_fp2 = r2[j] * (std::int64_t)(*function_pool)[LSHFunctions[j]](sample);
			tmp_fp1 = tmp_fp1 % tablesize;// 5 * (tmp_fp1 >> 32) + (tmp_fp1 & 0xffffffff);
			tmp_fp2 = tmp_fp2 % Prime;// 5 * (tmp_fp2 >> 32) + (tmp_fp2 & 0xffffffff);

			fingerprint1 += tmp_fp1;// (tmp_fp1 >> 32) ? (tmp_fp1 - Prime) : tmp_fp1;
			fingerprint2 += tmp_fp2;// (tmp_fp2 >> 32) ? (tmp_fp2 - Prime) : tmp_fp2;

			fingerprint1 %= tablesize;// (fingerprint1 >> 32) ? (fingerprint1 - Prime) : fingerprint1;
			fingerprint2 %= Prime;//(fingerprint2 >> 32) ? (fingerprint2 - Prime) : fingerprint2;

		}

		fingerprint1 %= tablesize;
		fingerprint2 %= Prime;
		if (lshTable[fingerprint1].find(fingerprint2) == lshTable[fingerprint1].end()) {
			lshTable[fingerprint1][fingerprint2] = new LSHPoint(identifier);
		}
		else {
			lshTable[fingerprint1][fingerprint2]->ptr_segments.emplace_back(identifier);
		}
	}
	//~HashTable() = default;
	void Query(std::vector<int>& results, int *pagetable, Vector3 point, const int* seg2line, int n_identifiers) {

		std::int64_t fingerprint1 = 0, fingerprint2 = 0;
		for (int j = 0; j < LSHFunctions.size(); j++) {

			std::int64_t tmp_fp1 = r1[j] * (std::int64_t)(*function_pool)[LSHFunctions[j]](point);
			std::int64_t tmp_fp2 = r2[j] * (std::int64_t)(*function_pool)[LSHFunctions[j]](point);
			tmp_fp1 = tmp_fp1 % tablesize;// 5 * (tmp_fp1 >> 32) + (tmp_fp1 & 0xffffffff);
			tmp_fp2 = tmp_fp2 % Prime;// 5 * (tmp_fp2 >> 32) + (tmp_fp2 & 0xffffffff);

			fingerprint1 += tmp_fp1;// (tmp_fp1 >> 32) ? (tmp_fp1 - Prime) : tmp_fp1;
			fingerprint2 += tmp_fp2;// (tmp_fp2 >> 32) ? (tmp_fp2 - Prime) : tmp_fp2;

			fingerprint1 %= tablesize;// (fingerprint1 >> 32) ? (fingerprint1 - Prime) : fingerprint1;
			fingerprint2 %= Prime;//(fingerprint2 >> 32) ? (fingerprint2 - Prime) : fingerprint2;

		}

		fingerprint1 %= tablesize;
		fingerprint2 %= Prime;
		auto result = lshTable[fingerprint1].find(fingerprint2);
		if ( result != lshTable[fingerprint1].end()) {
			auto resulting_segments = result->second->ptr_segments;
			for (const int& this_segment : resulting_segments)
			{
				int identifier = segments[this_segment].line;
				float dist = point.sqDist(centroids[this_segment]);
				if (dist <= 1.f)
				{
					if (pagetable[identifier] < 0 || point.sqDist(centroids[pagetable[identifier]]) > dist) {
						pagetable[identifier] = this_segment;
					}
				}
			}
		}

		for (int i = 0; i < n_identifiers; ++i)
		{
			if (pagetable[i] > 0)
			{
				results.emplace_back(pagetable[i]);
				pagetable[i] = -1;
			}
		}

	}
	template<bool _contraction = true>
	void Query(std::vector<int>& results, Vector3 point, int line = -1, bool from_distinct_lines = true, map_t<int, unsigned char> *res = 0, const float angle = -2.f, int seg_num = -1, int ptnum = -1, int* pagetable = 0, int *elements = 0) {

		std::int64_t fingerprint1 = 0, fingerprint2 = 0;
		bool noreturn = false;
		if (!res)
		{
			res = new map_t<int, unsigned char>();
			noreturn = true;
		}


		for (int j = 0; j < LSHFunctions.size(); j++) {

			const LshFunc curr_func = (*function_pool)[LSHFunctions[j]];
			const int n_buckets = curr_func.get_n_buckets();
			std::int64_t tmp_fp1 = r1[j] * (std::int64_t)curr_func(point);
			std::int64_t tmp_fp2 = r2[j] * (std::int64_t)curr_func(point);
			tmp_fp1 = tmp_fp1 % tablesize;// 5 * (tmp_fp1 >> 32) + (tmp_fp1 & 0xffffffff);
			tmp_fp2 = tmp_fp2 % Prime;// 5 * (tmp_fp2 >> 32) + (tmp_fp2 & 0xffffffff);

			fingerprint1 += tmp_fp1;// (tmp_fp1 >> 32) ? (tmp_fp1 - Prime) : tmp_fp1;
			fingerprint2 += tmp_fp2;// (tmp_fp2 >> 32) ? (tmp_fp2 - Prime) : tmp_fp2;

			fingerprint1 %= tablesize;// (fingerprint1 >> 32) ? (fingerprint1 - Prime) : fingerprint1;
			fingerprint2 %= Prime;//(fingerprint2 >> 32) ? (fingerprint2 - Prime) : fingerprint2;
		}
		fingerprint1 %= tablesize;
		fingerprint2 %= Prime;
		if (lshTable[fingerprint1].find(fingerprint2) != lshTable[fingerprint1].end())
			for (const int &resulting_seg : lshTable[fingerprint1][fingerprint2]->ptr_segments) {

				const Segment& this_seg = segments[resulting_seg];
				const int& _uniqueness_idx = allow_segs_on_same_line ? resulting_seg : this_seg.line;
				if constexpr (!allow_segs_on_same_line)
					if (this_seg.line == line)
						continue;
				if (pagetable[_uniqueness_idx] >= 0)
				{
					//const int seg = pagetable[_uniqueness_idx];// &0x3ffff;
					const float dist = point.sqDist(FileIO::f_streamlines[_uniqueness_idx][pagetable[_uniqueness_idx]]);// (pagetable[_uniqueness_idx] >> 18) / 8192.f;
																												//if (seg != resulting_seg// &&
																												//dir[resulting_seg].dot(dir[seg_num]) >= angle
																												//)
					{
						//const float this_dist = point.sqDist(centroids[resulting_seg]);
						const int nearest = second_level[resulting_seg].nearest(point);
						auto nearest_dist = (FileIO::f_streamlines[_uniqueness_idx][nearest].sqDist(point));
						const int g_idx = FileIO::getGlobalIndex(_uniqueness_idx, nearest);
						if (nearest_dist < dist && fabs(tangents[ptnum].dot(tangents[g_idx])) >= 0.98)
							pagetable[_uniqueness_idx] = nearest;// | (((int)(this_dist * 8192)) << 18);
					}
				}
				else
					//if (dir[resulting_seg].dot(dir[seg_num]) >= angle)
				{
					//const float this_dist = point.sqDist(centroids[resulting_seg]);
					const int nearest = second_level[resulting_seg].nearest(point);
					auto nearest_dist = (FileIO::f_streamlines[_uniqueness_idx][nearest].sqDist(point));
					//if (this_dist <= 1.f)
					;
					const int g_idx = FileIO::getGlobalIndex(_uniqueness_idx, nearest);

					if (nearest_dist <= .2527f&& fabs(tangents[ptnum].dot(tangents[g_idx])) >= 0.98)
					{
						pagetable[_uniqueness_idx] = nearest;// | (((int)(this_dist * 8192)) << 18);
															 //(*elements)++;
					}
				}

#if define(LAGACY)
				map_t<int, int>::iterator findings = res->find(_uniqueness_idx);

				if (findings != res->end()) {

					int& curr_seg = findings->second;

					if (curr_seg != resulting_seg && (point - centroids[this_seg.id]).sq() < (point - centroids[segments[curr_seg].id]).sq())
						curr_seg = resulting_seg;
				}
				else {
					if ((point - centroids[this_seg.id]).sq() < 1.5f)//enforce radius of 1
					{
						if constexpr (_contraction) {
							if (dir[this_seg.id].dot(dir[segments[seg_num].id]) < angle)
								(*res)[_uniqueness_idx] = resulting_seg;
						}
						else
							(*res)[_uniqueness_idx] = resulting_seg;
					}
				}
#endif
			}

		map_t<int, unsigned char>::iterator it = res->begin();
		if (noreturn)
		{
			while (it != res->end()) {
				results.push_back(it->second);
				it++;
			}
			delete res;
		}
	}
	void Query(std::vector<int>& results, Vector3 point, int line = -1, bool from_distinct_lines = true, map_t<int, unsigned char> *res = 0) {
		Query<false>(results, point, line, from_distinct_lines, res);
	}

};

void initialize(const char* filename, int reduced = 0, LSH_Application applicaions = LSH_None, float Radius = .005f);
void doCriticalPointQuery(const char *cp_filename);

extern float* alpha;
#endif
