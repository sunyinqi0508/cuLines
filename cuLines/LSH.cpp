//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
#include "LSH.h"
#include "FileIO.h"
#include "Vector.h"
#include "Common.h"
#include "cuStubs.cuh"
#include "Segmentation.h"
#include "Parameters.h"
#include "Stopwatch.h"
#include "Range.h"

#include <stdint.h>

#include <unordered_map>
#include <map>
#include <set>
#include <list>
#include <random>
#include <algorithm>
#include <iostream>
#include <future>
#include <fstream>

using namespace std;
using namespace FileIO;
typedef Vector3 Vector;

class LshFunc {

protected:
	Vector3 a_;
	float b_, w_;
	int _n_buckets;

	vector<int32_t> *buckets;
	unsigned char* _h;
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
		std::uniform_int_distribution<int> uni_intdist{0, (1<<29)};
		for (int funcidx : LSHFunctions) {
			r1.push_back(uni_intdist(engine));
			r2.push_back(uni_intdist(engine));
		}
		for (int i = 0; i < n_samples; i++) {//i -> ptr to segment
			
			int64_t fingerprint1 = 0, fingerprint2 = 0;
			for (int j = 0; j < LSHFunctions.size(); j++) {

				int64_t tmp_fp1 = r1[j] * (*function_pool)[LSHFunctions[j]].h(i);
				int64_t tmp_fp2 = r2[j] * (*function_pool)[LSHFunctions[j]].h(i);
				tmp_fp1 = 5 * (tmp_fp1 >> 32) + (tmp_fp1 & 0xffffffff);
				tmp_fp2 = 5 * (tmp_fp2 >> 32) + (tmp_fp2 & 0xffffffff);

				fingerprint1 += (tmp_fp1 >> 32) ? (tmp_fp1 - Prime) : tmp_fp1;
				fingerprint2 += (tmp_fp2 >> 32) ? (tmp_fp2  -Prime) : tmp_fp2;

				fingerprint1 = (fingerprint1 >> 32) ? (fingerprint1 - Prime) : fingerprint1;
				fingerprint2 = (fingerprint2 >> 32) ? (fingerprint2 - Prime) : fingerprint2;

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
	void Query(vector<int>& results, Vector3 point, bool from_distinct_lines = true, unordered_map<int,int> *res = 0) {

		int64_t fingerprint1 = 0, fingerprint2 = 0;
		bool noreturn = false;
		if (!res)
		{
			res = new unordered_map<int, int>();
			noreturn = true;
		}

		for (int j = 0; j < LSHFunctions.size(); j++) {
			
			const LshFunc curr_func = (*function_pool)[LSHFunctions[j]];
			const int n_buckets = curr_func.get_n_buckets();
			int64_t tmp_fp1 = r1[j] * curr_func(point);
			int64_t tmp_fp2 = r2[j] * curr_func(point);
			tmp_fp1 = 5 * (tmp_fp1 >> 32) + (tmp_fp1 & 0xffffffff);
			tmp_fp2 = 5 * (tmp_fp2 >> 32) + (tmp_fp2 & 0xffffffff);

			fingerprint1 += (tmp_fp1 >> 32) ? (tmp_fp1 - Prime) : tmp_fp1;
			fingerprint2 += (tmp_fp2 >> 32) ? (tmp_fp2 - Prime) : tmp_fp2;

			fingerprint1 = (fingerprint1 >> 32) ? (fingerprint1 - Prime) : fingerprint1;
			fingerprint2 = (fingerprint2 >> 32) ? (fingerprint2 - Prime) : fingerprint2;
		}
		fingerprint1 %= tablesize;
		fingerprint2 %= Prime;
		if(lshTable[fingerprint1].find(fingerprint2)!= lshTable[fingerprint1].end())
			for (int resultint_pt : lshTable[fingerprint1][fingerprint2]->ptr_segments) {
				const Segment this_seg = segments[resultint_pt];
				unordered_map<int, int>::iterator findings = res->find(this_seg.line);
			
				if (findings != res->end()) {
					int& curr_seg = findings->second;
					if ((point - this_seg.centroid).length() < (point - segments[curr_seg].centroid).length()) {
						curr_seg = resultint_pt;
					}
				}
				else
					(*res)[this_seg.line] = resultint_pt;
			}
		unordered_map<int, int>::iterator it = res->begin();

		if (from_distinct_lines) {
			auto seg_of_lines = new int[n_streamlines];
			std::fill(seg_of_lines, seg_of_lines + n_streamlines, -1);
			while (it != res->end()) {
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
			if (noreturn)
			{
				while (it != res->end()) {
					results.push_back(it->second);
					it++;
				}
				delete res;
			}
		}
	}


};

template <typename S, typename T>
bool compareFirstOnly(const pair<S, T> &lhs, const pair<S, T> &rhs) {
	return lhs.first < rhs.first;
}

template <typename S, typename T>
void sortByFirst(vector<pair<S, T>> &c) {
	sort(begin(c), end(c), compareFirstOnly<S, T>);
}

inline int getGlobalIndex(int line_idx, int pt_idx) {
	return &f_streamlines[line_idx][pt_idx] - &f_streamlines[0][0];
}

struct Point {
	int line_idx, point_idx, global_idx;

	Point(int line_index, int point_index) :
		line_idx{ line_index },
		point_idx{ point_idx },
		global_idx{ getGlobalIndex(line_index, point_index) }
	{}
};

vector<pair<float, int>> queryGroundTruth(Vector3 pt) {
	vector<pair<float, int>> result;
	// for each lines
	for (int i = 0; i < n_streamlines; i++) {
		auto nearest_dist = numeric_limits<float>::infinity();
		Vector *nearest_vec = nullptr;
		auto n = static_cast<int>(streamlines[i].size());
		for (int j = 0; j < n; j++) {
			auto dist = f_streamlines[i][j].distance(pt);
			if (dist < nearest_dist) {
				nearest_vec = &f_streamlines[i][j];
				nearest_dist = dist;
			}
		}
		if (nearest_vec) {
			auto pt_glob_idx = nearest_vec - &f_streamlines[0][0];
			result.emplace_back(nearest_dist, pt_glob_idx);
		}
	}
	sort(begin(result), end(result), compareFirstOnly<float, int>);
	return result;
}

void segmentIndexToLineIndex(vector<int> &indexes) {
	for (auto &idx : indexes)
		idx = static_cast<int>(segments[idx].line);
}

vector<int> querySegments(vector<HashTable> &hts, Vector3 pt) {
	// do first level query
	vector<int> query_result;
	auto seg_of_lines = new int[n_streamlines];
	std::fill(seg_of_lines, seg_of_lines + n_streamlines, -1);
	// merge query result from all hash tables
	for (auto &ht : hts) {
		query_result.clear();
		ht.Query(query_result, pt, false);
		for (auto seg_idx : query_result) {
			// line index of this segment
			auto line_idx = segments[seg_idx].line;
			// if this line has not been recorded
			if (seg_of_lines[line_idx] == -1)
				seg_of_lines[line_idx] = seg_idx;
			else {
				// test if nearer
				auto old_dist = (segments[seg_of_lines[line_idx]].centroid - pt).length();
				auto new_dist = (segments[seg_idx].centroid - pt).length();
				if (new_dist < old_dist)
					seg_of_lines[line_idx] = seg_idx;
			}
		}
	}
	query_result.clear();
	for (int i = 0; i < n_streamlines; i++)
		if (seg_of_lines[i] >= 0)
			query_result.emplace_back(i);
	delete[] seg_of_lines;
	return query_result;
}

// 

vector<pair<float, int>> queryANN(vector<HashTable> &hts, Vector3 pt) {
	// do first level query
	auto query_result = querySegments(hts, pt);
	// do second level query
	vector<pair<float, int>> result;
	for (auto seg_idx : query_result) {
		auto nearest = second_level[seg_idx].nearest(pt);
		auto nearest_glob_idx = getGlobalIndex(segments[seg_idx].line, nearest);
		auto nearest_dist = f_streamlines[0][nearest_glob_idx].distance(pt);
		result.emplace_back(nearest_dist, nearest_glob_idx);
	}
	sort(begin(result), end(result), compareFirstOnly<float, int>);
	return result;
}

float evaluateError(
	const vector<pair<float, int>> &groundtruth,
	const vector<pair<float, int>> &ann_result
) {
	auto n = min(groundtruth.size(), ann_result.size());
	auto errorSum = 0.f;
	for (size_t i = 0; i < n; i++) {
		auto dist_gt = groundtruth[i].first;
		auto dist_ann = ann_result[i].first;
		errorSum += fabsf(dist_gt - dist_ann);
	}
	return errorSum / static_cast<float>(n);
}

void arrangement(int n_buckets, int n_tuple, int* buckets) {
	float* x = new float[n_buckets];

}
vector<HashTable> hashtables;

// do application on critical points query

enum class CriticalPointType : int {
	RepelNode = 0x0,
	RepelFocus = 0x4,
	RepelNodeSaddle = 0x1,
	RepelFocusSaddle = 0x5,
	AttractNodeSaddle = 0x3,
	AttractFocusSaddle = 0x7,
	AttractNode = 0x2,
	AttractFocus = 0x6
};

struct CriticalPoint : public Vector3 {
	CriticalPointType type;
	float scale;
};

template <typename T>
inline void readRaw(std::istream &in, T &x) {
	in.read(reinterpret_cast<char*>(&x), sizeof(T));
}

std::vector<CriticalPoint> loadCriticalPoints(const char *filename) {
	std::ifstream inputFile{ filename, std::ifstream::binary };
	uint32_t n_points;
	readRaw(inputFile, n_points);
	std::vector<CriticalPoint> pts{ n_points };
	for (int i = 0; i < n_points; i++)
		readRaw<Vector3>(inputFile, pts[i]);
	for (int i = 0; i < n_points; i++)
		readRaw(inputFile, pts[i].type);
	for (int i = 0; i < n_points; i++)
		readRaw(inputFile, pts[i].scale);
	return pts;
}

vector<int> queryCriticalPoints(const vector<CriticalPoint> &query_points, vector<HashTable> &hashTables) {
	std::vector<bool> mark;
	mark.assign(n_streamlines, false);
	int total_marked = 0;
	for (auto &p : query_points) {
		auto query_result = querySegments(hashtables, p);
		segmentIndexToLineIndex(query_result);
		for (auto line_idx : query_result) {
			if (!mark[line_idx]) {
				mark[line_idx] = true;
				total_marked++;
			}
		}
	}
	std::vector<int> res;
	res.reserve(total_marked);
	for (int i = 0; i < n_streamlines; i++)
		if (mark[i])
			res.emplace_back(i);
	return res;
}

// application: do spectral clustering

float computeSimilarity(int p, int q, int windowRadius = 20) {
	// the middle point index of segment p and segment q
	auto p_middle = (segments[p].end + segments[p].begin) / 2;
	auto q_middle = (segments[q].end + segments[q].begin) / 2;
	// range of the window
	Range<int> window{ -windowRadius, +windowRadius };
	auto p_range = window + p_middle;
	auto q_range = window + q_middle;
	// clamp the window in the boundary of segment
	p_range.clampBy({ segments[p].begin, segments[p].end - 1 });
	q_range.clampBy({ segments[q].begin, segments[q].end - 1 });
	// find the common part
	auto common = (p_range -= p_middle).intersect(q_range -= q_middle);
	auto sum = 0.f;
	auto p_curve = f_streamlines[segments[p].line];
	auto q_curve = f_streamlines[segments[q].line];
	for (int i = common.left; i <= common.right; i++) {
		auto d = p_curve[p_middle + i].distance(q_curve[q_middle + i]);
		sum += d * d;
	}
	return sum / static_cast<float>(common.length());
}

void dumpAffinityMatrix(ostream &out, vector<HashTable> &hashTables) {
	// foreach centroid of segments
	out << segments.size() << '\n';
	for (size_t i = 0; i < segments.size(); i++) {
		auto ann_segments = querySegments(hashTables, segments[i].centroid);
		for (auto seg_idx : ann_segments) {
			out << i << ' '
				<< seg_idx << ' '
				<< computeSimilarity(i, seg_idx) << '\n';
		}
	}
}

float computeSimilarity_pt(int p, int q, int windowRadius = 20) {

	Range<int> window{ -windowRadius, +windowRadius };

	auto p_range = window + p;

	auto q_range = window + q;

	const int p_streamline = Streamline::getlineof(p);
	const int q_streamline = Streamline::getlineof(q);
	p_range.clampBy({Streamline::offsets[p_streamline], Streamline::offsets[p_streamline + 1]});

	q_range.clampBy({ Streamline::offsets[q_streamline], Streamline::offsets[q_streamline + 1] });

	auto common = (p_range -= p).intersect(q_range -= q);
	auto sum = 0.f;


	const float center_dist = f_streamlines[0][p] - f_streamlines[0][q];
	for (int i = common.left; i <= common.right; i++) {

		auto d = f_streamlines[0][p + i].distance(f_streamlines[0][q + i]);
		sum += square(d - center_dist);

	}

	return sum / static_cast<float>(common.length());

}
float *alpha;

void initialize(const char* filename) {
	// load file
	LoadWaveFrontObject(filename);
	FileIO::normalize(1);
	FileIO::toFStreamlines();
	// decomposition
  	decomposeByCurvature(2 * M_PI, 1000.f);
	// second level
	initializeSecondLevel();
	// output debug info
	cout << n_points << " points, " << n_streamlines << " stream lines and " << segments.size() << " segments\n";
	// hash function pool
	auto funcs = LshFunc::create(funcpool_size, segments.data(), segments.size(), 5);
	// create the hash tables stocastically
	uniform_int_distribution<int> uni_dist{ 0, funcpool_size - 1};
	for (int l = 0; l < L; l++) {
		vector<int> func_for_table;
		for (int k = 0; k < K; k++) 
			func_for_table.push_back(uni_dist(engine));
		hashtables.push_back(HashTable(func_for_table, TABLESIZE, funcs.first, segments.data(), segments.size()));
	}
	vector<int> _res;
	alpha = new float[n_points];
	int max_alpha = -1;
	for (int i = 0; i < n_points; i++) {
		/*for (auto table : hashtables) {
			unordered_map<int, int> *results = new unordered_map<int, int>();
			table.Query(_res,f_streamlines[0][i],false, results);
		}*/
		vector<pair<float, int>> result = queryANN(hashtables, f_streamlines[0][i]);
		int cnt_res = 0;
		float sum_gx = 0;
		alpha[i] = 0;
		for (auto res_q : result) {
			float gx = gaussianDist(res_q.first);
			alpha[i] += computeSimilarity_pt(i, res_q.second);
			sum_gx += gx;
		}
		if (sum_gx)
			alpha[i] /= sum_gx;
		else
			alpha[i] = 0;
		max_alpha = __macro_max(alpha[i], max_alpha);
	}
	for (int i = 0; i < n_points; i++)
		alpha[i] /= max_alpha;
}

void doCriticalPointQuery(const char *cp_filename) {
	auto pts = loadCriticalPoints(cp_filename);
	auto curves = queryCriticalPoints(pts, hashtables);
	alpha = new float[n_points];
	std::fill(alpha, alpha + n_points, 0.f);
	cout << "doCriticalPointQuery\n";
	cout << curves.size() << '/' << n_streamlines << " found\n";
	for (auto line_idx : curves) {
		for (int i = getGlobalIndex(line_idx, 0); i < streamlines[line_idx].size(); i++)
			alpha[i] = 1.f;
	}
}

int main() {
	 initialize("e:/flow_data/tornado.obj");

	 return 0;
}