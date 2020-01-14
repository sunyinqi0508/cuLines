//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
#include "LSH.h"
#include "FileIO.h"
#include "Vector.h"
#include "Common.h"
#include "cuProxy.cuh"
#include "Segmentation.h"
#include "Parameters.h"
#include "Stopwatch.h"
#include "Range.h"

#include <windows.h>
#undef max
#undef min

#include <omp.h>
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
//typedef Vector3 Vector;
normal_distribution<float> gauss_dist{ 0.f, 1.f };
uniform_real_distribution<float> uni_dist{ 0.f, 1.f };

float __global_radius = 0;
const float D = 172;
Vector3 *tangents;

template <typename S, typename T>
bool compareFirstOnly(const pair<S, T> &lhs, const pair<S, T> &rhs) {
	return lhs.first < rhs.first;
}

template <typename S, typename T>
void sortByFirst(vector<pair<S, T>> &c) {
	sort(begin(c), end(c), compareFirstOnly<S, T>);
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
		Vector<> *nearest_vec = nullptr;
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

vector<int> querySegments(vector<HashTable> &hts, Vector3 pt, int line = -1) {
	// do first level query
	vector<int> query_result;
	int _size = allow_segs_on_same_line ? segments.size() : n_streamlines;

	auto seg_of_lines = new int[_size];
	std::fill(seg_of_lines, seg_of_lines + _size, -1);
	// merge query result from all hash tables
	for (auto &ht : hts) {
		query_result.clear();
		ht.Query(query_result, pt, line, false);
		for (auto seg_idx : query_result) {
			// line index of this segment
			// if this line has not been recorded
			auto _idx = allow_segs_on_same_line ? seg_idx : segments[seg_idx].line;

			if (seg_of_lines[_idx] == -1)
				seg_of_lines[_idx] = seg_idx;
			else {
				// test if nearer
				if constexpr(allow_segs_on_same_line) {
					seg_of_lines[_idx] = seg_idx;
				}
				else {
					auto old_dist = (centroids[segments[seg_of_lines[_idx]].id] - pt).length();
					auto new_dist = (centroids[segments[_idx].id] - pt).length();
					if (new_dist < old_dist)
						seg_of_lines[_idx] = _idx;
				}
			}
		}
	}
	query_result.empty();
	for (int i = 0; i < _size; i++)
		if (seg_of_lines[i] >= 0)
			query_result.emplace_back(i);
	delete[] seg_of_lines;
	return query_result;
}

// 

vector<pair<float, int>> queryANN(vector<HashTable> &hts, Vector3 pt, int line = -1) {
	map_t<int, unsigned char> *_map = new map_t<int, unsigned char>;
	// do first level query
	//auto query_result = //querySegments(hts, pt, line);
	vector <int> tmp;
	for (HashTable table : hts) {
		table.Query(tmp, pt, line, false, _map);
	}


	// do second level query
	vector<pair<float, int>> result;
	for (auto seg_idx : *_map) {
		auto nearest = second_level[seg_idx.second].nearest(pt);
		int this_line;
		if constexpr(allow_segs_on_same_line)
			this_line = segments[seg_idx.second].line;
		else
			this_line = seg_idx.first;
		int nearest_glob_idx = getGlobalIndex(this_line, nearest);
		auto nearest_dist = f_streamlines[0][nearest_glob_idx].distance(pt);
		result.emplace_back(nearest_dist, nearest_glob_idx);
	}
	//sort(begin(result), end(result), compareFirstOnly<float, int>);
	delete _map;
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
		auto ann_segments = querySegments(hashTables, centroids[segments[i].id]);
		for (auto seg_idx : ann_segments) {
			out << i << ' '
				<< seg_idx << ' '
				<< computeSimilarity(i, seg_idx) << '\n';
		}
	}
}

float computeSimilarity_pt(int p, int q, int windowRadius = 1) {

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
	if (isnan(center_dist)||isinf(center_dist))
		printf("%d ", center_dist);
	for (int i = common.left; i < common.right; i++) {

		auto d = f_streamlines[0][p + i].distance(f_streamlines[0][q + i]);
		if (!(isnan(d) || isinf(d)))
			sum += square(d - center_dist);
		else
			printf("err %d %d %d\n", p, q, i);
	}
	if (common.length())
		return  sum / static_cast<float>(common.length());
	else
		return 16;

}
float *alpha;
void benchmark(const char* filename) {
//	LoadWaveFrontObject("d:/flow_data/laminar.obj");
//
//	FileIO::normalize(2, false);
//
//	FileIO::toFStreamlines();
//	// decomposition
////	decomposeByCurvature(2 * M_PI, 10.f);
//	// second level
////	initializeSecondLevel();
//	// output debug info
//	cout << n_points << " points, " << n_streamlines << " stream lines and " << segments.size() << " segments\n";
//	// hash function pool
//	auto funcs = LshFunc::create(funcpool_size, f_streamlines[0], n_points, 5);
//	// create the hash tables stocastically
//	uniform_int_distribution<int> uni_dist{ 0, funcpool_size - 1 };
//	for (int l = 0; l < L; l++) {
//		vector<int> func_for_table;
//		for (int k = 0; k < K; k++)
//			func_for_table.push_back(uni_dist(engine));
//		hashtables.push_back(HashTable(func_for_table, TABLESIZE, funcs.first, f_streamlines[0], n_points));
//	}
//	unordered_map<int, int> *result = new unordered_map<int, int>();
//	int *fuck = new int[n_points];
//	for (int i = 0; i < n_points; i++) {
//		result->clear();
//		vector<int> dummy;
//		for (int l = 0; l < L; l++) {
//			hashtables[l].Query(dummy, f_streamlines[0][i], Streamline::getlineof(i), false, result);
//		}
//		printf("%d ", result->size());
//	}
//	vector<int> _res;
}

void cleanupthread(map_t<int, unsigned char>* results, Vector3 **f_streamline2, int **pagetable) {
	if(results)
		delete[] results;
	if (f_streamline2) {

		delete[] f_streamline2[0];
		delete[] f_streamline2;
	}
	if (pagetable)
	{
		for (int i = 0; i < omp_threads; i++)
			delete[] pagetable[i];
		delete[] pagetable;
	}

	for (auto& t : hashtables) {
		delete[] t.lshTable;
	}
	hashtables.clear();
	second_level.clear();
}
#define _Page_table 1
void contraction(const int SmoothingWindow = 1, const float sigma = 0.01f) {
//	alpha = new float[n_points];
	_getTangent(tangents);
	omp_set_num_threads(omp_threads);

#if _Page_table
	const unsigned int pagesize = 1 << 25;
	uint64_t*  result_idx = new uint64_t[n_points];
	unsigned int** i_results = new unsigned int *[256];
	unsigned char this_pt = 0;
	unsigned char avail_pt = 4;
	atomic<int> atomic_Counter = 0;
	unsigned int sum_count = 0;
	for (int i = 0; i < avail_pt; i++)
		i_results[i] = new unsigned int[pagesize];
#else
	atomic<unsigned int> sum_count = 0;

#endif
	map_t<int, unsigned char>* results = new map_t<int, unsigned char>[n_points];
//#pragma omp parallel
//#pragma omp  for
//	for (int i = 0; i < n_points; i++) {
//		results[i].reserve(256);
//	}
	int **pagetable = new int*[omp_threads];
//	int *nums = new int[omp_threads];
	for (int i = 0; i < omp_threads; ++i)
	{
		pagetable[i] = new int[n_streamlines];
		std::fill(pagetable[i], pagetable[i] + n_streamlines, -1);
	}
	int firstrun = -1;
#pragma omp parallel
#pragma omp  for  schedule(dynamic)// firstprivate(firstrun)
	for (int ii = 0; ii < n_points; ii++) {
		int i = ii;
		if(n_points>800)
		if (omp_get_thread_num() == 0 && (i % (int)(n_points / 800) == 0))
			printf("\r%d %d %c ", (int)(i / (n_points / 800)), omp_get_num_threads(), '%');
		int *this_pagetable = pagetable[omp_get_thread_num()];
		//if (firstrun < 0)
		//	firstrun = ii;
		//i = (i - firstrun) * 8 + omp_get_thread_num();
		/*if (i > n_points)
			return;*/
		/*for (auto table : hashtables) {
		unordered_map<int, int> *results = new unordered_map<int, int>();
		table.Query(_res,f_streamlines[0][i],false, results);
		}*/
		vector <int> tmp;
		//nums[omp_get_thread_num()] = 0;
		int count = 0;

		for (HashTable table : hashtables) {
			table.Query<>(tmp, f_streamlines[0][i], Streamline::getlineof(i), false, results + i, .97f, pt_to_segs[i], i, this_pagetable, &count);// , &nums[omp_get_thread_num()]);
		}
		count = 0;
		//results[i].reserve(nums[omp_get_thread_num()] *2);
		for (int j = 0; j < n_streamlines; ++j) //{
			if (this_pagetable[j] >= 0)
		//	{
		//		//this_pagetable[j] &= 0x3ffff;
		//		//const int nearest = second_level[this_pagetable[j]].nearest(f_streamlines[0][i]);
		//		//auto nearest_dist = (f_streamlines[j][nearest].sqDist(f_streamlines[0][i]));
		//		if (tangents[i].dot(tangents[getGlobalIndex(j, this_pagetable[j])]) >= 0.98f)//.2527) //r2 = sq(R/Radius)
		//		{
		//			//this_pagetable[j] = nearest;//results[i][j] = nearest;
					count++;
		//		}
		//		else
		//			this_pagetable[j] = -1;
		//	}
		//}
#if _Page_table
		bool page_created = false;
		unsigned char this_end_pt= 0;
		unsigned int this_count_thread = 0, last_count_thread = 0;
		atomic_Counter += count;
#pragma omp critical
		{
			last_count_thread = sum_count;
			sum_count += count;
			if (sum_count >= pagesize ) {
				
				page_created = true;
				sum_count -= pagesize;
				if (++this_pt >= avail_pt)
				{
					++avail_pt;
					i_results[this_pt] = new unsigned int[pagesize];
				}
			}
			this_end_pt = this_pt;
			this_count_thread = sum_count;
		}
		int this_eot = page_created ? pagesize : this_count_thread;
		unsigned char this_start_pt = page_created ? this_end_pt - 1: this_end_pt;
		int pt_pos = last_count_thread;
		result_idx[i] = (((uint64_t)this_start_pt) << 54ll) | ((uint64_t)(last_count_thread) << 27ll) | (uint64_t)(count);
		for (int j = 0; j < n_streamlines; ++j) {
			if (this_pagetable[j] >= 0) {
				i_results[this_start_pt][pt_pos] = (j<<14)|(this_pagetable[j]);
				this_pagetable[j] = - 1;
				if (++pt_pos >= this_eot)
				{
					this_eot = this_count_thread;
					pt_pos = 0;
					++this_start_pt;
					if (this_start_pt > this_end_pt)
						break;
				}
			}
		}
#else
		results[i].reserve(count*2);
		//i_results[i] = new unsigned int[count];
		sum_count += count;
		for (int j = 0; j < n_streamlines; ++j) {
			if (this_pagetable[j] >= 0)
			{
				results[i][j] = this_pagetable[j];
				this_pagetable[j] = -1;
			}
		}
#endif

		//const int this_seg = pt_to_segs[i];
		//map_t<int, int>::iterator it = results[i].begin();
		//while (it != results[i].end()) {

		//	const int curr_seg = pt_to_segs[it->second];

		//	//const Vector3 vec1 = (segments[this_seg].end_point - segments[this_seg].start_point).normalized();
		//	//const Vector3 vec2 = (segments[curr_seg].end_point - segments[curr_seg].start_point).normalized();

		//	//const float cos_theta = vec1.dot(vec2);
		//	////angle check
		//	//if (cos_theta < .85f)
		//	//	it = results[i].erase(it);
		//	//else
		//	//{
		//	const int nearest = second_level[it->second].nearest(f_streamlines[0][i]);
		//	auto nearest_dist = (f_streamlines[it->first][nearest] - f_streamlines[0][i]).sq();
		//	//distance check
		//	if (nearest_dist > 1.f)//2.f __DIST
		//		it = results[i].erase(it);
		//	else {
		//		it->second = nearest;
		//		it++;
		//	}
		//	//}
		//}
	}
//	return;

	cout << "\ndone contraction\n" << endl;
	printf("found: %u %d\n", sum_count + this_pt * pagesize, atomic_Counter._My_val);
//	i_results[0] = new unsigned int[sum_count];
//	std::fill(i_results[0], i_results[0] + sum_count, 54682u);
	const unsigned vb_size = ((this_pt + 1)*pagesize + 7) / 8;
	unsigned char* validbits = new unsigned char[vb_size];
	memset(validbits, 0xff, sizeof(unsigned char)* vb_size);
	omp_set_num_threads(8);
	auto condition_routine = [](const int& _val, const int & _first) {return _val < _first - 1; };
#pragma omp parallel
#pragma omp  for// schedule(dynamic) 
	for (int i = 0; i < n_points; i++) {
		int ii = i;
		if (n_points>800)
			if (omp_get_thread_num() == 0 && (ii % (int)(n_points / 800) == 0))
				printf("\r%d%c ", (int)(ii / (n_points / 800)), '%');
#if _Page_table
		const uint64_t this_tableInfo = result_idx[i];
		unsigned char start_page = this_tableInfo >> 54;
		unsigned int page_offset = (this_tableInfo >> 27) & 0x7ffffff;
		const unsigned int count = this_tableInfo & 0x7ffffff;
		//unsigned char end_page = start_page;
		unsigned int end_table = page_offset + count;
		const unsigned int this_line = Streamline::getlineof(i);
		const unsigned int this_pos_on_line = i - Streamline::offsets[this_line];
		const unsigned int this_signature = (this_line << 14) | this_pos_on_line;
		auto search_routine = [&](const int& end_search) {
			for (int j = page_offset; j < end_search; ++j)
			{
				const unsigned int this_result = i_results[start_page][j];
				const unsigned int res_line = this_result >> 14;
				const unsigned int res_point = this_result & 0x3fff;
				const uint64_t res_tableInfo = result_idx[getGlobalIndex(res_line, res_point)];

				unsigned char res_start_page = res_tableInfo >> 54ll;
				unsigned int res_page_offset = (res_tableInfo >> 27ll) & 0x7ffffff;
				const unsigned int res_count = res_tableInfo & 0x7ffffff;
				unsigned int res_end_table = res_page_offset + res_count;
				bool found = false, find_rhs = true;
				if (res_end_table > pagesize) {
					if ((int)(i_results[res_start_page][pagesize - 1] - this_signature) > -1)
					{
						found = std::binary_search(i_results[res_start_page] + res_page_offset, i_results[res_start_page] + pagesize - 1, this_signature, condition_routine);
						find_rhs = false;
					}
					else if ((int)(i_results[res_start_page][pagesize - 1] - this_signature) <= 1) {
						find_rhs = false;
						found = true;
					}
					res_start_page++;
					res_page_offset = 0;
					res_end_table -= pagesize;

				}
				if (find_rhs) {

					found = std::binary_search(i_results[res_start_page] + res_page_offset, i_results[res_start_page] + res_end_table, this_signature, condition_routine);
				}
				if (!found)
				{
					const unsigned total_count = pagesize * start_page + j;
					validbits[total_count >> 3] &= (0xff - (1 << (total_count & 0x7)));// mark deleted;
					--atomic_Counter;
				}
			}

		};
		if (end_table > pagesize)
		{
			search_routine(pagesize);
			start_page++;
			page_offset = 0;
			end_table -= pagesize;
		}
		search_routine(end_table);

#else
		map_t<int, unsigned char>::iterator it = results[i].begin();
		while (it != results[i].end()) {

			const int this_line = Streamline::getlineof(i);
			const int pos_on_line = i - Streamline::offsets[this_line];
			int pt2 = getGlobalIndex(it->first, it->second);
			if (results[pt2].size() > 0) {
				auto nn_nn_v = results[pt2].find(this_line);
				if (nn_nn_v != results[pt2].end()) {
					if (fabs(nn_nn_v->second - pos_on_line) > 1) {
						it = results[i].erase(it);
						continue;
					}
					else
						;// results[pt2].erase(this_line);
				}
				else
					it = results[i].erase(it);
			}
			else
			{
				it = results[i].erase(it);
				continue;
			}
			it++;
		}
#endif
	}
//#pragma omp parallel
//#pragma omp  for schedule(dynamic) 
//	for (int i = 0; i < n_points; i++) {
//		//__macro_bound()
//		alpha[i] =  sqrtf((__macro_bound(results[i].size(), 50, 200) -50) / 150.f);//saturate
//	}
	//pair<Vector3, int>* displacement = new pair<Vector3, int>[n_points];
	//for (int i = 0; i < n_points; i++)
	//{
	//	displacement[i].first = 0;
	//	displacement[i].second = 0;
	//}
	//return;
	printf("found: %u %d\n", sum_count + this_pt * pagesize, atomic_Counter._My_val);

	Vector3** f_streamline2 = new Vector3*[n_streamlines];
	f_streamline2[0] = new Vector3[n_points];
	Vector3** displacement = new Vector3*[n_streamlines];
	displacement[0] = new Vector3[n_points];
	Vector3* directions = new Vector3[n_points];
	for (int i = 1; i < n_streamlines; i++) {
		f_streamline2[i] = f_streamline2[i - 1] + Streamline::size(i - 1);
		displacement[i] = displacement[i - 1] + Streamline::size(i - 1);
	}
	for (int j = 0; j < 40; j++)
	{
		_getTangent(directions);
#pragma omp parallel
#pragma omp  for schedule(dynamic) 
		for (int i = 0; i < n_points; i++) {
			float sum_gx = 0;
			Vector3 this_displacement = 0;
#if _Page_table
			const uint64_t this_tableInfo = result_idx[i];
			unsigned char start_page = this_tableInfo >> 54;
			unsigned int page_offset = (this_tableInfo >> 27) & 0x7ffffff;
			const unsigned int count = this_tableInfo & 0x7ffffff;
			unsigned int end_table = page_offset + count;

			if (end_table > pagesize) {
				for (int k = page_offset; k < pagesize; ++k) {
					if (j == 0) {
						const int res_signature = ((start_page * pagesize)<<14) | k;
						if (!((validbits[res_signature >> 3] >> (res_signature & 0x7)) & 0x1))
							i_results[start_page][k] = numeric_limits<uint32_t>::max();
					}
					const unsigned page_entry = i_results[start_page][k];
					if (page_entry != numeric_limits<uint32_t>::max())
					{
						const Vector3 this_vector = f_streamlines[page_entry >> 14][page_entry & 0x3fff] - f_streamlines[0][i];
						const float gx = 1.f/2.f;// (float)(exp(-(this_vector.dot(this_vector)) / (2.f*16.f)) / (sqrt(2 * M_PI*16.f))) / 2.f;
						this_displacement += this_vector * gx;
						sum_gx += gx*2.f;
					}
				}
				start_page++;
				page_offset = 0;
				end_table -= pagesize;
			}
			for (int k = page_offset; k < end_table; ++k) {
				if (j == 0) {
					const int res_signature = ((start_page * pagesize) << 14) | k;
					if (!((validbits[res_signature >> 3] >> (res_signature & 0x7)) & 0x1))
						i_results[start_page][k] = numeric_limits<uint32_t>::max();
				}
				const unsigned page_entry = i_results[start_page][k];
				if (page_entry != numeric_limits<uint32_t>::max())
				{
					const int globalIdx = getGlobalIndex(page_entry >> 14, page_entry & 0x3fff);
					Vector3 this_vector = f_streamlines[page_entry >> 14][page_entry & 0x3fff] - f_streamlines[0][i];
					const float gx = 1.f/2.f;// (float)(exp(-(this_vector.dot(this_vector)) / (2.f*16.f)) / (sqrt(2 * M_PI*16.f))) / 2.f;
					const Vector3 localDir = (directions[i] + directions[globalIdx]).normalized_checked();
					this_vector -= localDir*this_vector.dot(localDir);
					this_displacement += this_vector * gx;
					sum_gx += gx*2.f;
				}
			}
#else
			for (auto edge : results[i]) {
				const Vector3 this_vector = (f_streamlines[edge.first][edge.second] - f_streamlines[0][i]);
				const float gx = 0.5f;//(float)(exp(-(this_vector.dot(this_vector)) / (2.f*16.f)) / (sqrt(2 * M_PI*16.f)));
				//displacement[i].first += this_vector/2.f;
				this_displacement += this_vector*gx;
				sum_gx += gx*2.f;// *2;
				//displacement[i].second++;
			}
#endif

			if (sum_gx != 0)
				this_displacement /= sum_gx;
			else
				this_displacement = 0;
			displacement[0][i] = this_displacement;
		}
		if (j % 1 == 0)
#pragma omp parallel
#pragma omp  for schedule(dynamic)
		for (int i = 0; i < n_points; i++) {
			const int this_line = Streamline::getlineof(i);
			const int pt_on_line = i - Streamline::offsets[this_line];
			int start = __macro_max(0, pt_on_line - 8);
			int end = __macro_min(Streamline::size(this_line), pt_on_line + 8);
			float sum_gx = 1.1f;
			Vector3 this_displacement = displacement[0][i] * 1.1f;
			const Vector3 dir = directions[i];
			for (int k = start; k < end; k++) {
				//const float distik = (f_streamlines[0][i] - f_streamlines[this_line][k]).sq();
				if (k != pt_on_line /*&& !isnan(distik) && !isinf(distik)*/) {
					const float distik_online = fabsf(pt_on_line - k);
					const float gx = (float)(exp(-distik_online* distik_online /64.f));
					sum_gx += gx;
					this_displacement += (displacement[this_line][k] - dir*(displacement[this_line][k].dot(dir))) * gx;
				}
			}
			this_displacement /= sum_gx;
			start = __macro_max(0, pt_on_line - 1);
			end = __macro_min(Streamline::size(this_line) - 1, pt_on_line + 1);
			//const Vector3 dir = directions[i];
			//this_displacement -= dir *(this_displacement.dot(dir));
			f_streamline2[0][i] = f_streamlines[0][i] + this_displacement;
		}
		else
#pragma omp parallel
#pragma omp  for schedule(dynamic)
			for (int i = 0; i < n_points; i++)
			{
				const int this_line = Streamline::getlineof(i);
				const int pt_on_line = i - Streamline::offsets[this_line];

				Vector3 this_displacement = displacement[0][i];
				int start = __macro_max(0, pt_on_line - 1);
				int end = __macro_min(Streamline::size(this_line) - 1, pt_on_line + 1);
				const Vector3 dir = directions[i];//(f_streamlines[0][end] - f_streamlines[0][start]).normalized_checked();
				this_displacement -= dir * (this_displacement.dot(dir));


				f_streamline2[0][i] = f_streamlines[0][i] + this_displacement;
			}
/*
		for (int i = 0; i < n_points; i++) {
			const int this_line = Streamline::getlineof(i);
			const int pt_on_line = i - Streamline::offsets[this_line];
			const Vector3 this_displacement = (displacement[i].first / (float)displacement[i].second);
			if (displacement[i].second != 0)
			{
				f_streamlines[0][i] += this_displacement;

			}
			displacement[i].first = 0;
			displacement[i].second = 0;
		}
*/
		//std::memcpy(f_streamline2[0], f_streamlines[0], sizeof(Vector3) * n_points);
		if(j % 41 == 0)
#pragma omp parallel
#pragma omp  for schedule(dynamic)
		for (int i = 0; i < n_points; i++) {
			const int this_line = Streamline::getlineof(i);
			const int pt_on_line = i - Streamline::offsets[this_line];

			int start = __macro_max(0, pt_on_line - 1);
			int end = __macro_min(Streamline::size(this_line), pt_on_line + 1);
			float sum_gx = 1;
			for (int k = start; k < end; k++) {
				//const float distik = (f_streamlines[0][i] - f_streamlines[this_line][k]).sq();
				if (k != pt_on_line/*&&!isnan(distik)&&!isinf(distik)*/) {
					const float distik_online = fabsf(pt_on_line - k);
					const float gx = (float)(exp(-distik_online * distik_online/16.f) / (sqrt(M_PI)*16));
					sum_gx += gx;
					f_streamline2[0][i] += f_streamlines[this_line][k] * gx;
				}
			}
			f_streamline2[0][i] /= sum_gx;
		}
		std::memcpy(f_streamlines[0], f_streamline2[0], sizeof(Vector3) * n_points);

	}
	for (int j = 0; j < 0; j++)
	{
		std::memcpy(f_streamline2[0], f_streamlines[0], sizeof(Vector3) * n_points);
#pragma omp parallel
#pragma omp  for
		for (int i = 0; i < n_points; i++) {
			const int this_line = Streamline::getlineof(i);
			const int pt_on_line = i - Streamline::offsets[this_line];

			int start = __macro_max(0, pt_on_line - 1);
			int end = __macro_min(Streamline::size(this_line), pt_on_line + 1);
			float sum_gx = 1;
			for (int k = start; k < end; k++) {
				const float distik = f_streamlines[0][i] - f_streamlines[this_line][k];
				if (k != pt_on_line && !isnan(distik) && !isinf(distik)) {
					const float gx = (float)(exp(-16.f / (2.f*16.f)) / (sqrt(2 * M_PI*16.f)));
					sum_gx += gx;
					f_streamline2[0][i] += f_streamlines[this_line][k] * gx;
				}
			}
			f_streamline2[0][i] /= sum_gx;
		}
		std::memcpy(f_streamlines[0], f_streamline2[0], sizeof(Vector3) * n_points);

	}
	//std::thread *cleanup_thread = new std::thread(cleanupthread, results, f_streamline2, pagetable);
	
	//std::async(std::launch::async, [&results]() {delete[] results; });
	//std::async(std::launch::async, [&f_streamline2]() {delete[] f_streamline2[0]; delete[] f_streamline2; });
}
void Transparency() {



	vector<int> _res;
	alpha = new float[n_points];
	float max_alpha = -1;
	float sum_alpha = 0;
	omp_set_num_threads(8);
	bool nonn = true;
#pragma omp parallel
#pragma omp  for
	for (int i = 0; i < n_points; i++) {
		int ii = i;
		if(n_points>800)
		if (omp_get_thread_num() == 0 && (ii % (int)(n_points / 800) == 0))
			printf("\r%d%c ", (int)(ii / (n_points / 800)), '%');
		/*for (auto table : hashtables) {
		unordered_map<int, int> *results = new unordered_map<int, int>();
		table.Query(_res,f_streamlines[0][i],false, results);
		}*/
#if(1)
		vector<pair<float, int>> result = queryANN(hashtables, f_streamlines[0][i], Streamline::getlineof(i));
#ifdef CENTRAL_SURROUNDED
		float _sum_1 = 0, _sum_2 = 0, _gauss_sum_1 = 0, _gauss_sum_2 = 0;
		nonn = false;
#else
		float sum_gx = 0;
		alpha[i] = 0;
#endif
		for (auto res_q : result) {
#ifdef CENTRAL_SURROUNDED 
			if (res_q.first < 2 * R1) {
				float _similarity = computeSimilarity_pt(i, res_q.second);
				float _gx = exp(-square(res_q.first) / (2 * square(2 * R1)));
				if (isinf(_gx))
					printf("error\n");
				_gauss_sum_1 += _gx;
				_sum_1 += _gx * _similarity;
				if (res_q.first < R1) {
					float _gx2 = exp(-square(res_q.first) / (2 * square(R1)));
					_gauss_sum_2 += _gx2;
					_sum_2 += _gx2 * _similarity;
					nonn = true;
				}
			}
#else
			float gx = gaussianDist(res_q.first);
			alpha[i] += gx * square(curvatures[0][i] - curvatures[0][res_q.second]);// computeSimilarity_pt(i, res_q.second);
			sum_gx += gx;
#endif
		}
#ifdef CENTRAL_SURROUNDED
		if (_gauss_sum_1)
			_sum_1 /= _gauss_sum_1;
		else
			_sum_1 = 0;
		if (_gauss_sum_2)
			_sum_2 /= _gauss_sum_2;
		else
			_sum_2 = 0;
		alpha[i] = fabs(_sum_1 - _sum_2);
#else
		if (sum_gx)
			alpha[i] /= sum_gx;
		else
			alpha[i] = 1;

		
#endif
#else
		alpha[i] = curvatures[0][i];
#endif
		if (isnan(alpha[i])||isinf(alpha[i]))
		{
			//printf("%d ", i);
			alpha[i] = 0;
		}
		else {
			sum_alpha += alpha[i];
			max_alpha = __macro_max(alpha[i], max_alpha);
		}
		if (!nonn)
			alpha[i] = -1;
	}

	sum_alpha /= n_points;
	sum_alpha *= 4;

	for (int i = 0; i < n_points; i++){
		if (alpha[i] != -1)
		{
			//alpha[i] /= sum_alpha;
			alpha[i] = __macro_bound(alpha[i], 0, 1);
		}
	}

}
void initialize(const char* filename, int reduced, LSH_Application application, float Radius) {
#if 0
	benchmark(filename);
#else

	// load file
	//LoadWaveFrontObject(filename);
	__global_radius = Radius; //R = Radius/D
	ReadBSL(filename);
	if (reduced > 0 &&reduced < n_streamlines)
	{
		n_streamlines = reduced;
		n_points = Streamline::offsets[reduced];
	}
	tangents = new Vector3[n_points];
	printf("contraction radius: %f\n", Radius);
	FileIO::normalize(Radius, true, Format::STREAMLINE_ARRAY);

	//FileIO::toFStreamlines();
//	FileIO::gaussianSmooth(1);

	// decomposition
	Segment::seg_init(M_PI/2.f, 0.05/Radius);
	//Segment check

	// second level
	initializeSecondLevel();
	// output debug info
	cout << n_points << " points, " << n_streamlines << " streamlines and " << segments.size() << " segments\n";
	// hash function pool
	auto funcs = LshFunc::create(funcpool_size, segments.data(), segments.size(), 4);
	// create the hash tables stocastically
	uniform_int_distribution<int> uni_dist{ 0, funcpool_size - 1 };
	for (int l = 0; l < L; l++) {
		vector<int> func_for_table;
		for (int k = 0; k < K; k++)
			func_for_table.push_back(uni_dist(engine));
		hashtables.emplace_back(func_for_table, TABLESIZE, funcs.first, segments.data(), segments.size());
	}
	auto begin_time = chrono::high_resolution_clock::now();
	switch ((int)application) {
	case LSH_Contraction:
		contraction();
		break;
	case LSH_Alpha:
		Transparency();
		break;
	case LSH_None:
	default:
		break;
	}
	auto end_time = chrono::high_resolution_clock::now();
	std::chrono::duration<double> delta_time = end_time - begin_time;
	cout<<"\nLSH Time: "<<delta_time.count()<<"s\n";
	//Transparency();
	FileIO::normalize(1.f / (Radius*2), false, false, 0,true);//Radius*25 best
	map_t<int, int> map1;
	//std::thread *cleanup_thread = new std::thread(cleanupthread, nullptr, nullptr, nullptr);
#endif

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
#define _LIB
#ifndef _LIB
extern "C" {
	int __declspec(dllexport)
		main() {
		//AllocConsole();

		//freopen("CONIN$", "r", stdin);
		//freopen("CONOUT$", "w", stdout);
		//freopen("CONOUT$", "w", stderr);
		initialize("D:/flow_data/bsldatanormalized/contraction.bsl", 0.02f);

		return 0;
	}
}
#endif
