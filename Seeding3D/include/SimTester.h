#pragma once
#include "field2d.h"
#include "../cuLines/LSH.h"
#include "streamline.h"
#include <deque>
#include <vector>
class SimTester
{
public:
	static bool isSimilarWithLines(std::vector<Line> &streamLines, std::deque<vec3> &si_tmp, int id);
	static bool isSimilarWithLines(std::vector<std::vector<Vector3>>& streamlines, int * pagetable, const int * seg2line, HashTable & ht, Vector3 * this_streamline, int ptr_front, int ptr_rear, int id);
	static bool isSimilarWithSelf( std::deque<vec3> &si, int siIdx );

	static bool self_line_similarty(std::vector<vec3> &si_tmp, int id);
	template<class Container>
	static bool sampleLine(	const Container& line, int idx,
							double lineLength, int nHalfSample, 
							std::vector<vec3>& result, int& idxPos);
	template<class Container>
	static bool findIdxRange(const Container&line, const vec3& centerPnt , double radius, int& lowID, int& highID);
private:
};
