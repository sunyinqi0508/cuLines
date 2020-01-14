#include "SimTester.h"
#include <algorithm>
using namespace std;

bool SimTester::isSimilarWithLines( std::vector<Line> &streamLines, std::deque<vec3> &si, int siIdx )
{
	double lineLength = g_param.w;
	int   nHalfSample = g_param.nHalfSample;
	vector<vec3> siSampled;
	int          siPos;
	if(!sampleLine(si, siIdx, lineLength, nHalfSample, siSampled, siPos))
		return false;

	vec3 p = siSampled[nHalfSample];
	for (int i = 0; i<streamLines.size(); ++i)
	{	
		vector<vec3>& sj = streamLines[i].path;

		// find nearest point
		vec3 q;
		double min_dist = FLT_MAX;
		int sjIdx = -1;
		for (int j = 0; j< sj.size(); j++)
		{
			double l = (p - sj[j]).length();
			if (l < min_dist)
			{
				q = sj[j];
				min_dist = l;
				sjIdx = j;
			}
		}
		if (min_dist >= g_param.dSep)
			continue;

		// sample line
		vector<vec3>	sjSampled;
		int				sjPos;
		if (!sampleLine(sj, sjIdx, lineLength, nHalfSample, sjSampled, sjPos))
			continue;

		// enough points to compare
		double term1 = (siSampled[nHalfSample] - sjSampled[nHalfSample]).length();//min_dist;
		double term2 = 0.0f;
		for (int i = 0; i < siSampled.size(); ++i)
		{
			double a = (siSampled[i] - sjSampled[i]).length();
			term2 += abs(a - term1);
		}
		term2 = g_param.alpha * term2 / siSampled.size();

		if ((term1 + term2) < g_param.dSep)
		{
			// two lines are too similar
			return true;
		}
	}
	return false;
}
bool SimTester::isSimilarWithLines(vector<vector<Vector3>>& streamlines, int* pagetable, const int* seg2line, HashTable & ht,  Vector3 * this_streamline, int ptr_front, int ptr_rear, int id)
{
	ptr_front++, ptr_rear--;//.---.
	vector<int> results;
	int len_currentline = ptr_rear - ptr_front + 1;
	if (len_currentline > g_param.nHalfSample) {
		int mid_pt;
		int half_sample = g_param.nHalfSample;
		if (len_currentline <= g_param.nHalfSample * 2) {
			mid_pt = len_currentline / 2.f + ptr_front;
			half_sample = ptr_rear - mid_pt;
		}
		else {
			if (id == ptr_rear)
				mid_pt = ptr_rear - half_sample;
			else if (id == ptr_front)
				mid_pt = ptr_front + half_sample;
			else
				throw "Unspicified usage.";//return false;
		}
		 
		ht.Query(results, pagetable, this_streamline[mid_pt]/10.f, seg2line, streamlines.size());
		for (const int& nearest_segment : results)
		{
			const int this_line = segments[nearest_segment].line;
			const int nearest_point = second_level[nearest_segment].nearest(this_streamline[mid_pt]);
			int _begin = nearest_point - half_sample, _end;
			if (_begin < 0)
			{
				_begin = 0;
				half_sample = nearest_point;
			}
			_end = nearest_point + half_sample;
			if (_end > streamlines[this_line].size())
			{
				_end = streamlines[this_line].size();
				half_sample = _end - nearest_point - 1;
			}
			const int mid_pt_target = (_end - _begin) / 2.f;
			const int begin_source = mid_pt - half_sample, end_source = mid_pt + half_sample + 1;

			float centralDist = this_streamline[mid_pt].distance(streamlines[this_line][mid_pt_target]);
			float variation = 0;
			for (int i = 0; i < 2 * half_sample; ++i)
				variation += std::abs(this_streamline[begin_source + i].distance(streamlines[this_line][_begin + i]) - centralDist);
			variation /= (2 * half_sample);
			if (centralDist + g_param.alpha*variation < g_param.dSep)
				return true;
		}
		return false;
	}
	return false;
}
bool SimTester::isSimilarWithSelf(std::deque<vec3> &si, int siIdx )
{
	double lineLength = g_param.w;
	int   nHalfSample = g_param.nHalfSample;
	vector<vec3> siSampled;
	int          siPos;
	if(!sampleLine(si, siIdx, lineLength, nHalfSample, siSampled, siPos))
		return false;

	vec3 p = siSampled[nHalfSample];
	int lowID, highID;
	findIdxRange(si, p, g_param.dMin, lowID, highID);

	// find nearest point
	deque<vec3>& sj = si;
	vec3 q;
	int sjIdx = -1;
	double min_dist = FLT_MAX;
	for (int j = 0; j< sj.size(); j++)
	{
		if (j >= lowID && j <= highID)
			continue;
		double l = (p - sj[j]).length();
		if (l < min_dist)
		{
			q = sj[j];
			min_dist = l;
			sjIdx = j;
		}
	}

	if (min_dist >= g_param.dSelfsep || sjIdx == -1)
		return false;
	printf("hit");
	getchar();
	// sample line
	vector<vec3>	sjSampled;
	int				sjPos;
	if (!sampleLine(sj, sjIdx, lineLength, nHalfSample, sjSampled, sjPos))
		return false;

	// enough points to compare
	double term1 = (siSampled[nHalfSample] - sjSampled[nHalfSample]).length();//min_dist;
	double term2 = 0.0f;
	for (int i = 0; i < siSampled.size(); ++i)
	{
		double a = (siSampled[i] - sjSampled[i]).length();
		term2 += abs(a - term1);
	}
	term2 = g_param.alpha * term2 / siSampled.size();

	if ((term1 + term2) < g_param.dSelfsep)
	{
		// two lines are too similar

		return true;
	}
	return false;
}

bool SimTester::self_line_similarty( std::vector<vec3> &si_tmp, int id )
{
	vec3 p = si_tmp[id];
	vec3 q = si_tmp[0];
	int compare_id = -1;
	double min_dist = 100000;;
	for(int j=0; j<si_tmp.size() - g_param.w/2.0f; j++)
	{
		if(min_dist > (p - si_tmp[j]).length() && (p - si_tmp[j]).length() > g_param.dMin)
		{		 
			min_dist = (p - si_tmp[j]).length();
			q = si_tmp[j];
			compare_id = j;
		}
	}
	if(compare_id == -1)
	{
		return false;
	}

	if(compare_id < g_param.w/2 || compare_id > si_tmp.size() - g_param.w/2)
	{
		return false;
	}

	std::vector<vec3> si;
	std::vector<vec3> sj;
	for(int i=id-g_param.w/2.0f; i<id+g_param.w/2.0f; i++)
	{
		si.push_back(si_tmp[i]);
	}
	for(int i=compare_id-g_param.w/2.0f; i<compare_id+g_param.w/2.0f; i++)
	{
		sj.push_back(si_tmp[i]);
	}

	double term1 = (p-q).length();
	double term2 = 0.0f;
	for(int k=0; k<si.size(); k++)
	{
		double a = (si[k] - sj[k]).length();
		term2 += abs(a-term1);
	}
	term2 = g_param.alpha * term2 / si.size();
	if ((term1 + term2) > g_param.dSelfsep)
	{
		return true;
	} 

	return false;
}
template<class Container>
bool SimTester::sampleLine(const Container& line, int idx,
						   double lineLength, int nHalfSample, 
						   vector<vec3>& result, int& idxPos )
{
	if (idx<0 || idx >= line.size())
		return false;

	double segmentLength = lineLength / (nHalfSample*2);
	vector<vec3> buffer[2];
	double totLength[2] = {0, 0};
	int   idxDir[2]    = {1, -1};
	int   idxBound[2]  = {line.size()-1, 0};

	// grow pnts
	for (int ithDir = 0; ithDir<2; ++ithDir)
	{
		buffer[ithDir].reserve(nHalfSample*2+1);
		if (idx != idxBound[ithDir])
		{
			// forword resample
			int thisIdx = idx, nextIdx = idx + idxDir[ithDir];
			vec3 curPnt = line[thisIdx];
			vec3 curDir = line[nextIdx] - curPnt;
			double allocateLength = curDir.length();
			curDir /= allocateLength;

			while (buffer[ithDir].size() < nHalfSample*2+1)
			{
				if (totLength[ithDir] > allocateLength)
				{
					nextIdx += idxDir[ithDir];
					thisIdx += idxDir[ithDir];
					if (nextIdx >= line.size() || nextIdx < 0)
						break;

					vec3  delta = line[nextIdx] - line[thisIdx];
					double deltaLength = delta.length();
					double remainLength = totLength[ithDir] - allocateLength;
					allocateLength += deltaLength;
					curDir = delta / deltaLength;
					curPnt = line[thisIdx] + curDir * remainLength;
				}
				else
				{
					buffer[ithDir].push_back(curPnt);
					curPnt += curDir * segmentLength;
					totLength[ithDir] += segmentLength;
				}
			}
			totLength[ithDir] -= segmentLength;
		}
		else
			buffer[ithDir].push_back(line[idx]);
	}

	// line is too short
	if (buffer[0].size() + buffer[1].size() < nHalfSample * 2 + 2)
		return false;

	int validData[2] = {nHalfSample, nHalfSample};
	for (int i = 0; i < 2; ++i)
	{
		int nSample = buffer[i].size()-1;
		if (nSample < nHalfSample)
		{
			validData[i] = nSample;
			validData[1-i] += nHalfSample - nSample;
		}
	}

	result.clear();
	result.reserve(nHalfSample*2+1);
	for (int i = validData[1]; i > 0; i--)
		result.push_back(buffer[1][i]);
	idxPos = result.size();
	for (int i = 0; i <= validData[0]; i++)
		result.push_back(buffer[0][i]);
	return true;
}
template<class Container>
bool SimTester::findIdxRange( const Container&line, const vec3& centerPnt, double radius, int& lowID, int& highID )
{
	lowID = 0;
	highID = line.size();

	int i;
	int centerID[2] = {0, line.size()-1};
	double initDist[2] = {0,0};
	double minDist = FLT_MAX;
	for (i = 0; i < line.size()-1; ++i)
	{
		vec3 d1 = line[i+1] - line[i];
		vec3 d2 = (centerPnt - line[i]);
		double t = d2.dot(d1) / d1.dot(d1);
		t = std::min(1.0, std::max(0.0, t));
		vec3 td1 = t * d1;
		double dist = (d2 - td1).length();
		if (dist < minDist)
		{
			minDist = dist;
			centerID[0] = i;
			centerID[1] = i+1;
			initDist[0] = td1.length();
			initDist[1] = d1.length() - initDist[0];
		}
	}

	for (i = centerID[0]-1; i > 0; --i)
	{
		initDist[0] += (line[i] - line[i+1]).length();
		if (initDist[0] >= radius)
		{
			lowID = i;
			break;
		}
	}

	for (i = centerID[1]+1; i < line.size(); ++i)
	{
		initDist[1] += (line[i] - line[i-1]).length();
		if (initDist[1] >= radius)
		{
			highID = i;
			break;
		}
	}
	return true;
}

