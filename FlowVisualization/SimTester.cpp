#include "SimTester.h"
#include <iostream>



#define NEW_killlines_killpoints

//#define OLD 
//#define  FAST_killpoints
//#define SLOW






//#define OLDdefination
#define NEWdefination





#ifdef OLD


bool SimTester::isSimilarWithLines_F(std::vector<Line> &streamLines, std::deque<vec3> &si, int siIdx)
{

	double lineLength = g_param.w;
	int   nHalfSample = g_param.nHalfSample;
	vector<vec3> siSampled;
	int          siPos;

	if (!sampleLine(si, siIdx, lineLength, nHalfSample, siSampled, siPos))
		return false;

	vec3 p = siSampled[nHalfSample];

	All_Min_Dist = FLT_MAX;
	//线的优先级
	/*

	for(int i=0;i<Min_dist.size();i++)
	{

	Min_distJG[i]=Min_dist[i];


	}
	int k;
	*/
	//线的优先级


	for (int i = 0; i<streamLines.size(); ++i)
	{


		//线的优先级
		/*
		int J=-1;

		double min=99999;

		for(int j=0;j<Min_distJG.size();j++)
		{

		if(Min_distJG[j]<min)
		{
		min=Min_distJG[j];
		J=j;
		}
		}

		Min_distJG[J]=99999;
		k=J;
		*/

		//线的优先级





		//线的优先级i变k
		deque<vec3>& sj = streamLines[i].path;


		// find nearest point
		vec3 q;

		//与 main中方法冲突

		double  min_dist = FLT_MAX;

		int sjIdx = -1;


#ifdef SLOW

		for (int j = 0; j< sj.size(); j++)
		{


			double l = length(p - sj[j]);
			if (l < min_dist)
			{
				q = sj[j];
				min_dist = l;
				sjIdx = j;
			}
		}

#endif


#ifdef FAST_killpoints

		int j = 0;

		while (j<sj.size())
		{

			double l = length(p - sj[j]);

			if (l < min_dist)
			{
				q = sj[j];
				min_dist = l;
				sjIdx = j;
			}

			int next = (l - g_param.dSep) / Steplenghth;

			if (next>0)

				j = j + next;

			else j = j + 1;

		}

#endif



		//线的优先级 i变k

		Min_dist[i] = min_dist;

		//寻找对所有线的All_Min_Dist


		if (min_dist<All_Min_Dist)

		{
			All_Min_Dist = min_dist;
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
			double a = length(siSampled[i] - sjSampled[i]);
			term2 += abs(a - term1);
		}
		term2 = g_param.alpha * term2 / siSampled.size();



		//改定义



		//double yz=0.5;
		// if ((term2) <yz )


		if ((term1 + term2) < g_param.dSep)

		{
			// two lines are too similar
			return true;
		}
	}




	//cout<<"    ";


	return false;

}





bool SimTester::isSimilarWithLines_B(std::vector<Line> &streamLines, std::deque<vec3> &si, int siIdx)
{
	double lineLength = g_param.w;
	int   nHalfSample = g_param.nHalfSample;
	vector<vec3> siSampled;
	int          siPos;

	if (!sampleLine(si, siIdx, lineLength, nHalfSample, siSampled, siPos))
		return false;

	vec3 p = siSampled[nHalfSample];


	All_Min_Dist = FLT_MAX;

	for (int i = 0; i<streamLines.size(); ++i)
	{

		deque<vec3>& sj = streamLines[i].path;

		// find nearest point
		vec3 q;

		//与 main中方法冲突

		double  min_dist = FLT_MAX;

		int sjIdx = -1;

#ifdef SLOW

		for (int j = 0; j< sj.size(); j++)
		{


			double l = length(p - sj[j]);
			if (l < min_dist)
			{
				q = sj[j];
				min_dist = l;
				sjIdx = j;
			}
		}

#endif


#ifdef FAST_killpoints

		int j = 0;

		while (j<sj.size())
		{

			double l = length(p - sj[j]);

			if (l < min_dist)
			{
				q = sj[j];
				min_dist = l;
				sjIdx = j;
			}

			int next = (l - g_param.dSep) / Steplenghth;

			if (next>0)

				j = j + next;


			else j = j + 1;


		}


#endif

		//寻找对所有线的All_Min_Dist


		if (min_dist<All_Min_Dist)

		{
			All_Min_Dist = min_dist;
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
			double a = length(siSampled[i] - sjSampled[i]);
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

	//cout<<"  ";

}



#endif





#ifdef NEW_killlines_killpoints


bool SimTester::isSimilarWithLines(std::vector<Line> &streamLines, std::deque<vec3> &si, int siIdx, int number_line)
{


	double lineLength = g_param.w;

	int   nHalfSample = g_param.nHalfSample;
	vector<vec3> siSampled;
	int          siPos;


	//最终debug
	if (!MysampleLine(si, siIdx, lineLength, nHalfSample, siSampled, siPos))
	{
		//cout << "fuck" << endl;

		Simimarity[number_line].push_back(0);

		Term1[number_line].push_back(0);
		Term2[number_line].push_back(0);
		Dsim[number_line][siIdx].N = 0;

		return false;
	}



	vec3 p = siSampled[nHalfSample];



	All_Min_Dist = FLT_MAX;
	double Term = 9999999;

	int n = 0;
	double Term_1 = 0;
	double Term_2 = 0;



	for (int i = 0; i<streamLines.size(); ++i)
	{

		if (i == number_line)
			continue;

		
		if (JG[i]<0)
		{
			JG[i] = 0;
		}

		if (J[i] == JG[i])
		{

		
			deque<vec3>& sj = streamLines[i].path;

			// find nearest point
			vec3 q;
			double  min_dist = FLT_MAX;
			int sjIdx = -1;

			//在此替换函数 其中  vec3 p为目前的点  deque<vec3>& sj为点集  寻找p到sj的最近距离min_dist  并将sj中最近点设为 q





			int j = 0;
			while (j<sj.size())
			{
				double l = length(p - sj[j]);
				if (l < min_dist)
				{
					q = sj[j];
					min_dist = l;
					sjIdx = j;	
				}
				int next = (l - g_param.dSep) / max_ds;
				if (next>0)
					j = j + next;
				else j = j + 1;
			}






			//新增
			JG[i] = (min_dist - g_param.dSep) / max_ds - 1;
			J[i] = 0;

			//寻找对所有线的All_Min_Dist
			
		


			if (min_dist >= g_param.dSep)
				continue;


			// sample line
			vector<vec3>	sjSampled;
			int				sjPos;


			if (!MysampleLine(sj, sjIdx, lineLength, nHalfSample, sjSampled, sjPos))
			{
				continue;
			}





#ifdef OLDdefination
			//Vector3 term1 = siSampled[nHalfSample] - sjSampled[nHalfSample];//min_dist;
			double term1 = (siSampled[nHalfSample] - sjSampled[nHalfSample]).length();//min_dist
			double term2 = 0.0f;
			for (int i = 0; i < siSampled.size(); ++i)
			{
				//Vector3 a = siSampled[i] - sjSampled[i];
				//Vector3 cha=a-term1;
				//term2 += cha.length();
				double a = length(siSampled[i] - sjSampled[i]);
				term2 += abs(a - term1);
			}

			//term2 = term2 / siSampled.size()/term1.length();
			term2 = term2 / siSampled.size() / term1;
			n++;
			// if(Term2>term)
			//Term_1=Term_1+term1.length();

			Term_1 = Term_1 + term1;
			Term_2 = Term_2 + term2;

#endif




#ifdef NEWdefination

			Vector3 term1 = siSampled[nHalfSample] - sjSampled[nHalfSample];//min_dist;

			double term2 = 0;
			for (int i = 0; i < siSampled.size(); ++i)
			{
				Vector3 a = siSampled[i] - sjSampled[i];
				Vector3 cha = a - term1;
				term2 += cha.length();
			}

			double dd = term1.length();
			term2 = term2 / siSampled.size();

			struct dsim a;
			a.nj = i;
			a.term1 = term1.length();
			a.term2 = term2;

			
			Dsim[number_line][siIdx].data.push_back(a);

#endif
		}

		else J[i] = J[i] + 1;
	  }


	Simimarity[number_line].push_back(0);
	Dsim[number_line][siIdx].N = 1;


	dsim b;
	b = min_dsim(Dsim[number_line][siIdx]);
	Term1[number_line].push_back(b.term1);
	Term2[number_line].push_back(b.term2);

	
	return false;

}




bool SimTester::isSimilarWithLines_B(std::vector<Line> &streamLines, std::deque<vec3> &si, int siIdx)
{
	double lineLength = g_param.w;
	int   nHalfSample = g_param.nHalfSample;
	vector<vec3> siSampled;
	int          siPos;

	if (!sampleLine(si, siIdx, lineLength, nHalfSample, siSampled, siPos))
		return false;

	vec3 p = siSampled[nHalfSample];


	All_Min_Dist = FLT_MAX;


	for (int i = 0; i<streamLines.size(); ++i)
	{


		if (JGb[i]<0)
		{
			JGb[i] = 0;
		}



		if (Jb[i] == JGb[i])
		{

			deque<vec3>& sj = streamLines[i].path;

			// find nearest point
			vec3 q;

			//与 main中方法冲突

			double  min_dist = FLT_MAX;

			int sjIdx = -1;



			//在此替换函数 其中  vec3 p为目前的点  deque<vec3>& sj为点集  寻找p到sj的最近距离min_dist  并将sj中最近点设为 q


			int j = 0;

			while (j<sj.size())
			{

				double l = length(p - sj[j]);

				if (l < min_dist)
				{
					q = sj[j];
					min_dist = l;
					sjIdx = j;
				}

				int next = (l - g_param.dSep) / Steplenghth;

				if (next>0)

					j = j + next;

				else j = j + 1;

			}

			//



			//新增

			JGb[i] = (min_dist - g_param.dSep) / Steplenghth - 1;

			Jb[i] = 0;



			//寻找对所有线的All_Min_Dist


			if (min_dist<All_Min_Dist)
			{
				All_Min_Dist = min_dist;
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

			double term = 0;
			for (int i = 0; i < siSampled.size(); ++i)
			{
				double a = length(siSampled[i] - sjSampled[i]);
				term2 += abs(a - term1);
			}


			term2 = g_param.alpha * term2 / siSampled.size();

			term = term1 + term2;


			if ((term1 + term2) < g_param.dSep)
			{
				// two lines are too similar
				return true;
			}


		}
		else Jb[i]++;


	}

	return false;

}






bool SimTester::isSimilarWithLines_F(std::vector<Line> &streamLines, std::deque<vec3> &si, int siIdx)
{


	//int WLENTH=181;


	double lineLength = g_param.w;

	int   nHalfSample = g_param.nHalfSample;

	vector<vec3> siSampled;
	int          siPos;

	if (!sampleLine(si, siIdx, lineLength, nHalfSample, siSampled, siPos))

		return false;

	vec3 p = siSampled[nHalfSample];


	for (int i = 0; i<streamLines.size(); ++i)
	{





		if (JGf[i]<0)
		{
			JGf[i] = 0;
		}



		if (Jf[i] == JGf[i])
		{


			deque<vec3>& sj = streamLines[i].path;

			// find nearest point

			vec3 q;



			double  min_dist = FLT_MAX;

			int sjIdx = -1;


			//在此替换函数 其中  vec3 p为目前的点  deque<vec3>& sj为点集  寻找p到sj的最近距离min_dist  并将sj中最近点设为 q

			int j = 0;

			while (j<sj.size())
			{

				double l = length(p - sj[j]);

				if (l < min_dist)
				{
					q = sj[j];
					min_dist = l;
					sjIdx = j;
				}

				int next = (l - g_param.dSep) / Steplenghth;

				if (next>0)

					j = j + next;

				else j = j + 1;

			}


			//




			//新增部分

			//if(si.size()<=WLENTH){

			//	JGf[i]=-1;
			//	}else{
			JGf[i] = (min_dist - g_param.dSep) / Steplenghth - 1;
			//	}


			Jf[i] = 0;


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
				double a = length(siSampled[i] - sjSampled[i]);
				term2 += abs(a - term1);
			}
			term2 = g_param.alpha * term2 / siSampled.size();

			if ((term1 + term2) < g_param.dSep)
			{
				// two lines are too similar
				return true;
			}

		}
		else Jf[i]++;
	}

	return false;

}



#endif
















bool SimTester::isSimilarWithSelf(std::deque<vec3> &si, int siIdx)
{
	double lineLength = g_param.w;
	int   nHalfSample = g_param.nHalfSample;
	vector<vec3> siSampled;
	int          siPos;
	if (!sampleLine(si, siIdx, lineLength, nHalfSample, siSampled, siPos))
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
		double l = length(p - sj[j]);
		if (l < min_dist)
		{
			q = sj[j];
			min_dist = l;
			sjIdx = j;
		}
	}




	if (min_dist >= g_param.dSelfsep || sjIdx == -1)
		return false;




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
		double a = length(siSampled[i] - sjSampled[i]);
		term2 += abs(a - term1);
	}


	//调整自环
	double alpha = 5;

	term2 = alpha * term2 / siSampled.size();

	if ((term1 + term2) < g_param.dSelfsep)
	{
		// two lines are too similar
		return true;
	}
	return false;
}





bool SimTester::self_line_similarty(std::vector<vec3> &si_tmp, int id)
{
	vec3 p = si_tmp[id];
	vec3 q = si_tmp[0];
	int compare_id = -1;
	double min_dist = 100000;;
	for (int j = 0; j<si_tmp.size() - g_param.w / 2.0f; j++)
	{
		if (min_dist > length(p - si_tmp[j]) && length(p - si_tmp[j]) > g_param.dMin)
		{
			min_dist = length(p - si_tmp[j]);
			q = si_tmp[j];
			compare_id = j;
		}
	}
	if (compare_id == -1)
	{
		return false;
	}

	if (compare_id < g_param.w / 2 || compare_id > si_tmp.size() - g_param.w / 2)
	{
		return false;
	}

	std::vector<vec3> si;
	std::vector<vec3> sj;
	for (int i = id - g_param.w / 2.0f; i<id + g_param.w / 2.0f; i++)
	{
		si.push_back(si_tmp[i]);
	}
	for (int i = compare_id - g_param.w / 2.0f; i<compare_id + g_param.w / 2.0f; i++)
	{
		sj.push_back(si_tmp[i]);
	}

	double term1 = length(p - q);
	double term2 = 0.0f;
	for (int k = 0; k<si.size(); k++)
	{
		double a = length(si[k] - sj[k]);
		term2 += abs(a - term1);
	}
	term2 = g_param.alpha * term2 / si.size();
	if ((term1 + term2) > g_param.dSelfsep)
	{
		return true;
	}

	return false;
}



bool SimTester::sampleLine(const std::deque<vec3>& line, int idx,
	double lineLength, int nHalfSample,
	vector<vec3>& result, int& idxPos)
{
	if (idx<0 || idx >= line.size())
		return false;

	double segmentLength = lineLength / (nHalfSample * 2);

	vector<vec3> buffer[2];
	double totLength[2] = { 0, 0 };
	int   idxDir[2] = { 1, -1 };
	int   idxBound[2] = { line.size() - 1, 0 };

	// grow pnts
	for (int ithDir = 0; ithDir<2; ++ithDir)
	{
		buffer[ithDir].reserve(nHalfSample * 2 + 1);
		if (idx != idxBound[ithDir])
		{
			// forword resample
			int thisIdx = idx, nextIdx = idx + idxDir[ithDir];
			vec3 curPnt = line[thisIdx];
			vec3 curDir = line[nextIdx] - curPnt;
			double allocateLength = curDir.length();
			curDir /= allocateLength;

			while (buffer[ithDir].size() < nHalfSample * 2 + 1)
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

	int validData[2] = { nHalfSample, nHalfSample };
	for (int i = 0; i < 2; ++i)
	{
		int nSample = buffer[i].size() - 1;
		if (nSample < nHalfSample)
		{
			validData[i] = nSample;
			validData[1 - i] += nHalfSample - nSample;
		}
	}

	result.clear();
	result.reserve(nHalfSample * 2 + 1);
	for (int i = validData[1]; i > 0; i--)
		result.push_back(buffer[1][i]);
	idxPos = result.size();
	for (int i = 0; i <= validData[0]; i++)
		result.push_back(buffer[0][i]);
	return true;
}





bool SimTester::MysampleLine(const std::deque<vec3>& line, int idx, double lineLength, int nHalfSample, vector<vec3>& result, int& idxPos)
{

	if (idx < nHalfSample || line.size() - idx - 1 < nHalfSample)
	{
		//cout << "NO" << endl;
		return false;
	}

	result.resize(nHalfSample * 2 + 1);
	for (int i = 0; i < 2 * nHalfSample + 1; ++i)
	{
		result[i] = line[i + idx - nHalfSample];
		//cout << result[i].z << endl;
	}

	//cout << endl;
	return true;


}












bool SimTester::findIdxRange(const std::deque<vec3>&line, const vec3& centerPnt, double radius, int& lowID, int& highID)
{
	lowID = 0;
	highID = line.size();

	int i;
	int centerID[2] = { 0, line.size() - 1 };
	double initDist[2] = { 0, 0 };
	double minDist = FLT_MAX;
	for (i = 0; i < line.size() - 1; ++i)
	{
		vec3 d1 = line[i + 1] - line[i];
		vec3 d2 = (centerPnt - line[i]);
		double t = d2.dot(d1) / d1.dot(d1);
		t = min(1.0, max(0.0, t));
		vec3 td1 = t * d1;
		double dist = (d2 - td1).length();
		if (dist < minDist)
		{
			minDist = dist;
			centerID[0] = i;
			centerID[1] = i + 1;
			initDist[0] = td1.length();
			initDist[1] = d1.length() - initDist[0];
		}
	}

	for (i = centerID[0] - 1; i > 0; --i)
	{
		initDist[0] += (line[i] - line[i + 1]).length();
		if (initDist[0] >= radius)
		{
			lowID = i;
			break;
		}
	}

	for (i = centerID[1] + 1; i < line.size(); ++i)
	{
		initDist[1] += (line[i] - line[i - 1]).length();
		if (initDist[1] >= radius)
		{
			highID = i;
			break;
		}
	}
	return true;
}

