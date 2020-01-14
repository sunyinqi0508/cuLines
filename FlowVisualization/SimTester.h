//#pragma once
#ifndef SIM_TESTER
#define SIM_TESTER

#include "field2d.h"
#include "streamline.h"
#include "Integrator.h"
#include <cuda.h>

#ifndef max
#define max(a,b) ((a)<(b) ? (b) : (a))
#define min(a,b) ((a)>(b) ? (b) : (a))
#endif

#define HEAP_SIZE 64

struct dsim
{
	double term1;
	double term2;
	int  nj;
};


struct Point
{
	std::vector<dsim> data;
	int N;
	int formal_lineNum;
};

struct heap_Point
{

	int *nj = new int[HEAP_SIZE];
	float *term1 = new float[HEAP_SIZE];
	float *term2 = new float[HEAP_SIZE];
	int N_nj = HEAP_SIZE;
	int N;
	int formal_lineNum;

};



class SimTester
{
public:

	double All_Min_Dist;


	// bool isSimilarWithLines(double h_all[], int *h_start, int  *h_length_l, int nHalfSample, int  n_line, int i, int nofl, double dslf, double max_ds);
	// bool isSimilarWithLinesA (std::vector<Line> &streamLines, std::deque<vec3> &si, int siIdx, int number_line,int a);

	bool isSimilarWithLines_F(std::vector<Line> &streamLines, std::deque<vec3> &si_tmp, int id);
	bool isSimilarWithLines_B(std::vector<Line> &streamLines, std::deque<vec3> &si_tmp, int id);



	bool isSimilarWithSelf(std::deque<vec3> &si, int siIdx);

	bool self_line_similarty(std::vector<vec3> &si_tmp, int id);

	bool sampleLine(const std::deque<vec3>& line, int idx,
		float lineLength, int nHalfSample,
		vector<vec3>& result, int& idxPos);


	bool MysampleLine(const std::deque<vec3>& line, int idx,
		int nHalfSample,
		vector<vec3>& result);




	bool findIdxRange(const std::deque<vec3>&line, const vec3& centerPnt, float radius, int& lowID, int& highID);
private:



};



extern StreamlineParam g_param;

extern float Steplenghth;
extern vector<vector<float>> Simimarity;
extern vector<vector<float>> Term1;
extern vector<vector<float>> Term2;


extern vector<float>Min_dist;
extern vector<float>Min_distJG;


extern vector<int>JGf;
extern vector<int>Jf;

extern vector<int>JGb;
extern vector<int>Jb;



extern vector<int>JG;
extern vector<int>J;


extern vector<int>JGbb;
extern vector<vector<Point>> Dsim;
extern dsim min_dsim(vector<dsim> a);
extern float  ** pints;


//cuda����
extern vector<float>allp;
extern vector<int>n_line;

//__constant__ extern float d_hash[];

#endif
//__constant__  int ids[2000];
