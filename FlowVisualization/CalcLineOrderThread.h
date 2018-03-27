#ifndef CALCLINEORDERTHREAD_H
#define CALCLINEORDERTHREAD_H

#include <qthread.h>
#include <vector>
#include <qdebug.h>

#include "cuda_runtime.h"
#include "cuda.h"
#include <windows.h>
#include <stdint.h>
#include <assert.h>

#include "SimTester.h"
#include "Parameters.h"
#include "CudaVarIableStruct.h"
#include <bitset>

using namespace std;

class CalcLineOrderThread : public QThread
{
	Q_OBJECT
public:
	CalcLineOrderThread(vector<Line> &streamlines);
	~CalcLineOrderThread();

	void calcGlobalLineOrder();
	void calcLocalLineOrder();

//	CudaVariableStruct *cuda_struct;
	int *idxMappings;
	vector<Line> streamlines;
	int *p_left[BUFFERS];
	vector<int> *line_division;
	int *linearKD_id;
	cudaStream_t nonBlockingCpy;
	vector<int> global_order;
	Mutex buffer_flags[BUFFERS];
	int *linearKd;
	int N_p = 0;
	int *p2line, *seg2line;
	int *d_buckets_g;
	int lkd_idx = 0;
	int *d_id;
	float *d_varidata;
	int *h_start;
	int *d_lkd;
	int *d_lb;
	float orig_minx = 0, orig_miny = 0;
	int gap;
	int block_point;
	bool *ptAvailable;
	//bool *d_avail_pt;
	int *h_nj;
	static float *g_saliency;
	vector<vector<int>> segs2pts;
	vector<vector<float>> Term1;
	vector<vector<float>> Term2;
	vector<vector<Point>> Dsim;
	vector<float>allp;
	vector<int>n_line;
	unsigned char *ptavail;
	uchar4* d_bucketsforpt;

	void makeData();
	int BinaryLoacate(int & k);
	Vector3& ptMappings(int);

	vector<l_vector> splitlines(int *p2seg, float max_k);
	Vector3 getcenter(int k, int start, int end);
	float length_curve(int k, int start, int end);
	float length_vector(int k, int start, int end);

	bool *bool_arr;

signals:
	void sendLineOrder(vector<int> order, int sign);//sign is 1:global order won't changed,sign is 2:each calc need to be calc again
	void sendParameters(vector<int> *line_division, int *linearKd, int *linearKD_id);
	void sendDevicePointers(void *devicepointer);

private slots:
void calcCurrLineOrder(vector<int> vec1, vector<int> vec2);

private:
	SimTester SimTester;
	void makeMappings();
	float max_x, max_y, max_z, min_x, min_y, min_z;

protected:
	void run();
};


static LARGE_INTEGER start_precise, end_precise, freq;
#endif //CALCLINEORDERTHREAD_H