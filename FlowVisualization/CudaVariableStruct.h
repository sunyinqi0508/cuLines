#ifndef CUDAVARIABLESTRUCT
#define CUDAVARIABLESTRUCT
#define QT_NO_DEBUG_OUTPUT

#include <set>
#include <mutex>
#include <queue>
#include <vector_types.h>
//#include <KD.h>

#define SAVED_TREES 128
#define PI 3.1415926f 
#define INF  999999999
#define HEAP_SIZE 64
#define BUFFERS 3
#define RIGHT_MASK 0
#define LEFT_MASK 1
#define BOTH_MASK 2
#define NEITHER_MASK 3
#define GET_SEG 0x3ffff
#define HALF_SAMPLE	5
#define THREADS_PER_BLOCK 128
#define MAX_CHAR 128
#define BUCKET_NUM 100 	
#define E 2.71828

using namespace std;

namespace KD_Tree {


	struct KD_Vector3  {
		float x, y, z;
		int id;
		KD_Vector3(float x, float y, float z, int id) :x(x), y(y), z(z), id(id) {};
		KD_Vector3() { x = y = z = 0; }
		KD_Vector3(Vector3& vec) { x = vec.x; y = vec.y; z = vec.z; id = vec.id; }

		operator Vector3() {
			return Vector3(x, y, z, id);
		}

	};

	template<int dim>
	class cmp {
	public:
		inline bool operator()(const KD_Vector3  &a, const KD_Vector3 & b) {
			switch (dim) {
			case 0:return a.x < b.x;
			case 1:return a.y < b.y;
			default:return a.z < b.z;
			}
		}
	};

	template<int dim>
	using type = priority_queue<KD_Vector3, vector<KD_Vector3 >, cmp<dim>>;

	template<int dim>
	void build(type<dim> input, int back, int* linearKd, int &lkd_idx)
	{
		float *f_linearKd = reinterpret_cast<float *>(linearKd);
		linearKd[back] = lkd_idx;
		const int nextsize = input.size() >> 1;
		const int nextd = (dim + 1) % 3;

		int i = 0;
		type<nextd> l;
		type<nextd> r;
		while (i++ < nextsize - 1)
		{
			r.push(input.top());
			input.pop();
		}
		KD_Vector3 node = input.top();
		int *id = &linearKd[lkd_idx];
		linearKd[lkd_idx++] = node.id;// << 2;
		f_linearKd[lkd_idx++] = node.x;
		f_linearKd[lkd_idx++] = node.y;
		f_linearKd[lkd_idx++] = node.z;
		input.pop();
		while (!input.empty())
		{
			l.push(input.top());
			input.pop();
		}

		if (l.size() <= 0)
		{
			linearKd[lkd_idx++] = -1;
			
			//write back - min max
			if (back > 0)
			{
				f_linearKd[back + 1] = f_linearKd[back + 4] = node.x;
				f_linearKd[back + 2] = f_linearKd[back + 5] = node.y;
				f_linearKd[back + 3] = f_linearKd[back + 6] = node.z;
			}
		}
		else
		{
			i = lkd_idx;
			//*id |= 1;
			if (r.size() <= 0)
			{
				lkd_idx += 7;
				linearKd[lkd_idx++] = -1;
				build<nextd>(l, i, linearKd, lkd_idx);
				if (back > 0)
				{
					f_linearKd[back + 1] = min(node.x, f_linearKd[i + 1]);
					f_linearKd[back + 4] = max(node.x, f_linearKd[i + 4]);
					f_linearKd[back + 2] = min(node.y, f_linearKd[i + 2]);
					f_linearKd[back + 5] = max(node.y, f_linearKd[i + 5]);
					f_linearKd[back + 3] = min(node.z, f_linearKd[i + 3]);
					f_linearKd[back + 6] = max(node.z, f_linearKd[i + 6]);
				}
			}
			else
			{
				lkd_idx += 14;
				//*id |= 2;
				build<nextd>(l, i, linearKd, lkd_idx);
				if (back > 0) {
					f_linearKd[back + 1] = min(node.x, f_linearKd[i + 1]);
					f_linearKd[back + 4] = max(node.x, f_linearKd[i + 4]);
					f_linearKd[back + 2] = min(node.y, f_linearKd[i + 2]);
					f_linearKd[back + 5] = max(node.y, f_linearKd[i + 5]);
					f_linearKd[back + 3] = min(node.z, f_linearKd[i + 3]);
					f_linearKd[back + 6] = max(node.z, f_linearKd[i + 6]);
				}
				build<nextd>(r, i + 7, linearKd, lkd_idx);
				if (back > 0)
				{
					f_linearKd[back + 1] = min(f_linearKd[back + 1], f_linearKd[i + 8]);
					f_linearKd[back + 4] = max(f_linearKd[back + 4], f_linearKd[i + 11]);
					f_linearKd[back + 2] = min(f_linearKd[back + 2], f_linearKd[i + 9]);
					f_linearKd[back + 5] = max(f_linearKd[back + 5], f_linearKd[i + 12]);
					f_linearKd[back + 3] = min(f_linearKd[back + 3], f_linearKd[i + 10]);
					f_linearKd[back + 6] = max(f_linearKd[back + 6], f_linearKd[i + 13]);
				}
			}

		}

	}

	inline bool search(KD_Vector3 pt, float r0, int *currentKD) {
		float * f_currentKD = reinterpret_cast<float *> (currentKD);
		int next = 0;
		short dim = 0, stackidx = 0, currdata = 0;
		int stack[64];
		float dss = INT_MAX;
		int idx = -1;	float ds;
		//vector<int> res;
		while (next != -1) {//using mask to reduce memory usage 
			switch (dim)
			{
			case 0:currdata = pt.x;
			case 1:currdata = pt.y;
			default:currdata = pt.z;
			}//use pos in kernel

			ds = pow(pt.x - f_currentKD[next + 1], 2) + pow(pt.y - f_currentKD[next + 2], 2) + pow(pt.z - f_currentKD[next + 3], 2);

			if (ds < r0)
				return true;

			if (f_currentKD[next + dim + 1] < currdata)
			{
				if (currentKD[next + 4] != -1 && currentKD[next + 11] != -1)
					stack[stackidx++] = (currentKD[next + 11] << 2) + (currentKD[currentKD[next + 11] + 4] < 0 ?
				NEITHER_MASK : currentKD[currentKD[next + 11] + 11] < 0 ? LEFT_MASK : BOTH_MASK);
				next = currentKD[next + 4];
			}
			else
			{
				if (currentKD[next + 4] != -1)
				{
					stack[stackidx++] = (currentKD[next + 4] << 2) + (currentKD[currentKD[next + 4] + 4] < 0 ?
					NEITHER_MASK : currentKD[currentKD[next + 4] + 11] < 0 ? LEFT_MASK : BOTH_MASK);
					next = currentKD[next + 11];
				}
				else
					break;
			}

			dim = (++dim - 3) ? dim : dim - 3;
		}

		int r, rt;

	ctn:	while (stackidx > 0) {

		rt = stack[--stackidx];
		r = rt&NEITHER_MASK;
		rt >>= 2;
		ds = pow(pt.x - f_currentKD[rt + 1], 2) + pow(pt.y - f_currentKD[rt + 2], 2) + pow(pt.z - f_currentKD[rt + 3], 2);
		if (ds < r0)
			return true;
		ds = 0;
		switch (r)
		{
		case 0: rt = rt + 11;
			break;
		case 1: rt = rt + 4;
			break;
		case 3:continue;
		default:
			rt = rt + 4;
			break;
		}
		if (pt.x < f_currentKD[rt + 1])
			ds += pow(pt.x - f_currentKD[rt + 1], 2);
		else if (pt.x > f_currentKD[rt + 4])
			ds += pow(pt.x - f_currentKD[rt + 4], 2);
		if (pt.y < f_currentKD[rt + 2])
			ds += pow(pt.y - f_currentKD[rt + 2], 2);
		else if (pt.y > f_currentKD[rt + 5])
			ds += pow(pt.y - f_currentKD[rt + 5], 2);
		if (pt.z < f_currentKD[rt + 3])
			ds += pow(pt.z - f_currentKD[rt + 3], 2);
		else if (pt.z > f_currentKD[rt + 6])
			ds += pow(pt.z - f_currentKD[rt + 6], 2);
		if (ds < r0)
		{
			stack[stackidx] = currentKD[rt] << 2;
			stack[stackidx++] += currentKD[currentKD[rt] + 4] < 0 ? NEITHER_MASK : currentKD[currentKD[rt] + 11] < 0 ? LEFT_MASK : BOTH_MASK;//BOTH_MASK;
		}
		if (r == 2)
		{
			rt = rt + 7;
			ds = 0;
			if (pt.x < f_currentKD[rt + 1])
				ds += pow(pt.x - f_currentKD[rt + 1], 2);
			else if (pt.x > f_currentKD[rt + 4])
				ds += pow(pt.x - f_currentKD[rt + 4], 2);
			if (pt.y < f_currentKD[rt + 2])
				ds += pow(pt.y - f_currentKD[rt + 2], 2);
			else if (pt.y > f_currentKD[rt + 5])
				ds += pow(pt.y - f_currentKD[rt + 5], 2);
			if (pt.z < f_currentKD[rt + 3])
				ds += pow(pt.z - f_currentKD[rt + 3], 2);
			else if (pt.z > f_currentKD[rt + 6])
				ds += pow(pt.z - f_currentKD[rt + 6], 2);
			if (ds < r0)
			{
				stack[stackidx] = currentKD[rt] << 2;
				stack[stackidx++] += currentKD[currentKD[rt] + 4] < 0 ? NEITHER_MASK : currentKD[currentKD[rt] + 11] < 0 ? LEFT_MASK : BOTH_MASK;
			}
		}
	}
			return false;
	}
	inline pair<int, float> search(KD_Vector3 pt, int *currentKD) {
		float * f_currentKD = reinterpret_cast<float *> (currentKD);
		int next = 0;
		short dim = 0, stackidx = 0, currdata = 0;
		int stack[64];
		float dss = INT_MAX;
		int idx = -1;	float ds;
		//vector<int> res;
		while (next != -1) {//using mask to reduce memory usage 
			switch (dim)
			{
			case 0:currdata = pt.x;
			case 1:currdata = pt.y;
			default:currdata = pt.z;
			}//use pos in kernel

			ds = pow(pt.x - f_currentKD[next + 1], 2) + pow(pt.y - f_currentKD[next + 2], 2) + pow(pt.z - f_currentKD[next + 3], 2);

			if (ds < dss)
			{
				dss = ds;
				idx = currentKD[next];
			}
			if (f_currentKD[next + dim + 1] < currdata)
			{
				if (currentKD[next + 4] != -1 && currentKD[next + 11] != -1)
					stack[stackidx++] = (currentKD[next + 11] << 2) + (currentKD[currentKD[next + 11] + 4] < 0 ?
						NEITHER_MASK : currentKD[currentKD[next + 11] + 11] < 0 ? LEFT_MASK : BOTH_MASK);
				next = currentKD[next + 4];
			}
			else
			{
				if (currentKD[next + 4] != -1)
				{
					stack[stackidx++] = (currentKD[next + 4] << 2) + (currentKD[currentKD[next + 4] + 4] < 0 ?
						NEITHER_MASK : currentKD[currentKD[next + 4] + 11] < 0 ? LEFT_MASK : BOTH_MASK);
					next = currentKD[next + 11];
				}
				else
					break;
			}

			dim = (++dim - 3) ? dim : dim - 3;
		}

		int r, rt;

	ctn:	while (stackidx > 0) {

		rt = stack[--stackidx];
		r = rt&NEITHER_MASK;
		rt >>= 2;
		ds = pow(pt.x - f_currentKD[rt + 1], 2) + pow(pt.y - f_currentKD[rt + 2], 2) + pow(pt.z - f_currentKD[rt + 3], 2);
		if (ds < dss)
		{
			dss = ds;
			idx = currentKD[rt];
		}
		ds = 0;
		switch (r)
		{
		case 0: rt = rt + 11;
			break;
		case 1: rt = rt + 4;
			break;
		case 3:continue;
		default:
			rt = rt + 4;
			break;
		}
		if (pt.x < f_currentKD[rt + 1])
			ds += pow(pt.x - f_currentKD[rt + 1], 2);
		else if (pt.x > f_currentKD[rt + 4])
			ds += pow(pt.x - f_currentKD[rt + 4], 2);
		if (pt.y < f_currentKD[rt + 2])
			ds += pow(pt.y - f_currentKD[rt + 2], 2);
		else if (pt.y > f_currentKD[rt + 5])
			ds += pow(pt.y - f_currentKD[rt + 5], 2);
		if (pt.z < f_currentKD[rt + 3])
			ds += pow(pt.z - f_currentKD[rt + 3], 2);
		else if (pt.z > f_currentKD[rt + 6])
			ds += pow(pt.z - f_currentKD[rt + 6], 2);
		if (ds < dss)
		{
			stack[stackidx] = currentKD[rt] << 2;
			stack[stackidx++] += currentKD[currentKD[rt] + 4] < 0 ? NEITHER_MASK : currentKD[currentKD[rt] + 11] < 0 ? LEFT_MASK : BOTH_MASK;//BOTH_MASK;
		}
		if (r == 2)
		{
			rt = rt + 7;
			ds = 0;
			if (pt.x < f_currentKD[rt + 1])
				ds += pow(pt.x - f_currentKD[rt + 1], 2);
			else if (pt.x > f_currentKD[rt + 4])
				ds += pow(pt.x - f_currentKD[rt + 4], 2);
			if (pt.y < f_currentKD[rt + 2])
				ds += pow(pt.y - f_currentKD[rt + 2], 2);
			else if (pt.y > f_currentKD[rt + 5])
				ds += pow(pt.y - f_currentKD[rt + 5], 2);
			if (pt.z < f_currentKD[rt + 3])
				ds += pow(pt.z - f_currentKD[rt + 3], 2);
			else if (pt.z > f_currentKD[rt + 6])
				ds += pow(pt.z - f_currentKD[rt + 6], 2);
			if (ds < dss)
			{
				stack[stackidx] = currentKD[rt] << 2;
				stack[stackidx++] += currentKD[currentKD[rt] + 4] < 0 ? NEITHER_MASK : currentKD[currentKD[rt] + 11] < 0 ? LEFT_MASK : BOTH_MASK;
			}
		}
	}
			return make_pair(idx, dss);
	}
}
struct Bits
{
	unsigned b0 : 1, b1 : 1, b2 : 1, b3 : 1, b4 : 1, b5 : 1, b6 : 1, b7 : 1;
};
union CBits
{
	Bits bits;
	unsigned char byte;
	CBits() {}
	CBits(bool* const & begin) {
		bits.b0 = begin[0];
		bits.b1 = begin[1];
		bits.b2 = begin[2];
		bits.b3 = begin[3];
		bits.b4 = begin[4];
		bits.b5 = begin[5];
		bits.b6 = begin[0];
		bits.b7 = begin[7];
	}
};

struct Heaps {

	float **val;
	int **heap;
	float **variation;
	int *p2seg;

	Heaps(float **val, int **heap, float **variation, int *p2seg) :
		val(val), heap(heap), variation(variation), p2seg(p2seg)
	{}

};

struct DevicePointers {

	UINT32 n_blocks;//n of block

	UINT32 n; //n of pts

	UINT32 slotsize;

	float *d_lineinfo;//line information

	int *d_buckets; float *segments;//LSH

	int *heap; float *val; float *variation;//outputs

	int *lkd; int *id;//Kdtree

	int* d_output;

	unsigned char *ptavail;

	unsigned char *searched;

	uchar4* d_bucketsforpt;

	DevicePointers(
		UINT32 n_blocks,
		UINT32 n, UINT32 slotsize,
		float *d_lineinfo,
		float *segments, int *d_buckets, int* d_output,
		int *heap, float *val, float *variation, 
		int *lkd, int*id, unsigned char *searched,
		uchar4* d_bucketsforpt, unsigned char *ptavail) : 
		d_bucketsforpt(d_bucketsforpt),
		n_blocks(n_blocks), n(n), d_lineinfo(d_lineinfo), d_output(d_output),
		segments(segments), d_buckets(d_buckets), heap(heap), val(val), searched(searched),
		variation(variation), lkd(lkd), id(id), slotsize(slotsize), ptavail(ptavail)
	{}

};

struct HeapParam {
	int *start;
	int *end;
	int *idx;
	set<int> *res;
};
struct HeapDeletion{
	int start;
	int end;
	int *heap;
	float *variation;
	float *val;
	int *p2seg;
	int *p2line;
	int* pter;
	int* backward;
	int front;
	int *p_left;
	int left = 0;// to be optimized 
	int t_orig;
	int last;
	int idx;
	float min;
#ifdef _DEBUG
	bool hajime = true;
#endif
	void init(int* heap, float* variation, float* val, int *p_left,
		int* pter, int* backward, int t_orig, int *&p2seg,int *&p2line)
	{
		this->heap = heap;
		this->variation = variation;
		this->val = val;
		this->pter = pter;
		this->backward = backward;
		this->t_orig = t_orig;
		this->p2seg = p2seg;
		this->p_left = p_left;
		this->p2line = p2line;
	}

	//	HeapDeletion() {}
};


struct l_vector
{
	int n_l;
	Vector3 center;
	Vector3 start;
	Vector3 end;
	int offset, length, n_p_offset;
};
struct Mutex {
public:
	bool _val = false;
	vector<pair<void(*)(LPVOID), LPVOID>> _callbacks;
	void(*setAvailPLines)(LPVOID, int);
	LPVOID lParam;
	inline void safeSet(const bool &val) {
		lock.lock();
		_val = val;
		lock.unlock();
	}
	inline void Lock(){
		lock.lock();
	}
	inline void Unlock(){
		lock.unlock();
	}
private:
	std::mutex lock;
};

#endif //CUDAVARIABLESTRUCT
