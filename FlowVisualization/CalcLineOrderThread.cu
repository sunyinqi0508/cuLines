#include "CalcLineOrderThread.h"
#include <iostream>
#include <ctime>
#define GAUSSIAN_SCALE	150
#undef max
#undef min
struct ThreadParam {
	Heaps *heap;
	DevicePointers *device_pointer;
	CalcLineOrderThread* calcLineOrderThread;
	ThreadParam(Heaps *heap, DevicePointers *device_pointer, CalcLineOrderThread* calcLineOrderThread) :
		heap(heap), device_pointer(device_pointer), calcLineOrderThread(calcLineOrderThread)
	{}
};
ThreadParam *threadparam;
//vector<int> *res;
mutex global_order_lock;
//cuda kernel
extern __global__ void UpdateLsh(int *lsh, int target);
extern __global__ void cuDeterminingAvaillines(int *avail, float *val, int n);

extern __global__ void LSH_determining_bucket(uchar4 *bucketsforpt, float *ptinfo, int* output, int n);

extern __global__ void LSHSearch( //output need to be zeroed before first use;

	unsigned char *searched, unsigned char *ptavail,

	int *buckets, float *segments, float *ptinfo,

	uchar4 *bucketsforpt, int *output,

	int n, int slotsize

	);

extern __global__ void KDSearch(

	int *lkd, int*id,

	int *outputs, unsigned short * variation,

	float *lineinfo, unsigned char* ptavail, //ignore negative values in lineinfo.x(s) 

	int n

	);
extern __global__ void VectorizedHashing(

	int *linearsegs, short* offsets,

	int *outputs, unsigned short *variation,

	float *linfo, /* linfo optimized for random access */

	unsigned char* ptavail, //ignore negative values in lineinfo.x(s) 

	int n

);
extern __global__ void CoupledHeapsFiltration(

	float *lineinfo, unsigned char* ptavail,

	int *heap, float *val, float *variation,

	int *outputs, int n//, float *saliency = 0

	);

extern __global__ void local_sim(
	float *sel_pts, int *othersegs, //pt info
	float *out,  //output
	int *lkd, float *vari_data,
	int n, int prob_n //static data
	);
extern void constant_cpy(float *, float);
float genGaussianRandom() {
	float x1, x2;
	do {
		x1 = ((float)rand() / (float)RAND_MAX);
	} while (x1 == 0);
	x2 = ((float)rand() / (float)RAND_MAX);
	return sqrt(-2.0 * log(x1)) * cos(2.0 * PI * x2);
}
template<int buf, bool>	
DWORD WINAPI Replentish(LPVOID);
bool _lt_cmp(const int &i, const int &j) { return (i<j); }
template<class _PtrType>
inline void _Heap_Del_Descend(_PtrType &_heap_start, int _Diff, const int & _Offset,
	const std::remove_reference<decltype(*std::declval<_PtrType>())>::type _Target,
	int *attachment, float *attachment2) {
	int a1_val = *(_Offset + attachment + _Diff);
	float a2_val = *(_Offset + attachment2 + _Diff);

	assert(_Diff > 0);
	if (_Diff <= 0)
		exit(623);
	_PtrType _ptr;
	while ((_Diff <<= 1) <= _Offset)
	{
		_ptr = _heap_start + _Diff;
		if (*(_ptr) > *(_ptr + 1)) {
			_ptr++;
			_Diff++;
		}
		if (*_ptr < _Target)
		{
			*(_heap_start + (_Diff >> 1)) = *_ptr;
			*(attachment + (_Diff >> 1)) = *(attachment + _Diff);
			*(attachment2 + (_Diff >> 1)) = *(attachment2 + _Diff);
		}
		else {
			*(_heap_start + (_Diff >> 1)) = _Target;
			*(attachment + (_Diff >> 1)) = a1_val;
			*(attachment2 + (_Diff >> 1)) = a2_val;
			return;
		}
	}
	*(_heap_start + (_Diff >> 1)) = _Target;
	*(attachment + (_Diff >> 1)) = a1_val;
	*(attachment2 + (_Diff >> 1)) = a2_val;
}

template<class _PtrType>//ptr type
inline void _Heap_Del_Ascend(_PtrType &_heap_start, int _Diff,
	const std::remove_reference<decltype(*std::declval<_PtrType>())>::type _Target,
	int*attachment, float* attachment2) {
	_PtrType _ptr;
	int a1_tar = *(attachment + 1);
	float a2_tar = *(attachment2 + 1);
	while (_Diff > 0)
	{
		_ptr = _heap_start + (_Diff >> 1);
		if (*_ptr > _Target)
		{
			*(_heap_start + _Diff) = *_ptr;
			*(attachment + _Diff) = *(attachment + (_Diff >> 1));
			*(attachment2 + _Diff) = *(attachment2 + (_Diff >> 1));
		}
		else {
			*(_heap_start + _Diff) = _Target;
			*(attachment + _Diff) = a1_tar;
			*(attachment2 + _Diff) = a2_tar;
			return;
		}
		_Diff >>= 1;
	}
	*(_heap_start + 1) = _Target;
	*(attachment + 1) = a1_tar;
	*(attachment2 + 1) = a2_tar;
}

template<class PtrType = float*>
inline void heap_deletion(PtrType _heap_start, PtrType _heap_end, PtrType _Target, int*attachment, float*attachment2) {
	if (*_Target > *_heap_end)
		_Heap_Del_Ascend<PtrType>(--_heap_start, _Target - _heap_start, *(_heap_end--), --attachment, --attachment2);
	else if (*_Target < *_heap_end)
		_Heap_Del_Descend<PtrType>(--_heap_start, _Target - _heap_start, _heap_end - _Target, *(_heap_end--), --attachment, --attachment2);
	else {
		int _Diff = _Target - _heap_start;
		int _Offset = _heap_end - _heap_start;
		attachment[_Diff] = attachment[_Offset];
		attachment2[_Diff] = attachment2[_Offset];
	}
}


template<class PtrType = float*>
inline void heap_pop(PtrType _heap_start, PtrType _heap_end, int* attachment, float* attachment2) {
	*_heap_start = *_heap_end;
	int diff = _heap_end - _heap_start;
	*attachment = *(attachment + diff);
	*attachment2 = *(attachment2 + diff);
	--_heap_end;
	_Heap_Del_Descend<PtrType>(_heap_start, 1, diff, *(_heap_start--), --attachment, --attachment2);
}
int _qsort_cmp(const void *a, const void*b) { return *(UINT32 *)a < *(UINT32 *)b; }

DWORD WINAPI Update(LPVOID param) {
	HeapDeletion* p = (HeapDeletion*)param;
	int target = (p->t_orig);// >> 18;
	int t_orig = (p->t_orig);
	/*float min = INT_MAX;
	int idx = -1;*/
	int left = 0;
	p->front = p->start;
	//p->last = p->end ;
	p->backward[p->start] = -1;
	int *heap = p->heap;
	float *val = p->val;
	float *variation = p->variation;
	for (int i = p->start; i < p->end; i = p->pter[i])
	{
		if ((p->p2line[i]) == target) {
			p->p_left[i] = -1;
			left--;
			if (p->backward[i] >= 0)
			{
				if (p->pter[i] < p->end)
					p->backward[p->pter[i]] = p->backward[i];
				else
					p->last = p->backward[i];

				p->pter[p->backward[i]] = p->pter[i];
			}
			else
			{
				p->front = p->pter[i];
				if (p->pter[i] < p->end)
					p->backward[p->pter[i]] = -1;
				else
					p->last = -1;
			}

			continue;
		}
		int begin = i*HEAP_SIZE, end = begin + p->p_left[i];
		for (int j = begin; j < end; j++) {//3
			while ((heap[j]) == target) {//2
				int tmp = heap[j];
				heap[j] = -1;
				while ((heap[--end]) == target);
				heap[j] = tmp;
				heap_deletion<float*>(&val[begin], &val[end], &val[j], &heap[begin], &variation[begin]);//4
				p->p_left[i] = end - begin;
				if (end - begin <= 0)
					break;
			}
		}

		if (p->p_left[i] <= 0)
		{
			left--;
			if (p->backward[i] >= 0)
			{
				if (p->pter[i] < p->end)
					p->backward[p->pter[i]] = p->backward[i];
				else
					p->last = p->backward[i];

				p->pter[p->backward[i]] = p->pter[i];
			}
			else
			{
				p->front = p->pter[i];
				if (p->pter[i] < p->end)
					p->backward[p->pter[i]] = -1;
				else
					p->last = -1;
			}

		}
	}
	/*p->min = min;
	p->idx = idx;*/
	p->left = left;
	//	p->hajime = false;
	return 0;
}

template<bool exhaust = false>
inline void hDelete(int *p2seg, int *heap, float *val, float *variation, int buf, int *p_left, CalcLineOrderThread *calcT,
	Vector3 *pt = NULL, float percentage = .0f, float r0 = .0f, int heap_num = HEAP_SIZE) {//FP marking impl on CPU code
	static cudaStream_t nonBlockingCpy = cudaStreamDefault;
	if (!nonBlockingCpy)
		cudaStreamCreateWithFlags(&nonBlockingCpy, cudaStreamNonBlocking);
	int n_del = 0;
	int *p_lft_curr = p_left;

	float *sims = new float[calcT->streamlines.size()];
	int *sim_count = new int[calcT->streamlines.size()];
	vector<int> sim_avails;

	vector<int> *res = new vector<int>;
	int _DEBUG_T = 0;
	float min = INT_MAX;
	int idx = -1;
	int time1 = 0, time2 = 0, time3 = 0, time4 = 0;

	int *pter = new int[calcT->N_p + 1], *backward = new int[calcT->N_p + 1];
	int t_orig = 0;

	HeapDeletion hd_param[10];
	for (int i = 0; i < 10; i++) {
		hd_param[i].init(heap, variation, val, p_left, pter, backward, t_orig, p2seg, calcT->p2line);
	}


	int i_list = 0;
	int front = 0, last = calcT->N_p, last_pos = -1;


	while (i_list <= calcT->N_p&& !calcT->ptAvailable[i_list])i_list++;

	front = i_list;

	int left = 0;
	while (i_list < calcT->N_p) {

		while (i_list < calcT->N_p && !calcT->ptAvailable[i_list])i_list++;


		if (last_pos >= 0) {
			backward[i_list] = last_pos;
			pter[i_list] = i_list + 1;
			pter[last_pos] = i_list;
			left++;
			i_list++;
		}

		for (; i_list < calcT->N_p; i_list++)
		{
			if (!calcT->ptAvailable[i_list])
			{
				i_list--;
				break;
			}
			left++;
			backward[i_list] = i_list - 1;
			pter[i_list] = i_list + 1;
		}
		last_pos = i_list;


		++i_list;

	}

	last = last_pos;
	pter[last_pos - 1] = last_pos + 1;


	HANDLE handles[10];
	//for (int i = front; i < last; i = pter[i])
	//{
	//	int offset = HEAP_SIZE *i;
	//	if (val[offset] > 0 && variation[offset] < min)
	//	{
	//		min = variation[offset];
	//		idx = offset;
	//	}
	//}
	hd_param[9].end = last;

	SetThreadPriority(GetCurrentThread(), THREAD_BASE_PRIORITY_MAX);

	while (exhaust || res->size() < n_del) {
		_DEBUG_T++;
		time3 -= clock();

		std::fill(sim_count, sim_count + calcT->streamlines.size(), 0);
		min = INT_MAX;
		idx = -1;
		for (int i = front; i < last; i = pter[i])
		{
			int offset = HEAP_SIZE *i;
			if (val[offset] > 0 && p_left[i] >0)
			{
				int line = calcT->p2line[i];

				if (sim_count[line] == 0)
				{
					sim_avails.push_back(line);
					sims[line] = variation[offset];
				}
				else
				{
					sims[line] += variation[offset] * pow(64-p_left[i],.1);
				}
				sim_count[line] ++;
			}
		}
		for each (int line in sim_avails)
		{
			float current_sim = //sims[line] / sim_count[line];
			(sims[line] + calcT->streamlines[line].path.size() - sim_count[line]) / sim_count[line];

			current_sim += .1*(calcT->streamlines[line].path.size() - sim_count[line]);
			if (min > current_sim) {
				min = current_sim;
				idx = line;
			}
		}

		if (idx == -1) {

		//	cout << time1 << endl << time2 << endl << time3 + clock() << endl << time4 << endl;
			calcT->buffer_flags[buf].safeSet(false);

			global_order_lock.lock();
			calcT->global_order.insert(calcT->global_order.end(), res->begin(), res->end());
			/*for (int i = 0; i < res->size(); i++){
				if (!calcT->bool_arr[(*res)[i]]){
					calcT->bool_arr[(*res)[i]] = true;
					calcT->global_order.push_back((*res)[i]);
				}
			}*/
			global_order_lock.unlock();
			emit calcT->sendLineOrder(calcT->global_order, 1);

			int ptavail_size = (calcT->N_p + 7) / 8;
			CBits *bits = new CBits[ptavail_size];
			int i;
			for (i = 0; i < calcT->N_p - 8; i += 8)
				bits[i / 8] = CBits(calcT->ptAvailable + i);

			unsigned char tmp = 0;
			for (; i < calcT->N_p; i++)
			{
				tmp <<= 1;
				tmp += calcT->ptAvailable[i];
			}
			if (tmp)
				bits[i / 8].byte = tmp;
			cudaMemcpyAsync(calcT->ptavail, bits, sizeof(unsigned char)*ptavail_size, cudaMemcpyHostToDevice, nonBlockingCpy);


			//qDebug() << "11111111111111sss:" << calcT->global_order.size();

			/*vector<int> tmp_Ndelete = calcT->global_order;
			qSort(tmp_Ndelete.begin(), tmp_Ndelete.end());
			QString str_t = "";
			for (int i = 0; i < tmp_Ndelete.size(); i++)
			{
				str_t.append(QString::number(tmp_Ndelete[i]) + ",");
			}*/
			
			//qDebug() << str_t;
			//memcpy 
			static int deleted = 0;
			if (false&&calcT->streamlines.size() - calcT->global_order.size() > 0 && deleted!=calcT->streamlines.size() - calcT->global_order.size())
			{
				deleted = calcT->streamlines.size() - calcT->global_order.size();
				//qDebug() << "Replentish 1 start";
				Replentish<0, false>(threadparam);
				//qDebug() << "Replentish 1 end:" << calcT->global_order.size();
				//emit calcT->sendLineOrder(calcT->global_order, 1);
			}
			return;// new vector<int>(res->begin(), res->end());
		}
		else
		{
			//UPDATE LSH
			UpdateLsh << <25, 32 >> > (calcT->d_lb, heap[idx]);
			//std::cout << cudaGetErrorName(cudaGetLastError())<< std::endl;
			std::fill(calcT->ptAvailable + calcT->streamlines[idx].path[0].id,
				calcT->ptAvailable + calcT->streamlines[idx].path.back().id, false);

		}
		res->push_back(idx);
		if (res->size() > 30)
		{
		
			calcT->global_order.insert(calcT->global_order.end(), res->begin(), res->end());
			res->clear();
			emit calcT->sendLineOrder(calcT->global_order, 1);

		}

		int target = idx;// >> 18;
		t_orig = idx;
		//int heap_rear = idx + p_left[buf][t];
		//if (p_left[buf][t] >0)
		//	heap_pop<float*>(&val[idx], &val[heap_rear], &heap[idx], &variation[idx]);
		time3 += clock();
		time2 -= clock();

#ifdef _DEBUG
		int d_size = res->size();
#endif
		if (left >= 64)
		{
			int stepping = left / 10 + left % 10, n = 0, j = 0;
			hd_param[0].start = front;
			for (int i = front; i < last; i = pter[i])
			{

				if (n++ > stepping)//<continue
				{
					if (j >= 10)
						exit(2133);
					stepping += left / 10;
					hd_param[j].end = i;
					hd_param[j].last = backward[i];
					hd_param[j].left = 0;
					//hd_param[j].idx = -1;
					hd_param[j].t_orig = t_orig;
					hd_param[j + 1].start = i;
					hd_param[j].p2line = calcT->p2line;
					handles[j] = CreateThread(0, 0, Update, &hd_param[j], 0, 0);
					j++;
				}
			}
			if (j == 9) {
				hd_param[9].end = last + 1;
				hd_param[9].last = last;
				hd_param[9].left = 0;
				hd_param[9].t_orig = t_orig;
				//hd_param[9].idx = -1;

				hd_param[9].p2line = calcT->p2line;
				handles[9] = CreateThread(0, 0, Update, &hd_param[9], 0, 0);
			}//for (int i = 0; i < 8; i++)
			bool runonce = 1;
			for (int i = 0; i < 10; i++)
			{
				WaitForSingleObject(handles[i], INFINITE);
				if (hd_param[i].last != -1)
				{
					if (runonce) {
						front = hd_param[i].front;
						backward[front] = -1;
						runonce = 0;
					}
				}
				else last = hd_param[i].last;
				if (hd_param[i].left > 0)
					exit('f');
				left += hd_param[i].left;
				/*if (hd_param[i].idx >= 0)
				{
					if (hd_param[i].min < min)
					{
						min = hd_param[i].min;
						idx = hd_param[i].idx;
					}
				}*/
			}

			for (int i = 1; i < 10; i++)
			{
				int j = i - 1;
				while (j > 0 && hd_param[j].last == -1)j--;

				backward[hd_param[i].front] = hd_param[j].last;
			}
			for (int i = 0; i < 9; i++)
			{
				int j = i + 1;
				while (j < 9 && hd_param[j].last == -1)j++;
				pter[hd_param[i].last] = hd_param[j].front;

			}
		}
		else
		{
			for (int i = front; i < last; i = pter[i])//pter[i] = i
			{
				if ((p2seg[i]) == target) {
					if (backward[i] >= 0)
					{
						backward[pter[i]] = backward[i];
						pter[backward[i]] = pter[i];
					}
					else
					{
						front = pter[i];
						backward[pter[i]] = -1;
					}
					if(pter[i] > last) {
						last = backward[i] + 1;// backward[i];
					}
					
					p_left[i] = -1;
					//left--;
					continue;
				}
				int begin = i*HEAP_SIZE, end = begin + p_lft_curr[i];
				for (int j = begin; j < end; j++) {
					while ((heap[j]) == (target)) {
						int tmp = heap[j];
						heap[j] = -1;
						while ((heap[--end]) == (target));
						heap[j] = tmp;
						heap_deletion<float*>(&val[begin], &val[end], &val[j], &heap[begin], &variation[begin]);
						p_lft_curr[i] = end - begin;

					}
				}

				//time4 += clock();
				if (p_lft_curr[i] > 0)
				{
					/*if ((begin / HEAP_SIZE < last) && val[begin] > 0 && variation[begin] < min)
					{
						min = variation[begin];
						idx = begin;
					}*/
				}
				else
				{
					if (backward[i] >= 0)
					{
						backward[pter[i]] = backward[i];
						pter[backward[i]] = pter[i];
					}
					else
					{
						front = pter[i];
						backward[pter[i]] = -1;
					}
				}
			}
		}
		time2 += clock();
	}
	return;// new vector<int>(res->begin(), res->end());
}

template<int buf>
inline void Exhaust(Heaps *info,CalcLineOrderThread *calcT) {
	hDelete<true>
		(info->p2seg, info->heap[buf], info->val[buf], info->variation[buf], buf, calcT->p_left[buf], calcT);//enum?
	//return res->size();
}

template<int buf, bool first = true>
DWORD WINAPI Replentish(LPVOID lparam) {
	//int *buf = (int*)param;
	ThreadParam *param = (ThreadParam *)lparam;
	Heaps *heap = param->heap;
	DevicePointers *device_pointers = param->device_pointer;
	/*if (first)
	{
		float **val_tmp = new float*[BUFFERS], **variation_tmp = new float *[BUFFERS];
		val_tmp[0] = new float[param->calcLineOrderThread->N_p*HEAP_SIZE];
		variation_tmp[0] = new float[param->calcLineOrderThread->N_p*HEAP_SIZE];
		cudaMemcpy(val_tmp[0], device_pointers->val, sizeof(float)  * (device_pointers->n)* HEAP_SIZE, cudaMemcpyDeviceToHost);
		cudaMemcpy(variation_tmp[0], device_pointers->variation, sizeof(float) * (device_pointers->n) * HEAP_SIZE, cudaMemcpyDeviceToHost);
	*/	
	//}
	/*float **val_tmp = new float*[BUFFERS], **variation_tmp = new float *[BUFFERS];
	val_tmp[0] = new float[param->calcLineOrderThread->N_p*HEAP_SIZE];
	variation_tmp[0] = new float[param->calcLineOrderThread->N_p*HEAP_SIZE];
	cudaMemcpy(val_tmp[0], device_pointers->val, sizeof(float)  * (device_pointers->n)* HEAP_SIZE, cudaMemcpyDeviceToHost);
	cudaMemcpy(variation_tmp[0], device_pointers->variation, sizeof(float) * (device_pointers->n) * HEAP_SIZE, cudaMemcpyDeviceToHost);
	emit param->calcLineOrderThread->sendValVariation(val_tmp, variation_tmp);
	return 0;
	*/
	//gpu calculation
	/*
	LSHSearch << < (device_pointers->n_blocks), THREADS_PER_BLOCK >> >(

		device_pointers->searched, device_pointers->ptavail, device_pointers->d_buckets, device_pointers->segments,

		device_pointers->d_lineinfo, device_pointers->d_bucketsforpt, device_pointers->d_output, 

		device_pointers->n, device_pointers->slotsize

		);
	cudaDeviceSynchronize();
	cout << cudaGetErrorName(cudaGetLastError()) << endl;
	KDSearch << <(device_pointers->n_blocks), THREADS_PER_BLOCK >> >(

		device_pointers->lkd, device_pointers->id, device_pointers->d_output, device_pointers->d_lineinfo,

		device_pointers->ptavail, device_pointers->n

		);
	cudaDeviceSynchronize();
	cout << cudaGetErrorName(cudaGetLastError()) << endl;
	CoupledHeapsFiltration << <(device_pointers->n_blocks), THREADS_PER_BLOCK >> >(

		device_pointers->d_lineinfo, device_pointers->ptavail, device_pointers->heap, device_pointers->val,

		device_pointers->variation, device_pointers->d_output, device_pointers->n

		);
		*/

//	cudaDeviceSynchronize();
	//qDebug() << cudaGetErrorName(cudaGetLastError());
	cudaMemcpy(heap->val[buf], device_pointers->val, sizeof(float)  * (device_pointers->n)* HEAP_SIZE, cudaMemcpyDeviceToHost);
	//qDebug() << cudaGetErrorName(cudaGetLastError());
	cudaMemcpy(heap->variation[buf], device_pointers->variation, sizeof(float) * (device_pointers->n) * HEAP_SIZE, cudaMemcpyDeviceToHost);
//	qDebug() << cudaGetErrorName(cudaGetLastError());
	cudaMemcpy(heap->heap[buf], device_pointers->heap, sizeof(int)  * (device_pointers->n) * HEAP_SIZE, cudaMemcpyDeviceToHost);
	QueryPerformanceCounter(&end_precise);
	cout << "Kernel Time: " << end_precise.QuadPart - start_precise.QuadPart << "/" << freq.QuadPart << " = " << (double)(end_precise.QuadPart - start_precise.QuadPart) / (double)freq.QuadPart << "s" << endl;
	//check
	//init
	qDebug() << cudaGetErrorName(cudaGetLastError());
	emit param->calcLineOrderThread->sendDevicePointers(device_pointers);
	
	int *d_pleft;
	cudaMalloc(&d_pleft, param->calcLineOrderThread->N_p * sizeof(int));
	cuDeterminingAvaillines << <(param->calcLineOrderThread->N_p + 1023) / 1024, 1024 >> >(d_pleft, device_pointers->val, param->calcLineOrderThread->N_p);
	cudaMemcpy(param->calcLineOrderThread->p_left[buf], d_pleft, param->calcLineOrderThread->N_p  * sizeof(int), cudaMemcpyDeviceToHost);
	
	//Exhaustion of lines
	param->calcLineOrderThread->buffer_flags[buf].safeSet(false);


	qDebug() << "Exhaust start------------------";

	Exhaust<0>(heap, param->calcLineOrderThread);
	qDebug() << "Exhaust end------------------";
	


	bool *avail = new bool[param->calcLineOrderThread->streamlines.size()];
	std::fill(avail, avail + param->calcLineOrderThread->streamlines.size(), 1);
	auto order = param->calcLineOrderThread->global_order.data();
	for (int i = 0; i < param->calcLineOrderThread->global_order.size(); i++) {
		avail[order[i]] = 0;
	}
	for (int i = 0; i < param->calcLineOrderThread->streamlines.size(); i++) {
		if (avail[i])
			param->calcLineOrderThread->global_order.push_back(i);
	}

	FILE *forder, *forder_txt;
	char _buffer[128];
	//time_t a;
	time_t curr_time;
	time(&curr_time);
	
	_itoa_s((int)curr_time, _buffer, 16);
	string str(_buffer);
	fopen_s(&forder, ("order_"+str+".dat").c_str(), "w");
	fopen_s(&forder_txt, ("order_"+str+".txt").c_str(), "w");
	int *buf_order = param->calcLineOrderThread->global_order.data();
	fwrite(buf_order, sizeof(int), param->calcLineOrderThread->global_order.size(), forder);
	for (int i = 0; i < param->calcLineOrderThread->global_order.size(); i++)
		fprintf_s(forder_txt, "%d ", buf_order[i]);
	fclose(forder);
	fclose(forder_txt);

	if (param->calcLineOrderThread->streamlines.size() != param->calcLineOrderThread->global_order.size())
		cout << "Error: Size Mismatch" << param->calcLineOrderThread->streamlines.size() - param->calcLineOrderThread->global_order.size()<<endl;
		//delete[] avail;
#ifdef TXTOPT
	ofstream of;
	char buffer[6];
	float orig_minx = param->calcLineOrderThread->orig_minx, 
		orig_miny = param->calcLineOrderThread->orig_miny;
	Line* _Streamline_Data = param->calcLineOrderThread->streamlines.data();

	for (int i = 100; i < param->calcLineOrderThread->streamlines.size(); i += 100) {
		_itoa_s(i, buffer, 6, 10);
		of.open(string(buffer).append(".txt"));
	//	of << "#acid;x;y;fl;cl" << endl;
		int j = 0;
		auto it = param->calcLineOrderThread->global_order.end(),
			begin = param->calcLineOrderThread->global_order.end() - i;
		while (it != begin) {
			auto it_line = _Streamline_Data[*it].path.begin(), end_line = _Streamline_Data[*it].path.end();
			while (it_line != end_line)
			{
				of << *it + 1 << ";" << it_line->x + orig_minx << ";"
					<< it_line->y + orig_miny << ";" << (float)0 << ";0;0" << endl;
				it_line++;
			}
			it--;
			j++;
		}
		of.close();
	}
#endif

	param->calcLineOrderThread->buffer_flags[buf].safeSet(true);
	

	//cope with callback funcs
	param->calcLineOrderThread->buffer_flags[buf].Lock();
	for each (pair<void(*)(LPVOID), LPVOID> func in param->calcLineOrderThread->buffer_flags[buf]._callbacks)
	{
		func.first(func.second);
	}
	param->calcLineOrderThread->buffer_flags[buf]._callbacks.clear();
	param->calcLineOrderThread->buffer_flags[buf].Unlock();//may put in other threads;

	//Return Available lines
	//return 0;
	/*param->calcLineOrderThread->buffer_flags[buf].Lock();
	if (param->calcLineOrderThread->buffer_flags[buf].setAvailPLines)
		param->calcLineOrderThread->buffer_flags[buf].setAvailPLines(param->calcLineOrderThread->buffer_flags[buf].lParam, availLines);
	param->calcLineOrderThread->buffer_flags[buf].Unlock();*/
	//decide next replentish
	return 0;
}

CalcLineOrderThread::CalcLineOrderThread(vector<Line> &streamlines){
	moveToThread(this);
	this->streamlines = streamlines;
	//Nul a;
	
	
	qDebug() << "streamlines size in new thread :" << streamlines.size();
}
CalcLineOrderThread::~CalcLineOrderThread(){
	this->quit();
	this->wait();
}

void CalcLineOrderThread::run(){
	clock_t start, finish;
	start = clock();
	makeData();
	finish = clock();
	qDebug() << "make data in thread time:" << (finish - start);

	exec();
}
void CalcLineOrderThread::calcCurrLineOrder(vector<int> line_sel, vector<int> all_sel) {
	qDebug() << "calcCurrLineOrder:" << line_sel.size() << "," << all_sel.size();
	bool found;

	vector<int> segs_not_sel;
	vector<float> allp;
	vector<int>::iterator it_seg;
	vector<int> seg2line_local;
	int i_line = 0;
	for each (int line in all_sel)
	{
		found = false;
		for each (int sel_line in line_sel)
		{
			if (line == sel_line)
			{
				found = true;
				break;
			}
		}

		if (!found)
		{
			it_seg = line_division[line].begin();
			while (it_seg != line_division[line].end())
			{
				segs_not_sel.push_back(linearKD_id[*it_seg++]);
				seg2line_local.push_back(i_line);
			}


		}

		i_line++;
	}// to be optimized.
	//	segs_not_sel.data();

	vector<Vector3>::iterator it;
	for each (auto sel_line in all_sel)
	{
		it = streamlines[sel_line].path.begin();
		while (it != streamlines[sel_line].path.end())
		{
			allp.push_back(it->x);
			allp.push_back(it->y);
			allp.push_back(it->z);
			it++;
		}
	}

	int left_size = all_sel.size();
	float *d_pts, *d_out;
	bool *l_flags = new bool[left_size];
	int *d_segnsel, *seg2line_local_data = seg2line_local.data();
	UINT32 *out = new UINT32[segs_not_sel.size()];

	std::fill(l_flags, l_flags + left_size, true);

	cudaMalloc(&d_pts, allp.size() * sizeof(float));
	cudaMemcpy(d_pts, allp.data(), sizeof(float)*allp.size(), cudaMemcpyHostToDevice);
	cudaMalloc(&d_segnsel, segs_not_sel.size() * sizeof(int));
	cudaMemcpy(d_segnsel, segs_not_sel.data(), segs_not_sel.size() * sizeof(int), cudaMemcpyHostToDevice);
	cudaMalloc(&d_out, sizeof(float)*segs_not_sel.size());

	int problem_size = allp.size() / 3;
	int seg_size = segs_not_sel.size();
	cudaMemset(d_out, INT_MAX, segs_not_sel.size()*sizeof(float));
	local_sim << <(problem_size + 64) / 64, 64, 8192 / 2 >> > (
		d_pts, d_segnsel,//inputs 
		d_out, //outputs
		d_lkd, d_varidata,//const
		problem_size, seg_size
		);

	cudaMemcpy(out, d_out, segs_not_sel.size() * sizeof(float), cudaMemcpyDeviceToHost);
	//cout << cudaGetErrorName(cudaGetLastError()) << endl;
	//DWithID *outputs = new DWithID[seg_size];
	UINT32 mask = UINT32_MAX - (1 << ((unsigned)(log2(seg_size) + 1))) + 1;
	for (int i = 0; i < seg_size; i++)
	{
		out[i] <<= 1;
		out[i] &= mask;
		out[i] |= i;
	}
	qsort(out, seg_size, sizeof(UINT32), _qsort_cmp);
	//Ndelete = new vector<int>;

	mask ^= UINT32_MAX;
	vector<int> lineOrder;
	for (int i = 0; i < seg_size; i++)
	{
		int tmp = seg2line_local_data[out[i] & mask];//getting segs rather than lines
		if (l_flags[tmp])
		{
			l_flags[tmp] = false;
			lineOrder.push_back(all_sel[tmp]);
		}
	}
	//seg2line


	delete[] out;
	delete[] l_flags;
	cudaFree(d_pts);
	cudaFree(d_segnsel);
	cudaFree(d_out);
	
	
	//bool *tmp_bool = new bool[streamlines.size()];
	//fill(tmp_bool, tmp_bool + streamlines.size(), 0);
	//for (int i = 0; i < lineOrder_part_tmp.size(); i++){
	//	if (!tmp_bool[lineOrder_part_tmp[i]]){
	//		tmp_lineOrder.push_back(lineOrder_part_tmp[i]);
	//		tmp_bool[lineOrder_part_tmp[i]] = true;
	//	}
	//}
	////delete[] tmp_bool;

	emit sendLineOrder(lineOrder, 2);
}
float* CalcLineOrderThread::g_saliency = 0;
void CalcLineOrderThread::makeData(){
	return;

	/*bool_arr = new bool[streamlines.size()];
	fill(bool_arr, bool_arr + streamlines.size(), 0);*/

	linearKd = new int[200000000];// 800MB 
	for (int i = 0; i < streamlines.size(); i++)
		N_p = N_p + streamlines[i].path.size();
	int *p2seg = new int[N_p];
	p2line = new int[N_p];

	int n_iter = 0, li = 0;
	for each (auto var in streamlines)
	{
		li++; 
		for each (auto v in var.path)
		{
			p2line[n_iter++] = li - 1;
		}
	}

	int idx_pt = 0, i = 0;
	line_division = new vector<int>[streamlines.size()];

	auto dit = streamlines.begin();
	int index_pt = 0;
	while (dit != streamlines.end()) {

		auto it = dit->path.begin();
		while (it != dit->path.end()) {
			it->id = index_pt++;
			it++;
		}
		dit++;
	}
	vector<l_vector>  parts = splitlines(p2seg, 1.10);//__test__:3 分段

	seg2line = new int[parts.size()];
	i = 0;
	for each (l_vector var in parts)
	{
		seg2line[i++] = var.n_l;
	}

	float *segmentations, *d_seg, *dists[8];
	vector<int> buckets[BUCKET_NUM][8];
	int n_seg = parts.size();
	float min[8] = { INT_MAX }, max[8] = { INT_MIN }, t, w[8];
	Vector3 h[8];
	int  j;
	float lb[8];

	for (i = 0; i < 8; i++)
		for (j = 0; j < 3; j++)
			h[i] = Vector3(genGaussianRandom(), genGaussianRandom(), genGaussianRandom());

	segmentations = new float[n_seg * 8];
	for (j = 0; j < 8; j++)
		dists[j] = new float[n_seg];
	i = 0;

	float *hashRecords = new float[parts.size() * 10];
	int *i_hr = (int*)hashRecords;
	short *Vector_buckets = new short[N_p * 3];
	size_t pointer_vectorbuckets = 0;

	
	int pointer_hr = 0;

	for each (l_vector var in parts)
	{
		segmentations[i++] = var.start.x;
		segmentations[i++] = var.start.y;
		segmentations[i++] = var.start.z;
		Vector3 vec = { var.end.x - var.start.x , var.end.y - var.start.y , var.end.z - var.start.z };
		memcpy(segmentations, &vec, sizeof(float) * 3);
		i += 3;

		segmentations[i] = pow(segmentations[i - 3], 2);
		segmentations[i] += pow(segmentations[i - 2], 2);
		segmentations[i] += pow(segmentations[i - 1], 2);
		segmentations[i] = sqrt(segmentations[i]);
		i++;
		//struct Projection
		//{

		//	int ptbias;
		//	float projection = numeric_limits<float>::max();
		//	Projection(int ptbias, float projection) :
		//		ptbias(ptbias), projection(projection) {}
		//	Projection() = default;

		//	bool operator<(const Projection& proj) const {
		//		return projection < proj.projection;
		//	}
		//	float operator-(const Projection& proj) const {
		//		return this->projection - proj.projection;
		//	}
		//	operator float&() {
		//		return projection;
		//	}

		//};
		//vec3* this_seg = streamlines[var.n_l].path.data() + var.offset;
		//vec3 unit_vec = vec / vec.length();
		//Projection* projections = new Projection[var.length + 1];
		//int baseid = var.start.id;
		//for (int k = 0; k < var.length; k++) {

		//	float projection = (var.start - this_seg[k]).dot(unit_vec);
		//	projections[k].ptbias = k;
		//	projections[k].projection = projection;

		//}
		//std::sort(projections, projections + var.length);
		//float maxstep = -1;
		//for (int k = 1; k < var.length; k++) {
		//	float thisstep = projections[k].projection - projections[k - 1].projection;
		//	maxstep = thisstep > maxstep ? thisstep : maxstep;
		//}
		//if (maxstep < 0) {
		//	printf("error mStep = %f < 0\n", maxstep);
		//	maxstep = numeric_limits<float>::max(); //all pts goes to b0
		//}
		//else
		//	maxstep += numeric_limits<float>::epsilon();

		//float length_project = projections[var.length - 1] - projections[0];
		//short *this_vb = Vector_buckets + pointer_vectorbuckets;

		//int n_bucket = (length_project  + .5f)/ maxstep;
		//int pointer_bucket = 1;
		//*(unsigned short*)this_vb = n_bucket;
		//for (int k = 0; k < var.length; k++) {
		//	if (projections[k] - projections[0] > pointer_bucket*maxstep)
		//		this_vb[pointer_bucket++] = k;
		//	this_vb[n_bucket + 1 + k] = projections[k].ptbias;
		//}
		//((unsigned short*)this_vb)[n_bucket] = var.length;
		//memcpy(hashRecords + pointer_hr * 10, var.start, sizeof(float) * 3);
		//memcpy(hashRecords + pointer_hr * 10 + 3, unit_vec, sizeof(float) * 3);
		//i_hr[pointer_hr * 10 + 6] = baseid;
		//hashRecords[pointer_hr * 10 + 7] = projections[0];
		//hashRecords[pointer_hr * 10 + 8] = maxstep;
		//i_hr[pointer_hr * 10 + 9] = pointer_vectorbuckets;

		//pointer_vectorbuckets += n_bucket + 1 + var.length;
		//pointer_hr++;
		//delete[] projections;
		for (j = 0; j < 8; j++)
		{
			dists[j][i / 7 - 1] = var.center.dot(h[j]);
			max[j] = max[j] > dists[j][i / 7 - 1] ? max[j] : dists[j][i / 7 - 1];
			min[j] = min[j] < dists[j][i / 7 - 1] ? min[j] : dists[j][i / 7 - 1];
		}
	}
	int *d_hr;
	short *d_vb;
	cudaMalloc(&d_hr, pointer_hr * 10 * sizeof(float));
	cudaMalloc(&d_vb, pointer_vectorbuckets * sizeof(short));
	cudaMemcpy(d_hr, hashRecords, pointer_hr * 10 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_vb, Vector_buckets, pointer_vectorbuckets * sizeof(short), cudaMemcpyHostToDevice);

	for (j = 0; j < 8; j++)
	{
		w[j] = (max[j] - min[j]) / BUCKET_NUM;
		lb[j] = ((double)rand() / (double)RAND_MAX)*w[j];
		lb[j] /= 2.0;
		lb[j] = rand() >= (RAND_MAX / 2) ? lb[j] : -lb[j];
	}

	for (i = 0; i < n_seg; i++)
		for (j = 0; j < 8; j++)
		{
			t = ((dists[j][i] - min[j]) + lb[j]) / w[j];
			t = (int)t;
			if (t < 0)
				t = 0;
			else if (t >= BUCKET_NUM)
				t = BUCKET_NUM - 1;
			buckets[(int)t][j].push_back(i);
		}

	int *buffer = new int[n_seg * 5], *linearBuckets = new int[801 +n_seg * 40];
	int id_buf, k, m;
	l_vector tmp;
	bool found;
	int lbOffset = BUCKET_NUM * 8 + 1;
	for (i = 0; i < BUCKET_NUM; i++)
		for (j = 0; j < 8; j++)
		{
			id_buf = 0;
			for each (int var in buckets[i][j])
			{
				found = false;
				tmp = parts[var];
				for (k = 0; k < id_buf;)
				{
					if (buffer[k] == tmp.n_l)
					{
						found = true;
						break;
					}
					else
					{
						k += buffer[k + 1] + 2;
						if (buffer[k] == -1)
							break;
					}
				}
				if (found)
				{
					buffer[k + 1] ++;
					k += buffer[k + 1] + 1;
					for (m = id_buf; m > k; m--)
						buffer[m] = buffer[m - 1];
					buffer[k] = var;
					id_buf++;
					buffer[id_buf] = -1;
				}
				else
				{
					buffer[id_buf++] = tmp.n_l;
					buffer[id_buf++] = 1;
					buffer[id_buf++] = var;
					buffer[id_buf] = -1;
				}
			}
			linearBuckets[j * 100 + i] = lbOffset;
			for (k = 0; k < id_buf; k++)
				linearBuckets[lbOffset + k] = buffer[k];
			lbOffset += id_buf;
		}
	linearBuckets[BUCKET_NUM * 8] = lbOffset;

	delete[] buffer;

	//int *d_lb;
	cudaMalloc(&d_lb, lbOffset * sizeof(int));
	cudaMemcpy(d_lb, linearBuckets, lbOffset * sizeof(int), cudaMemcpyHostToDevice);
	d_buckets_g = d_lb;
	cudaMalloc(&d_seg, parts.size() * 7 * sizeof(float));
	cudaMemcpy(d_seg, segmentations, parts.size() * 7 * sizeof(float), cudaMemcpyHostToDevice);

	delete[] linearBuckets;
	delete[] segmentations;
	float *linearhash;// , *d_hash;
	int n_lh = 0;
	linearhash = new float[48];
	for (i = 0; i < 8; i++)
	{
		linearhash[n_lh++] = h[i].x;
		linearhash[n_lh++] = h[i].y;
		linearhash[n_lh++] = h[i].z;
		linearhash[n_lh++] = lb[i];
		linearhash[n_lh++] = min[i];
		linearhash[n_lh++] = w[i];
	}

	//float *linearKD;

	lkd_idx = 0;
	int *lkd_idxs = new int[parts.size()], n_lkdidx = 0, n_lkd = 0;
	int *lkd_bak = linearKd;
	for each (l_vector var in parts)
	{
		vector<Vector3>::iterator begin = streamlines[var.n_l].path.begin() + var.offset;
		vector<Vector3>::iterator end = begin + var.length + 1;
		priority_queue<KD_Tree::KD_Vector3, vector<KD_Tree::KD_Vector3>, KD_Tree::cmp<0>> pq;
		int i = 0;
		KD_Tree::KD_Vector3 tmp;
		while (begin != end) {
			tmp.x = begin->x;
			tmp.y = begin->y;
			tmp.z = begin->z;

			tmp.id = var.n_p_offset + i++;// var.offset + i++;
			pq.push(tmp);

			begin++;
		}
		KD_Tree::build<0>(pq, 0, linearKd, lkd_idx);
		lkd_idxs[n_lkdidx++] = n_lkd;
		linearKd += lkd_idx;
		n_lkd += lkd_idx;

		lkd_idx = 0;
	}
	linearKd = lkd_bak;

	cudaMalloc(&d_id, parts.size() * sizeof(int));
	cudaMemcpy(d_id, lkd_idxs, parts.size() * sizeof(int), cudaMemcpyHostToDevice);	

	linearKD_id = lkd_idxs;
	emit sendParameters(line_division, linearKd, linearKD_id);

	float max_term1 = -1, max_term2 = -1;
	float min_term1 = 999999999, min_term2 = 99999999;

	Term1.resize(streamlines.size());
	Term2.resize(streamlines.size());

	Dsim.resize(streamlines.size());

	for (i = 0; i < streamlines.size(); i++)
		Dsim[i].resize(streamlines[i].path.size());

	int pt_size = 0;
	for each (auto _streamline in streamlines)
	{
		pt_size += _streamline.path.size();
	}
	
	makeMappings();




	for (i = 0; i < streamlines.size(); i++)
		for (j = 0; j < streamlines[i].path.size(); j++)
			n_line.push_back(i);


	int *h_line = new int[n_line.size()];
	for (i = 0; i < n_line.size(); i++)
		h_line[i] = n_line.at(i);

	n_line.resize(0);

	float *h_all = new float[allp.size()];

	for (i = 0; i < allp.size(); i++)
		h_all[i] = allp.at(i);

	//2
	h_start = new int[streamlines.size()];
	h_start[0] = 0;

	for (i = 1; i < streamlines.size(); i++)
		h_start[i] = h_start[i - 1] + streamlines[i - 1].path.size();


	//3
	int *h_length_l = new int[streamlines.size()];
	for (i = 0; i < streamlines.size(); i++)
		h_length_l[i] = streamlines[i].path.size();

	int Nofline = streamlines.size();

	/*float *d_all;
	cudaMalloc((void **)&d_all, allp.size() * sizeof(float));
	cudaMemcpy(d_all, h_all, allp.size() * sizeof(float), cudaMemcpyHostToDevice);

	allp.resize(0);*/

	cudaMalloc(&d_lkd, sizeof(int)*n_lkd);
	cudaMemcpy(d_lkd, linearKd, sizeof(int)*n_lkd, cudaMemcpyHostToDevice);

	float *si;
	float *sj;
	float *siSampled;
	float *sjSampled;
	gap = Nofline;
	block_point = 1e3;

	float *d_val, *d_variation;
	int *d_heap;

	cudaMalloc(&d_heap, N_p *HEAP_SIZE * sizeof(int));
	cudaMalloc(&d_val, N_p *HEAP_SIZE * sizeof(float));
	cudaMalloc(&d_variation, N_p *HEAP_SIZE * sizeof(float));

	ptAvailable = new bool[N_p];
	std::fill(ptAvailable, ptAvailable + N_p, true);

	//cudaMalloc(&d_avail_pt, N_p * sizeof(bool));
	//cudaMemcpy(d_avail_pt, ptAvailable, sizeof(bool)*N_p, cudaMemcpyHostToDevice);

	float *lineinfo = new float[N_p * 3], *linfo = new float[N_p * 3];
	int lineinfo_rev2_idx = 0, linfo_idx = 0;
	float diag;
	max_x = max_y = max_z = INT_MIN;
	min_x = min_y = min_z = INT_MAX;

	for (size_t i = 0; i < streamlines.size(); i++)
	{
		lineinfo[lineinfo_rev2_idx] = -streamlines[i].path[0].x; //mark begining of line
		max_x = max_x > streamlines[i].path[0].x ? max_x : streamlines[i].path[0].x;
		min_x = min_x < streamlines[i].path[0].x ? min_x : streamlines[i].path[0].x;

		lineinfo[lineinfo_rev2_idx + N_p] = streamlines[i].path[0].y;
		max_y = max_y > streamlines[i].path[0].y ? max_y : streamlines[i].path[0].y;
		min_y = min_y < streamlines[i].path[0].y ? min_y : streamlines[i].path[0].y;

		lineinfo[lineinfo_rev2_idx++ + 2 * N_p] = streamlines[i].path[0].z;
		max_z = max_z > streamlines[i].path[0].z ? max_z : streamlines[i].path[0].z;
		min_z = min_z < streamlines[i].path[0].z ? min_z : streamlines[i].path[0].z;
		for (int k = 0; k < 3; k++)
			linfo[linfo_idx++] = streamlines[i].path[0][k];
		for (size_t j = 1; j < streamlines[i].path.size(); j++)
		{
			lineinfo[lineinfo_rev2_idx] = streamlines[i].path[j].x;
			max_x = max_x > streamlines[i].path[j].x ? max_x : streamlines[i].path[j].x;
			min_x = min_x < streamlines[i].path[j].x ? min_x : streamlines[i].path[j].x;
			lineinfo[lineinfo_rev2_idx + N_p] = streamlines[i].path[j].y;
			max_y = max_y > streamlines[i].path[j].y ? max_y : streamlines[i].path[j].y;
			min_y = min_y < streamlines[i].path[j].y ? min_y : streamlines[i].path[j].y;
			lineinfo[lineinfo_rev2_idx++ + 2 * N_p] = streamlines[i].path[j].z;
			max_z = max_z > streamlines[i].path[j].z ? max_z : streamlines[i].path[j].z;
			min_z= min_z < streamlines[i].path[j].z ? min_z : streamlines[i].path[j].z;
			for (int k = 0; k < 3; k++)
				linfo[linfo_idx++] = streamlines[i].path[j][k];
		}
	}
	cout << "bounding box" << endl;
	cout << max_x << " " << min_x << " " << max_y << " " << min_y << " " << max_z << " " << min_z << endl;

	diag = sqrt(pow(max_x - min_x, 2) + pow(max_y - min_y, 2) + pow(max_z - min_z, 2));
	diag *= .003*GAUSSIAN_SCALE;
	constant_cpy(linearhash, diag);
	delete[] linearhash;

	cout << cudaGetErrorName(cudaGetLastError()) << endl;
	float *d_lineinfo, *d_linfo;// lineinfo
	cudaMalloc(&d_lineinfo, sizeof(float) * N_p * 3);
	cudaMalloc(&d_linfo, sizeof(float)* N_p * 3);

	cudaMemcpy(d_lineinfo, lineinfo, sizeof(float) * N_p * 3, cudaMemcpyHostToDevice);
	cudaMemcpy(d_linfo, linfo, sizeof(float) * N_p * 3, cudaMemcpyHostToDevice);
	delete[] lineinfo;
	delete[] linfo;
	cout << cudaGetErrorName(cudaGetLastError()) << endl;

	int *d_output;
	cudaMalloc(&d_output, sizeof(int) * N_p * 128);
	cudaMemset(d_output, 0, sizeof(int)*N_p * 128);
	cout << cudaGetErrorName(cudaGetLastError()) << endl;

	int slotsize = (streamlines.size() + 7) / 8;
	unsigned char* d_searched, *searched = new unsigned char[N_p * slotsize];
	std::fill(searched, searched + N_p * slotsize, 0xff);
	for (int i = 0; i < N_p; i++) {
		searched[N_p * ((p2line[i]) >> 3) + i] &= (unsigned char)(0xff - (1 << ((p2line[i])& 0x7)));
	}
	cudaMalloc(&d_searched, sizeof(unsigned char) * N_p * slotsize);
	cudaMemcpy(d_searched, searched, sizeof(unsigned char) * N_p * slotsize,cudaMemcpyHostToDevice);
	//cudaMemset(d_searched, 0xff, sizeof(unsigned char)*N_p * slotsize);
	cout << cudaGetErrorName(cudaGetLastError()) << endl;

	cudaMalloc(&ptavail, sizeof(unsigned char)*((N_p + 7) / 8));
	cout << cudaGetErrorName(cudaGetLastError()) << endl;
	cudaMemset(ptavail, 0xff, sizeof(unsigned char)*((N_p + 7) / 8));
	cout << cudaGetErrorName(cudaGetLastError()) << endl;

	cudaMalloc(&d_bucketsforpt, 2 * sizeof(uchar4)* N_p);
	cout << cudaGetErrorName(cudaGetLastError()) << endl;

	allp.resize(0);

	h_nj = new int[block_point * gap];



	int nB =  (N_p / THREADS_PER_BLOCK + ((N_p%THREADS_PER_BLOCK) != 0));
	float **val = new float*[BUFFERS], **variation = new float*[BUFFERS];
	int **heap = new int*[BUFFERS];
	for (int i = 0; i < BUFFERS; i++)
	{
		val[i] = new float[N_p * HEAP_SIZE];
		variation[i] = new float[N_p * HEAP_SIZE];
		heap[i] = new int[N_p * HEAP_SIZE];
	}

	int *n_of_points = new int(N_p);

	cudaStreamCreate(&nonBlockingCpy);
//	cudaFuncSetCacheConfig(&CudaSimilarWithLines, cudaFuncCachePreferL1);

	DevicePointers *d_pointers = new DevicePointers(
		nB, N_p, slotsize,
		d_lineinfo,
		d_seg, d_lb,  d_output,
		d_heap, d_val, d_variation,
		d_lkd, d_id, d_searched,
		d_bucketsforpt, ptavail
		);
	Heaps *heaps = new Heaps(val, heap, variation, p2seg);

	ThreadParam *tp = new ThreadParam(heaps, d_pointers, this);
	threadparam = tp;

	for (int i = 0; i < BUFFERS; i++)
		p_left[i] = new int[N_p];
	buffer_flags[0].safeSet(true);
	//new d_bucketforpt d_lineinfo d_output


	LSH_determining_bucket << <nB, THREADS_PER_BLOCK >> >(d_bucketsforpt, d_lineinfo, d_output, N_p);

	cudaDeviceSynchronize();
	cout << cudaGetErrorName(cudaGetLastError()) << endl;
	long time_s = -clock();
	QueryPerformanceFrequency(&freq);
	QueryPerformanceCounter(&start_precise);
	
	//new searched ptavail d_buckets_g d_seg d_lineinfo 
	LSHSearch << <nB, THREADS_PER_BLOCK >> >(d_searched, ptavail, d_buckets_g, d_seg, d_lineinfo, d_bucketsforpt, d_output, N_p, slotsize);
	cudaDeviceSynchronize();

	cout << cudaGetErrorName(cudaGetLastError()) << endl;

	//delete searched d_buckets_g d_seg
	//new d_variation d_id d_lkd  
	KDSearch << <nB, THREADS_PER_BLOCK >> >(d_lkd, d_id, d_output,(unsigned short*)d_variation, d_lineinfo, ptavail, N_p);
	//VectorizedHashing << <nB, THREADS_PER_BLOCK >> >(d_hr, d_vb, d_output, (unsigned short*)d_variation, d_linfo, ptavail, N_p);
	cudaDeviceSynchronize();
	cout << cudaGetErrorName(cudaGetLastError()) << endl;

	//delete d_lkd d_id
	//new d_val d_heap
	//float *d_saliency, *saliency = new float[N_p];
	//cudaMalloc(&d_saliency, sizeof(float) * N_p);
	CoupledHeapsFiltration << <nB, THREADS_PER_BLOCK >> > (d_lineinfo, ptavail, d_heap, d_val, d_variation, d_output, N_p);//, d_saliency);
	cudaDeviceSynchronize();

	cout << cudaGetErrorName(cudaGetLastError()) << endl;

																														   //cudaMemcpy(saliency, d_saliency, sizeof(float)*N_p, cudaMemcpyDeviceToHost);
	//float max_sal = 0;
	//for (int i = 0; i < N_p; i++)
	//	max_sal = max_sal > saliency[i] ? max_sal : saliency[i];
	//max_sal *= .5;
	//for (int i = 0; i < N_p; i++)
	//	saliency[i] /= max_sal;
//g_saliency = saliency;
//del d_output d_lineinfo ptavail d_heap
HANDLE h_tmp = CreateThread(0, 0, Replentish<0>, (LPVOID)tp, 0, 0);

clock_t start, finish;
start = clock();
WaitForSingleObject(h_tmp, INFINITE);
finish = clock();
qDebug() << "Replentish end time==================" << (finish - start);

delete[] val;
delete[] variation;
delete[] heap;
//delete[] lineinfo;
delete[] h_start;
delete[] h_length_l;
delete[] h_all;
for (i = 0; i < 100; i++)
	for (j = 0; j < 8; j++)
		buckets[i][j].clear();

parts.clear();

for (i = 0; i < 8; i++)
	delete[] dists[i];

cudaFree(d_lineinfo);
cudaFree(d_heap);
cudaFree(d_seg);
//cudaFree(d_val);
//cudaFree(d_variation);
cudaFree(d_lb);
//	cudaFree(d_saved);
}
int __forceinline CalcLineOrderThread::BinaryLoacate(int &k) {
	int i = 0, j = streamlines.size() + 1;
	int mid = -1;
	do {
		mid = (i + j) / 2;
		if (idxMappings[mid] > k) {
			j = mid;
		}
		else {
			i = mid;
		}
	} while (i < j - 1);

	return i;
}

void CalcLineOrderThread::makeMappings() {
	idxMappings = new int[streamlines.size() + 1];
	idxMappings[0] = 0;
	for (int i = 1; i <= streamlines.size(); i++) {
		idxMappings[i] = idxMappings[i - 1] + streamlines[i - 1].path.size();
	}
}

Vector3& CalcLineOrderThread::ptMappings(int i)
{
	int ret = BinaryLoacate(i);
	return streamlines[ret].path[i - idxMappings[ret]];
}

Vector3 CalcLineOrderThread::getcenter(int k, int start, int end)
{
	Vector3 a(0, 0, 0);
	for (int i = start; i <= end; i++)
		a = a + streamlines[k].path.at(i);
	a = a / (end - start + 1);
	return a;
}
float CalcLineOrderThread::length_curve(int k, int start, int end)
{
	float l = 0;
	for (int i = start + 1; i <= end; i++)
		l = l + (streamlines[k].path.at(i) - streamlines[k].path.at(i - 1)).length();
	return l;
}
float CalcLineOrderThread::length_vector(int k, int start, int end)
{
	return (streamlines[k].path.at(end) - streamlines[k].path.at(start)).length();
}
vector<l_vector> CalcLineOrderThread::splitlines(int *p2seg, float max_k)
{
	vector<l_vector> parts;
	int pseg_offset = 0, curr_pt = 0;
	for (int i = 0; i < streamlines.size(); i++)
	{
		int start = 0, end = 1;
		float k = 0;

		int j;
		if (i != 0)
			curr_pt += streamlines[i - 1].path.size();
		for (j = 2; j < streamlines[i].path.size(); j++)
		{

			if (k >= max_k)
			{
				if (end + 2 == streamlines[i].path.size())
				{
					end++;
					j++;
				}
				vector<int> *ptsperseg = new vector<int>;
				for (int m = 0; m < end - start + 1; m++, pseg_offset++)
				{
					p2seg[pseg_offset] = i;// (i << 18) + parts.size();
					ptsperseg->push_back(streamlines[i].path.at(start + m).id);
				}
				line_division[i].push_back(parts.size());
				segs2pts.push_back(*ptsperseg);
				l_vector a;
				a.n_l = i;
				a.center = getcenter(i, start, end);
				a.start = a.center + (streamlines[i].path.at(start) - streamlines[i].path.at(end)) / 2;
				a.end = a.center + (streamlines[i].path.at(end) - streamlines[i].path.at(start)) / 2;

				a.offset = start;
				a.n_p_offset = curr_pt + start;

				a.length = end - start;
				parts.push_back(a);
				start = end + 1;

				end += 2;
				j++;
				if (end >= streamlines[i].path.size())
					continue;
				k = length_curve(i, start, end) / length_vector(i, start, end);

			}
			else if (k<max_k)
			{
				end++;
				k = length_curve(i, start, end) / length_vector(i, start, end);
				if (isnan(k) || isinf(k))
					k = 0;
			}

		}
		if (start < end - 1)
		{
			vector<int> *ptsperseg = new vector<int>;
			for (int m = 0; m < end - start + 1; m++, pseg_offset++)
			{
				p2seg[pseg_offset] = i;
				ptsperseg->push_back(streamlines[i].path.at(start + m).id);
			}
			line_division[i].push_back(parts.size());
			segs2pts.push_back(*ptsperseg);
			l_vector a;
			a.n_l = i;
			a.center = getcenter(i, start, end);
			a.start = a.center + (streamlines[i].path.at(start) - streamlines[i].path.at(end)) / 2;
			a.end = a.center + (streamlines[i].path.at(end) - streamlines[i].path.at(start)) / 2;
			a.offset = start;
			a.length = end - start;
			a.n_p_offset = curr_pt + start;

			parts.push_back(a);
		}

	}
	return parts;
}
//vector<l_vector> /*CalcLineOrderThread::*/splitlines_(int *p2seg, float max_k)
//{
//	const double dx2 = pow(max_x - min_x, 2);
//	const double dy2 = pow(max_y - min_y, 2);
//	const double dz2 = pow(max_z - min_z, 2);
//
//	const double datascale = sqrt(dx2 + dy2 + dz2);
//
//	vector<l_vector> parts;
//
//	int pseg_offset = 0, curr_pt = 0;
//	for (int i = 0; i < streamlines.size(); i++)
//	{
//		int start = 0, end = 1;
//		float k = 0;
//
//		int j;
//		if (i != 0)
//			curr_pt += streamlines[i - 1].path.size();
//		float curvelength = 0;
//
//		for (j = 2; j < streamlines[i].path.size(); j++)
//		{
//			if (k >= max_k || curvelength > datascale * .1)
//			{
//				if (end + 2 == streamlines[i].path.size())
//				{
//					end++;
//					j++;
//				}//next seg only have 1 element
//
//				vector<int> *ptsperseg = new vector<int>;
//				for (int m = 0; m < end - start + 1; m++, pseg_offset++)
//				{
//					p2seg[pseg_offset] = i;// (i << 18) + parts.size();
//					ptsperseg->push_back(streamlines[i].path.at(start + m).id);
//				}
//				line_division[i].push_back(parts.size());
//				segs2pts.push_back(*ptsperseg);
//				delete ptsperseg;
//				//make segment
//				l_vector a;
//				a.n_l = i;
//				a.center = getcenter(i, start, end);
//				a.start = a.center + (streamlines[i].path.at(start) - streamlines[i].path.at(end)) / 2;
//				a.end = a.center + (streamlines[i].path.at(end) - streamlines[i].path.at(start)) / 2;
//
//				a.offset = start;
//				a.n_p_offset = curr_pt + start;
//
//				a.length = end - start;
//				parts.push_back(a);
//
//				//next seg
//				start = end + 1;
//				end += 2;
//
//				j++;
//				if (end >= streamlines[i].path.size())
//					continue;
//				else {
//					curvelength = (streamlines[i].path[start] - streamlines[i].path[end]).length();
//					k = curvelength / length_vector(i, start, end);
//				}
//			}
//			else
//			{
//				end++;
//				curvelength += (streamlines[i].path[end - 1] - streamlines[i].path[end]).length();
//				k = curvelength / length_vector(i, start, end);
//				if (isnan(k) || isinf(k))
//					k = 0;
//			}
//
//		}
//
//		if (start < end - 1)//left over;
//		{
//			vector<int> *ptsperseg = new vector<int>;
//			for (int m = 0; m < end - start + 1; m++, pseg_offset++)
//			{
//				p2seg[pseg_offset] = i;
//				ptsperseg->push_back(streamlines[i].path.at(start + m).id);
//			}
//			line_division[i].push_back(parts.size());
//			segs2pts.push_back(*ptsperseg);
//			delete ptsperseg;
//			l_vector a;
//			a.n_l = i;
//			a.center = getcenter(i, start, end);
//			a.start = a.center + (streamlines[i].path.at(start) - streamlines[i].path.at(end)) / 2;
//			a.end = a.center + (streamlines[i].path.at(end) - streamlines[i].path.at(start)) / 2;
//			a.offset = start;
//			a.length = end - start;
//			a.n_p_offset = curr_pt + start;
//
//			parts.push_back(a);
//		}
//
//	}
//	return parts;
//}
