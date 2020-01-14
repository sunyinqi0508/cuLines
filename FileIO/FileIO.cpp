#include "Common.h"
#include "Parameters.h"
#include "Vector.h"
#include "FileIO.h"
#include <basetsd.h>
#include <fstream>
#include <iostream>
using namespace std;
namespace FileIO {
	vector<vector<Vector3>>	streamlines;
	Vector3 ** f_streamlines = 0;
	float** curvatures = 0;
	void* clean_handle = 0, *clean_handle_d = 0;
	int64_t n_streamlines;
	int64_t n_points;
	char availiblity_flag = 0;

	inline void newline(/*const size_t& approx_path*/)
	{
		streamlines.push_back(vector<Vector3>());
		if (streamlines.size())
			streamlines.reserve(streamlines.back().size() + 2);
	}

	inline float my_atof(char* &buffer) {

		float result = 0;
		bool _sign = true;
		if (*buffer == '-') {
			_sign = false;
			++buffer;
		}
		while (*buffer >= '0' && *buffer <= '9')
		{
			result *= 10.f;
			result += *buffer - '0';
			++buffer;
		}
		if (*buffer == '.')
		{
			buffer++;
			float frac = 10;

			while (*buffer >= '0' && *buffer <= '9')
			{
				result += (float)(*(buffer++) - '0') / (frac);
				frac *= 10;
			}
		}

		if (*buffer == 'e')
		{
			buffer++;
			bool _e_sign = true;
			if (*buffer == '-')
			{
				_e_sign = false;
				++buffer;
			}
			int _e_res = 0;
			while (*buffer >= '0' && *buffer <= '9')
			{
				_e_res *= 10.f;
				_e_res += *buffer - '0';
				++buffer;
			}
			_e_res = _e_sign ? _e_res : -_e_res;
			result *= pow(10, _e_res);
		}
		return _sign?result:-result;

	}

	inline void readVertex(char *&buffer, int &pt) {

		float _x = my_atof(buffer);
		while (*buffer == ' ') ++buffer;
		float _y = my_atof(buffer);
		while (*buffer == ' ') ++buffer;
		float _z = my_atof(buffer);
		while (*buffer == ' ') ++buffer;
		if(pt%downsample_pt == 0)
			streamlines.back().push_back(Vector3(_x, _y, _z));//-x for out2.obj
	}
	inline void readVertex_NG(char *&buffer, const size_t curr) {

		float _x = my_atof(buffer);
		while (*buffer == ' ') ++buffer;
		float _y = my_atof(buffer);
		while (*buffer == ' ') ++buffer;
		float _z = my_atof(buffer);
		while (*buffer == ' ') ++buffer;

		f_streamlines[0][curr](_x, _y, _z);
	}

	NormalizeParameter normalize(float R, const bool _normalize, bool format, Vector3 move, bool centralize) {

		Vector3 max = -std::numeric_limits<float>::max()/4.f;
		Vector3 min = std::numeric_limits<float>::max()/4.f;
		if (format == Format::STREAMLINE_VECTOR) {
			for (vector<Vector3>& line : streamlines)
				for (Vector3& vertex : line)
					for (int i = 0; i < 3; i++)
						if (vertex[i] > max[i])
							max[i] = vertex[i];
						else if (vertex[i] < min[i])
							min[i] = vertex[i];
		} else
			for (int i = 0; i < n_streamlines; i++)
				for (int j = 0; j < Streamline::size(i); j++)
					for (int k = 0; k < 3; k++)
						if (f_streamlines[i][j][k] > max[k])
							max[k] = f_streamlines[i][j][k];
						else if (f_streamlines[i][j][k] < min[k])
							min[k] = f_streamlines[i][j][k];
		max -= min;

		float interval = max[0];
		for (int i = 1; i < 3; i++)
			if (interval < max[i])
				interval = max[i];

		interval = (_normalize ? interval : 1) * R;

		if (centralize)
			min += (max - min) / 2.f;
		else {
			min = _normalize ? min : Vector3(0.f);
			min -= move;
		}
		if (format == Format::STREAMLINE_VECTOR)
			for (vector<Vector3>&line : streamlines)
				for (Vector3& vertex : line)
					vertex = (vertex - min) / interval;
		else
			for (int i = 0; i < n_streamlines; i++)
				for (int j = 0; j < Streamline::size(i); j++)
					f_streamlines[i][j] = (f_streamlines[i][j] - min) / interval;

		return NormalizeParameter{ min, interval };
	}
	void scaleByR(float R, const bool _normalize, bool format)
	{
		normalize(R, _normalize, format);
	}


	void LoadWaveFrontObject_NG(const char* file) {
		FILE* fp;
		fopen_s(&fp, file, "r");
		fseek(fp, SEEK_SET, SEEK_END);
		size_t file_size = ftell(fp);
		fseek(fp, SEEK_SET, SEEK_SET);
		char* buffer = new char[file_size];
		char* buffer_orig = buffer;
		fread_s(buffer, file_size, sizeof(char), file_size, fp);
		int ptr = file_size - 1;
		while (ptr > 0 && (buffer[ptr]<'0' || buffer[ptr] > '9'))ptr--;
		int n_p = 0, n_lines = 0, order = 1;
		while (ptr > 0 && buffer[ptr] <= '9'&&buffer[ptr] >= '0') {
			n_p += (buffer[ptr] - '0') * order;
			order *= 10;
			ptr--;
		}
		while (ptr > 0 && (buffer[ptr] != '\n'))ptr--;
		while (ptr > 0 && (buffer[ptr]<'0' || buffer[ptr] > '9'))ptr--;
		order = 1;
		while (ptr > 0 && buffer[ptr] <= '9'&&buffer[ptr] >= '0') {
			n_lines += (buffer[ptr] - '0') * order;
			order *= 10;
			ptr--;
		}
		while (ptr > 0 && (buffer[ptr] != '\n'))ptr--;
		buffer[ptr] = '\0';

		clean_handle = f_streamlines = new Vector3*[n_lines + 1];
		clean_handle_d = f_streamlines[0] = new Vector3[n_p];

		size_t ptr_l = 0, ptr_p = 0;
		while (*buffer) {
			bool skip = false;
			while (*buffer&&*buffer != '\n') {

				if (skip)
					buffer++;
				else if (*buffer == 'g') {
					f_streamlines[ptr_l + 1] = f_streamlines[0] + ptr_p;
					skip = true;
				}
				else if (*buffer++ == 'v' && *buffer++ == ' ') {
					//ptr_p ++;
					readVertex_NG(buffer, ptr_p);
					ptr_p++;
					skip = true;
				}
				else skip = true;

			}
			if (!*buffer)
				break;
			buffer++;
		}

		f_streamlines[n_lines] = f_streamlines[0] + n_p;

		Streamline::reinit();

		fclose(fp);
	}
	void LoadWaveFrontObject(const char* file, int Max_N) {
		reinitData();

		streamlines.clear();
		streamlines.resize(0);
		FILE* fp = 0;
		fopen_s(&fp, file, "rb");
		if (fp) {
			fseek(fp, SEEK_SET, SEEK_END);
			size_t file_size = ftell(fp);
			fseek(fp, SEEK_SET, SEEK_SET);
			char* buffer = new char[file_size];
			char* buffer_orig = buffer;
			fread_s(buffer, file_size, sizeof(char), file_size, fp);
			//const size_t approx_line = (size_t)sqrt(file_size / 2);
			//const size_t approx_path = (file_size / 25) / approx_line;
			//streamlines.reserve(approx_line);
			newline();
			int pt = 0;
			while (*buffer) {
				bool skip = false;
				while (*buffer&&*buffer != '\n') {
					if (skip)
						buffer++;
					else if (*buffer == 'g') {
						pt = 0;
						newline();
						streamlines.back().shrink_to_fit();
						if (streamlines.size() > Max_N)
							goto end;
						skip = true;
					}
					else if (*buffer++ == 'v' && *buffer++ == ' ') {
						//streamlines.back().push_back(readVertex(buffer));
						readVertex(buffer, pt);
						++pt;
						skip = true;
					}
					else skip = true;

				}
				if (!*buffer)
					break;
				buffer++;
			}
		end:
			streamlines.shrink_to_fit();
			streamlines.pop_back();
			availiblity_flag |= 1 << Format::STREAMLINE_VECTOR;
			//delete[]buffer_orig;
			fclose(fp);
		}
	}
	void gaussianSmooth(int SmoothingWindow, int _2D) {
		Vector3** f_streamline2 = new Vector3*[n_streamlines];
		f_streamline2[0] = new Vector3[n_points];
		for (int i = 1; i < n_streamlines; i++) {
			f_streamline2[i] = f_streamline2[i - 1] + Streamline::size(i - 1);
		}

		for (int j = 0; j < 160; j++) {
			std::memcpy(f_streamline2[0], f_streamlines[0], sizeof(Vector3) * n_points);
#pragma omp parallel
#pragma omp  for
			for (int i = 0; i < n_points; i++) {
				const int this_line = Streamline::getlineof(i);
				const int pt_on_line = i - Streamline::offsets[this_line];

				int start = __macro_max(0, pt_on_line - SmoothingWindow);
				int end = __macro_min(Streamline::size(this_line), pt_on_line + SmoothingWindow);
				float sum_gx = 1;
				for (int k = start; k < end; k++) {
					const float distik = f_streamlines[0][i] - f_streamlines[this_line][k];
					if (k != pt_on_line && !isnan(distik) && !isinf(distik)) {
						const float gx = (float)(exp(-9. * 10./ (2.f*16.f)) / (sqrt(2 * M_PI*16.f)));
						sum_gx += gx;
						f_streamline2[0][i] += f_streamlines[this_line][k] * gx;
					}
				}
				f_streamline2[0][i] /= sum_gx;
			}
			if (_2D) {
				Vector3 center1 = 0;
				for (int j = 0; j < 6; j++)
					center1 += f_streamline2[j][Streamline::size(j) - 1];
				center1 /= 6.f;
				for (int j = 0; j < 6; j++)
					f_streamline2[j][Streamline::size(j) - 1] = center1;
				f_streamline2[11][0] = f_streamline2[11][Streamline::size(11) - 1] = (f_streamline2[11][Streamline::size(11) - 1] + f_streamline2[11][0]) / 2.f;
				f_streamline2[12][0] = f_streamline2[12][Streamline::size(12) - 1] = (f_streamline2[12][Streamline::size(12) - 1] + f_streamline2[12][0]) / 2.f;
				f_streamline2[13][0] = f_streamline2[13][Streamline::size(13) - 1] = (f_streamline2[13][Streamline::size(13) - 1] + f_streamline2[13][0]) / 2.f;
				f_streamline2[16][0] = f_streamline2[16][Streamline::size(16) - 1] = (f_streamline2[16][Streamline::size(16) - 1] + f_streamline2[16][0]) / 2.f;
			}
			std::memcpy(f_streamlines[0], f_streamline2[0], sizeof(Vector3) * n_points);
		}
	}
	void toFStreamlines() {

		if (availiblity_flag & AvailFlags(Format::STREAMLINE_VECTOR))
		{

			n_streamlines = streamlines.size();
			n_points = 0;

			clean_handle = f_streamlines = new Vector3*[n_streamlines + 1];
			for (vector<Vector3> line : streamlines)
				n_points += line.size();

			clean_handle_d = f_streamlines[0] = new Vector3[n_points];
			int ptr_fs = 1;
			for (vector<Vector3> line : streamlines) {

				f_streamlines[ptr_fs] = f_streamlines[ptr_fs - 1] + line.size();

				int j = 0;
				for (Vector3 vertex : line) {
					f_streamlines[ptr_fs - 1][j++] = vertex;
				}
				ptr_fs++;
			}
			Streamline::reinit();
		}
	}

	void OutputBSL(const char* destination, Format source) {

		FILE *fp = 0;
		fopen_s(&fp, destination, "wb");
		if (source == Format::STREAMLINE_VECTOR) {
			int64_t _size = streamlines.size();
			fwrite(&_size, sizeof(int64_t), 1, fp);
			_size = 0;
			fwrite(&_size, sizeof(int64_t), 1, fp);
			for (vector<Vector3>& line : streamlines) {
				_size = line.size();
				fwrite(&_size, sizeof(int64_t), 1, fp);
			}

			for (vector<Vector3>& line : streamlines)
				for (Vector3& vertex : line)
					fwrite(vertex, sizeof(vertex), 1, fp);
		}
		else {
			fwrite(&n_streamlines, sizeof(int64_t), 1, fp);
			int64_t _size = 0;
			fwrite(&_size, sizeof(int64_t), 1, fp);
			for (int i = 0; i < n_streamlines; i++) {
				_size = Streamline::sizes[i];
				fwrite(&_size, sizeof(int64_t), 1, fp);
			}
			fwrite(f_streamlines[0], sizeof(Vector3), n_points, fp);
		}
		fclose(fp);

	}
	void OutputOBJ(const char* destination, const float* vt, Format source) {

		FILE *fp;
		fopen_s(&fp, destination, "w");
		size_t pt_num = 1, pt_cnt = 0;
		if (source == STREAMLINE_ARRAY) {
			for (int i = 0; i < n_streamlines; i++)
			{
				for (int j = 0; j < Streamline::size(i); j++) {
					fprintf(fp, "v ");
					for (int k = 0; k < 3; k++)
						fprintf(fp, "%f ", f_streamlines[i][j][k]);
					if (vt == 0)
						fprintf(fp, "\nvt 0\n");
					else
						fprintf(fp, "\nvt %g\n", vt[pt_cnt++]);

				}
				fprintf(fp, "g line%d\nl ", i + 1);
				for (int j = 0; j < Streamline::size(i); j++) {
					fprintf(fp, "%d ", j + pt_num);
				}
				fprintf(fp, "\n");
				pt_num += Streamline::size(i);
			}
		}
		else if (source == STREAMLINE_VECTOR) {
			int i = 0;
			for (vector<Vector3> line : streamlines) {
				for (Vector3 vertex : line) {
					fprintf(fp, "v ");
					for (int k = 0; k < 3; k++)
						fprintf(fp, "%f ", vertex[k]);
					vt ? fprintf(fp, "\nvt %g\n", vt[pt_cnt++]) : fprintf(fp, "\nvt 0\n");
				}
				fprintf(fp, "g line%d\nl ", ++i);
				for (int j = 0; j < line.size(); j++) {
					fprintf(fp, "%d ", j + pt_num);
				}
				fprintf(fp, "\n");
				pt_num += line.size();
			}
		}

		fclose(fp);

	}

	void ReadBSL64(const char* filename) {

		reinitData();

		FILE *fp;
		fopen_s(&fp, filename, "rb");
		if (fp)
		{
			void* buffer;
			fseek(fp, SEEK_SET, SEEK_END);
			size_t file_size = ftell(fp);
			fseek(fp, SEEK_SET, SEEK_SET);
			clean_handle = buffer = malloc(file_size);

			fread(buffer, 1, file_size, fp);

			n_streamlines = *((int64_t*)buffer);
			f_streamlines = (Vector3 **)buffer + 1;


			f_streamlines[0] = (Vector3*)(f_streamlines + n_streamlines + 1);
			for (int i = 1; i <= n_streamlines; i++)
				f_streamlines[i] = f_streamlines[i - 1] + *((int64_t*)f_streamlines + i);
			n_points = (file_size - sizeof(int64_t)*(n_streamlines + 2)) / sizeof(Vector3);


			fclose(fp);
		}

		Streamline::reinit();
	}

	void ReadBSL32() {
		throw "Not implimented.";
	}

	void ReadBSL(const char* filename) {
		sizeof(void*) == 8 ? ReadBSL64(filename) : ReadBSL32();
	}


	void _getTangent(Vector3 *tangents) {

		int idx_pt = 0;
		for (size_t i = 0; i <n_streamlines; i++)
		{
			for (size_t j = 0; j < Streamline::size(i); j++)
			{
				int idx1 = j + 1, idx2 = j - 1;
				idx2 = idx2 < 0 ? 0 : idx2;
				idx1 = idx1 >= Streamline::size(i) ? Streamline::size(i) - 1 : idx1;
				tangents[idx_pt + j] = (f_streamlines[i][idx1] - f_streamlines[i][idx2]).normalized();
			}

			idx_pt += Streamline::size(i);
		}
	}


	void reinitData() {

		streamlines.clear();
		if (clean_handle)
		{
			delete[] (clean_handle);
			clean_handle = f_streamlines= new Vector3*[1];
		}
		if(clean_handle_d )
		{
			delete[] clean_handle_d;
			clean_handle_d = 0;
		}
		n_points = 0;
		n_streamlines = 0;

	}

	int getGlobalIndex(int line_idx, int pt_idx) {
		return &f_streamlines[line_idx][pt_idx] - &f_streamlines[0][0];
	}

	int* Streamline::sizes = NULL;
	int* Streamline::pt_to_line = NULL;
	int* Streamline::offsets = NULL;

	size_t Streamline::_max_size = 0;
	void Streamline::initFromStreamlineData(Streamline_data * sd)
	{
		pt_to_line = sd->pt_to_line;
		offsets = sd->offsets;
		sizes = sd->sizes;
		n_streamlines = sd->n_streamlines;
		n_points = sd->n_points;
		f_streamlines = sd->f_streamlines;
		delete sd;
	}
	Streamline::Streamline_data * Streamline::storeIntoStreamlineData()
	{
		Streamline_data *ret = new Streamline_data;
		ret->pt_to_line = pt_to_line;
		ret->offsets = offsets;
		ret->sizes = sizes;
		ret->n_streamlines = n_streamlines;
		ret->n_points = n_points;
		ret->f_streamlines = f_streamlines;
		return ret;
	}
	size_t inline Streamline::size(size_t sl_pos)
	{
		//if (sizes[sl_pos] < 0)
		//_calc_size(sl_pos); 
		/*Earerly compute sizes instead of lazy strategy*/
		return sizes[sl_pos];
	}

	//Deprecated
	void Streamline::_calc_size(size_t sl_pos) {
		sizes[sl_pos] = f_streamlines[sl_pos + 1] - f_streamlines[sl_pos];
	}

	size_t Streamline::max_size() {

		if (_max_size == 0)
			for (size_t i = 0; i < n_streamlines; i++)
				if (size(i) > _max_size)
					_max_size = size(i);
		return _max_size;

	}

	int Streamline::getlineof(int p) {
		return pt_to_line[p];
	}
	inline void Streamline::reinit() {
		if (sizes)
		{
			delete[] sizes;
			sizes = 0;
		}
		if (pt_to_line)
		{
			delete[] pt_to_line;
			pt_to_line = 0;
		}
		if (offsets)
		{
			delete[] offsets;
			offsets = 0;
		}
		_max_size = 0;

		sizes = new int[n_streamlines + 1];
		offsets = new int[n_streamlines + 1];
		pt_to_line = new int[n_points + 1];

		offsets[0] = 0;
		//std::fill(sizes, sizes + n_streamlines - 1, -1);
		for (size_t i = 0; i < n_streamlines; i++) {
			sizes[i] = f_streamlines[i + 1] - f_streamlines[i];
			std::fill(pt_to_line + offsets[i], pt_to_line + offsets[i] + sizes[i], i);
			offsets[i + 1] = offsets[i] + sizes[i];
		}
		//std::fill(pt_to_line + last_sz, pt_to_line + n_points, n_streamlines);

		//sizes[n_streamlines - 1] = n_points - (f_streamlines[n_streamlines - 1] - f_streamlines[0]);
		sizes[n_streamlines] = n_points;

		availiblity_flag |= AvailFlags(Format::STREAMLINE_ARRAY);
	}
}
//bool Streamline::_init = false;
//size_t Streamline::last_len = -1;
//bool Streamline::init() {
//	if (_init)
//		return true;
//	else {
//		reinit();
//		_init = true;
//	}

//}
