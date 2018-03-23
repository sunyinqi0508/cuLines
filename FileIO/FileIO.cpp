#include "Vector.h"
#include "FileIO.h"
#include <basetsd.h>
#include <fstream>
#include <iostream>
using namespace std;
namespace FileIO {
	vector<vector<Vector3>>	streamlines;
	Vector3 ** f_streamlines;
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
		return result;

	}

	inline void readVertex(char *&buffer) {

		float _x = my_atof(buffer);
		while (*buffer == ' ') ++buffer;
		float _y = my_atof(buffer);
		while (*buffer == ' ') ++buffer;
		float _z = my_atof(buffer);
		while (*buffer == ' ') ++buffer;
		streamlines.back().push_back(Vector3(_x, _y, _z));
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

	//template <bool T>
	void normalize(float R, const bool _normalize) {

		float max = INT_MIN, min = INT_MAX;
		for (vector<Vector3>& line : streamlines)
			for (Vector3& vertex : line)
				for (int i = 0; i < 3; i++)
					if (vertex[i] > max)
						max = vertex[i];
					else if (vertex[i] < min)
						min = vertex[i];

		const float interval = (_normalize ? (max - min) : 1) * R;

		for (vector<Vector3>&line : streamlines)
			for (Vector3& vertex : line)
				for (int i = 0; i < 3; i++)
					vertex[i] = (vertex[i] - min) / interval;

	}

	void scaleByR(float R, const bool _normalize)
	{
		normalize(R, _normalize);
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

		f_streamlines = new Vector3*[n_lines];
		f_streamlines[0] = new Vector3[n_p];

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
		Streamline::reinit();

		fclose(fp);
	}
	void LoadWaveFrontObject(const char* file) {

		streamlines.clear();
		streamlines.resize(0);
		FILE* fp;
		fopen_s(&fp, file, "rb");
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

		while (*buffer) {
			bool skip = false;
			while (*buffer&&*buffer != '\n') {

				if (skip)
					buffer++;
				else if (*buffer == 'g') {
					newline();
					streamlines.back().shrink_to_fit();

					skip = true;
				}
				else if (*buffer++ == 'v' && *buffer++ == ' ') {
					//streamlines.back().push_back(readVertex(buffer));
					readVertex(buffer);
					skip = true;
				}
				else skip = true;

			}
			if (!*buffer)
				break;
			buffer++;
		}
		streamlines.shrink_to_fit();
		streamlines.pop_back();
		availiblity_flag |= 1 << Format::STREAMLINE_VECTOR;
		//delete[]buffer_orig;
		fclose(fp);

	}

	void toFStreamlines() {

		if (availiblity_flag & AvailFlags(Format::STREAMLINE_VECTOR))
		{

			n_streamlines = streamlines.size();
			n_points = 0;
			f_streamlines = new Vector3*[n_streamlines];
			for (vector<Vector3> line : streamlines)
				n_points += line.size();

			f_streamlines[0] = new Vector3[n_points];
			int ptr_fs = 1;
			for (vector<Vector3> line : streamlines) {
				if (ptr_fs < n_streamlines)
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

	void OutputBSL(const char* destination) {

		FILE *fp = 0;
		fopen_s(&fp, destination, "wb");

		int64_t _size = streamlines.size();
		fwrite(&_size, sizeof(int64_t), 1, fp);
		_size = 0;
		fwrite(&_size, sizeof(int64_t), 1, fp);
		for (vector<Vector3>& line : streamlines) {
			_size = line.size();
			fwrite(&_size, sizeof(int64_t), 1, fp);
		}
		fseek(fp, -(long)sizeof(int64_t), SEEK_END);
		for (vector<Vector3>& line : streamlines)
			for (Vector3& vertex : line)
				fwrite(vertex, sizeof(vertex), 1, fp);

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

		FILE *fp;
		fopen_s(&fp, filename, "rb");

		void* buffer;
		fseek(fp, SEEK_SET, SEEK_END);
		size_t file_size = ftell(fp);
		fseek(fp, SEEK_SET, SEEK_SET);
		buffer = malloc(file_size);

		fread(buffer, 1, file_size, fp);

		n_streamlines = *((int64_t*)buffer);
		f_streamlines = (Vector3 **)buffer + 1;
		f_streamlines[0] = (Vector3*)(f_streamlines + n_streamlines);
		for (int i = 1; i < n_streamlines; i++)
			f_streamlines[i] = f_streamlines[i - 1] + *((int64_t*)f_streamlines + i);
		n_points = (file_size - sizeof(int64_t)*(n_streamlines + 1)) / sizeof(Vector3);
		Streamline::reinit();
		fclose(fp);

	}

	void ReadBSL32() {
		throw "Not implimented.";
	}

	void ReadBSL(const char* filename) {
		sizeof(void*) == 8 ? ReadBSL64(filename) : ReadBSL32();
	}

	int* Streamline::sizes = NULL;
	size_t Streamline::_max_size = 0;
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

	inline void Streamline::reinit() {
		if (sizes)
			delete[] sizes;
		sizes = new int[n_streamlines + 1];
		std::fill(sizes, sizes + n_streamlines - 1, -1);
		for (size_t i = 0; i < n_streamlines - 1; i++) {
			sizes[i] = f_streamlines[i + 1] - f_streamlines[i];
		}
		sizes[n_streamlines - 1] = n_points - (f_streamlines[n_streamlines - 1] - f_streamlines[0]);
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