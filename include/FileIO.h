#ifndef _FileIO_H
#define _FileIO_H
#include "Vector.h"
#include <stdint.h>
#include <vector>

#define AvailFlags(i) (1<<(i))
namespace FileIO {
	struct NormalizeParameter {
		Vector3 offset;
		float multiplier;
	};

	extern std::vector<std::vector<Vector3>> streamlines;
	extern Vector3** f_streamlines;
	extern float **curvatures;
	extern char availiblity_flag;
	extern int64_t n_points;
	extern int64_t n_streamlines;

	const enum  Format { STREAMLINE_ARRAY, STREAMLINE_VECTOR };
	inline float my_atof(char* &buf);
	void LoadWaveFrontObject(const char* file, int Max_N = INT_MAX);

	NormalizeParameter normalize(float R = 1, const bool _normalize = true, bool format = Format::STREAMLINE_VECTOR, Vector3 move = 0.f, bool centralize = false);
	void gaussianSmooth(int SmoothingWindow = 1.f, int _2D = false);
	void OutputBSL(const char* destination, Format source = Format::STREAMLINE_VECTOR);
	void OutputOBJ(const char* destination, const float* vt = 0, Format source = Format::STREAMLINE_VECTOR);
	void ReadBSL(const char* filename);
	void _getTangent(Vector3* tangent);
	void toFStreamlines();
	void reinitData();
	/*
	*_normalize: true => normalize to [0, 1] then scale;
	*	 		 false => scale on original coords. (default)
	*			
	*/
	void scaleByR(float R, const bool _normalize = false, bool format = Format::STREAMLINE_ARRAY);

	int getGlobalIndex(int line_idx, int pt_idx);
	int FieldfromAmiramesh(void *field, const char* FileName);
	class Streamline {
	private:
		static size_t _max_size;

	public:

		struct Streamline_data {
			int *pt_to_line;
			int *offsets;
			int *sizes;
			int n_streamlines;
			int n_points;
			Vector3 **f_streamlines;
		};


		static void initFromStreamlineData(Streamline_data* sd);
		static Streamline_data* storeIntoStreamlineData();
		static size_t inline size(size_t sl_pos);
		static size_t max_size();
		static void _calc_size(size_t sl_pos);
		static inline void reinit();
		static int getlineof(int p);
		static int* sizes;
		static int* offsets;
		static int* pt_to_line;
	};

}

#endif
