#ifndef _FileIO_H
#define _FileIO_H
#include "Vector.h"
#include <stdint.h>
#include <vector>

#define AvailFlags(i) (1<<(i))
namespace FileIO {

	extern std::vector<std::vector<Vector3>> streamlines;
	extern Vector3** f_streamlines;
	extern char availiblity_flag;
	extern int64_t n_points;
	extern int64_t n_streamlines;

	enum Format { STREAMLINE_ARRAY, STREAMLINE_VECTOR };

	void LoadWaveFrontObject(const char* file);
	void normalize(float R = 1, const bool _normalize = true);
	void OutputBSL(const char* destination);
	void OutputOBJ(const char* destination, const float* vt, Format source);
	void ReadBSL(const char* filename);
	void toFStreamlines();
	/*
	*_normalize: true => normalize to [0, 1] then scale;
	*	 		 false => scale on original coords. (default)
	*			
	*/
	void scaleByR(float R, const bool _normalize = false);

	class Streamline {
	private:
		static size_t _max_size;
	public:
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
