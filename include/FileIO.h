#ifndef _FILEIO_H
#define _FILEIO_H
#include "Vector.h"
#include <basetsd.h>
#include <vector>

#define AvailFlags(i) (1<<(i))
namespace FILEIO {

	extern std::vector<std::vector<Vector3>> streamlines;
	extern Vector3** f_streamlines;
	extern char availiblity_flag;
	extern INT64 n_points;
	extern INT64 n_streamlines;

	enum Format { STREAMLINE_ARRAY, STREAMLINE_VECTOR };

	void LoadWaveFrontObject(const char* file);
	void normalize();
	void OutputBSL(const char* destination);
	void OutputOBJ(const char* destination, const float* vt, Format source);
	void ReadBSL(const char* filename);
	void toFStreamlines();

	class Streamline {
	private:
		static int* sizes;
		static size_t _max_size;
	public:
		static size_t size(size_t sl_pos);
		static size_t max_size();
		static void _calc_size(size_t sl_pos);
		static inline void reinit();

	};

}

#endif