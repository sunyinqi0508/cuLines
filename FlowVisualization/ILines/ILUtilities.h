/* $Id: ILUtilities.h,v 1.17 2005/10/17 10:12:09 ovidiom Exp $ */

/* forward declarations */
namespace ILines { class ILUtilities; }

#ifndef _ILUTILITIES_H_
#define _ILUTILITIES_H_

#include <cmath>
#include <climits>
#include <algorithm>
#include <time.h>
#include <vector>
#include <mutex>
#include <atomic>
#if defined(WIN32) || defined(__CYGWIN__)
#define WIN32_LEAN_AND_MEAN		1
#include <windows.h>
#endif

#include <GL/gl.h>
#include <GL/glu.h>

#include "Vector.h"
//extern class Viewer;


namespace ILines
{
	#define PI_FLOAT	3.141592653589f

	/**
	 * @brief Provides some general functions to be used by other classes.
	 */
	class ILUtilities
	{
	public:
		/** @brief Computes the rotational component of an OpenGL scene. */
		static float *computeRotationMatrix();

		/** @brief Performs z-sorting on a set of given line strips. */
		static std::pair<int *, int*>zsort(int *first, int *vertCount, int lineCount,
		                  const float *homVertices);
		static  bool repaint_pending;
		static int *dps_gc;
		static bool terminate;
		static std::vector<HANDLE>* threads;
		static std::mutex mtx;
		static std::atomic<unsigned int> *task_counter;
		/** @brief Core sorting algorithm for the z-sorting. */
		static void qsort(int * index, int * left, int * right, int * depths, std::atomic<unsigned int>* task_counter, int lv, int * indicator, LPCRITICAL_SECTION cs);

	private:
		/** @brief Helper class for z-sorting with the std::sort STL function. */
		struct DepthSorter;

		/** @brief Computes the depth for some point in an OpenGL scene. */
		static int getDepth(const GLfloat *v, GLfloat *mvp);
	};
}

#endif /* _ILUTILITIES_H_ */

