/* $Id: ILUtilities.cpp,v 1.20 2005/10/17 10:12:09 ovidiom Exp $ */

#include "ILUtilities.h"
#include <tuple>

namespace ILines
{
	bool ILUtilities::repaint_pending = true;
	bool ILUtilities::terminate = true;
	std::mutex ILUtilities::mtx;
	std::vector<HANDLE>* ILUtilities::threads;
	std::atomic<unsigned int> *ILUtilities::task_counter;
	int * ILUtilities::dps_gc = 0;
	struct ILUtilities::DepthSorter
	{
		int	*depths;

		DepthSorter(int *depths)
		{
			this->depths = depths;
		}

		bool operator()(const int &i, const int &j) const
		{
			return depths[i] > depths[j];
		}
	};


	/**
	 * @return  A 4x4 matrix describing the rotational component of
	 *          an OpenGL scene.
	 */
	float *ILUtilities::computeRotationMatrix()
	{
		static const Vector3f	X(1.0f, 0.0f, 0.0f), Y(0.0f, 1.0f, 0.0f);
		Vector3f				crossX, crossY;
		float					angleX, angleY;
		float					tmpMatrix[16];
		float					*rotMatrix;
		GLdouble				model[16], proj[16];
		GLint					view[4];
		double					objX, objY, objZ;
		Vector3f				p1, p2;
		Vector3f				rotX, rotY;

		glGetDoublev(GL_MODELVIEW_MATRIX, model);
		glGetDoublev(GL_PROJECTION_MATRIX, proj);
		glGetIntegerv(GL_VIEWPORT, view);

		gluUnProject(0.0, 0.0, 1.0,
		             model, proj, view,
		             &objX, &objY, &objZ);
		p1 = Vector3f((float)objX, (float)objY, (float)objZ);
		gluUnProject(100.0, 0.0, 1.0,
		             model, proj, view,
		             &objX, &objY, &objZ);
		p2 = Vector3f((float)objX, (float)objY, (float)objZ);
		rotX = normalize(p2 - p1);

		gluUnProject(0.0, 100.0, 1.0,
		             model, proj, view,
		             &objX, &objY, &objZ);
		p2 = Vector3f((float)objX, (float)objY, (float)objZ);
		rotY = normalize(p2 - p1);

		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();

		angleX = (float)acos(dot(X, rotX)) * 180.0f / PI_FLOAT;
		/* For angles near 0 or 180 we would not get a clean cross product. */
		if ((angleX < 1.0f) || (angleX > 179.0f))
			crossX = Vector3f(0.0f, 1.0f, 0.0f);
		else
			crossX = cross(rotX, X);

		glRotatef(angleX, crossX.x, crossX.y, crossX.z);

		tmpMatrix[0] = rotY.x;
		tmpMatrix[1] = rotY.y;
		tmpMatrix[2] = rotY.z;
		tmpMatrix[3] = 1.0f;

		glMultMatrixf((GLfloat *)tmpMatrix);
		glGetFloatv(GL_MODELVIEW_MATRIX, (GLfloat *)tmpMatrix);

		rotY.x = tmpMatrix[0];
		rotY.y = tmpMatrix[1];
		rotY.z = tmpMatrix[2];

		angleY = (float)acos(dot(Y, rotY)) * 180.0f / PI_FLOAT;
		/* For angles near 0 or 180 we would not get a clean cross product. */
		if ((angleY < 1.0f) || (angleY > 179.0f))
			crossY = Vector3f(1.0f, 0.0f, 0.0f);
		else
			crossY = cross(rotY, Y);

		glLoadIdentity();
		glRotatef(angleY, crossY.x, crossY.y, crossY.z);
		glRotatef(angleX, crossX.x, crossX.y, crossX.z);

		rotMatrix = new float[16];
		glGetFloatv(GL_MODELVIEW_MATRIX, (GLfloat *)rotMatrix);

		glLoadMatrixd(model);

		return (rotMatrix);
	}

	/**
	 * @param first        Array of indices into the OpenGL vertex array
	 *                     where the vertex set for each line strip starts.
	 * @param vertCount    Array of number of vertices forming each line strip.
	 * @param lineCount    Number of line strips to draw.
	 * @param homVertices  The \e homogeneous vertex coordinates.
	 * @return             Array of indices into the OpenGL vertex array
	 *                     for drawing the individual line segments of the
	 *                     line strips from back to front.
	 */
	unsigned long long draw_count = 0;
	bool MT_End = false;

	std::pair<int *, int*> ILUtilities::zsort(int *first, int *vertCount, int lineCount,
	                        const float *homVertices)
	{
		int			i, j;
		int			arrSize;
		int			segCount;
		int			idx;
		int			*indices, *indicesIter;
		int			*depths;
		GLfloat		model[16], proj[16], mvp[16];
		GLenum		matrixMode;

		/* Compute the modelview-projection matrix and
		 * restore the rendering state. */
		glGetIntegerv(GL_MATRIX_MODE, (GLint *)&matrixMode);
		glGetFloatv(GL_MODELVIEW_MATRIX, model);
		glGetFloatv(GL_PROJECTION_MATRIX, proj);
		glMatrixMode(GL_PROJECTION);
		glMultMatrixf(model);
		glGetFloatv(GL_PROJECTION_MATRIX, mvp);
		glLoadMatrixf(proj);
		glMatrixMode(matrixMode);

		arrSize = 0;
		for (i = 0; i < lineCount; i++)
			if (first[i] + vertCount[i] > arrSize)
				arrSize = first[i] + vertCount[i];

		/* Compute the number of line segments. */
		segCount = 0;
		for (i = 0; i < lineCount; i++)
			segCount += vertCount[i] - 1;

		indices = new int[segCount];
		depths = new int[arrSize];

		indicesIter = indices;
		for (i = 0; i < lineCount; i++)
		{
			idx = first[i];
			depths[idx] = getDepth(&homVertices[4 * idx], mvp);
			for (j = 1; j < vertCount[i]; j++)
			{
				*indicesIter++ = idx++;
				depths[idx] = getDepth(&homVertices[4 * idx], mvp);
				depths[idx - 1] += depths[idx];
			}
		}
		task_counter = new std::atomic<unsigned int>;

#if 0
		int *indicator = new int(false);
		if (segCount)
		{
			/*if (repaint_pending >= 2||repaint_pending<0)
				printf("\n\n\n\n\n\n>2!! %ld", repaint_pending);*/
			*indicator = true;
			ILUtilities::terminate = false;
			//ILUtilities::threads = new std::vector<HANDLE>;
			*task_counter = UINT_MAX;
			MT_End = false;
			CRITICAL_SECTION cs;
			InitializeCriticalSection(&cs);
			qsort(indices, indices, indices + (segCount - 1), depths, task_counter,clock(), indicator, &cs);

			if (ILUtilities::repaint_pending)
				 ILUtilities::terminate = true;

			repaint_pending = true;
		}
#else
		std::sort(indices, indices + segCount, DepthSorter(depths));
#endif
		dps_gc = depths;
		return std::make_pair(indices, nullptr);
	}

	/**
	 * @param left    The leftmost element to sort.
	 * @param right   The rightmost element to sort.
	 * @param depths  Array of depths used as sorting keys.
	 */
	struct Sort_Param {
		int *left, *right, *depths, *index;
		int lv;
		int *id;
		LPCRITICAL_SECTION cs;
		std::atomic<unsigned int> *task_counter;
		
		Sort_Param(int *index, int *left, int *right, int *depths, int lv, int *id, std::atomic<unsigned int>* task_counter, LPCRITICAL_SECTION cs)
			:index(index), left(left), right(right),depths(depths), lv(lv), id(id), task_counter(task_counter), cs(cs) {}
	};
	int threadID = 0;
	DWORD WINAPI qsort_launch(LPVOID param) {
		Sort_Param* p = (Sort_Param *)(param);

		ILUtilities::qsort(p->index, p->left, p->right, p->depths, p->task_counter,p->lv, p->id, p->cs);
		EnterCriticalSection(p->cs);
		if ((--*(p->task_counter)) == 0) {
			//gc
			//delete p->task_counter;
			LeaveCriticalSection(p->cs);

			*(p->id) = 0;
			delete[] p->depths;
		}
		else
			LeaveCriticalSection(p->cs);
		
		delete p;
		return 0;
	}

	void ILUtilities::qsort(int *index, int *left, int *right, int *depths, std::atomic<unsigned int> *task_counter, int lv, int *indicator, LPCRITICAL_SECTION cs)
	{
		int		pivot;
		int		*l, *r;
		int		tmp;

		pivot = depths[left[(right - left) / 2]];
		l = left;
		r = right;

		/* Sort in descending order with respect to the depths. */
		while (true)
		{
			while (depths[*l] > pivot)
				l++;
			while (depths[*r] < pivot)
				r--;

			if (l >= r)
				break;

			tmp = *l;
			*l = *r;
			*r = tmp;

			l++;
			r--;
		}

		if (l == r)
		{
			l++;
			r--;
		}
		if ((clock()-lv < 1 || !repaint_pending))//||repaint_pending<=1)
		{
			if (left < r)
			{
				if (l < right)
				{
					if (right - l > r - left) {
						EnterCriticalSection(cs);
						if (right - l > 500 && (*task_counter) > 0) {
							if ((*task_counter) == UINT_MAX)
								*task_counter = 0;
							(*task_counter)++;
							qsort(index, left, r, depths, task_counter, lv, indicator, cs);

							//CreateThread(0, 0, qsort_launch, new Sort_Param(index, l, right, depths, lv, indicator, task_counter, cs), 0, 0);
							LeaveCriticalSection(cs);
						}
						else {
							if ((*task_counter) == UINT_MAX)
							{
								*task_counter = 0;
								*indicator = 0;
							}
							LeaveCriticalSection(cs);
							qsort(index, l, right, depths, task_counter, lv, indicator, cs);
							
						}
						/*if (r - left > 50 && (*task_counter) > 0)
						{
							(*task_counter)++;
							CreateThread(0, 0, qsort_launch, new Sort_Param(index, left, r, depths, lv, indicator, task_counter), 0, 0);
						}
						else*/
							qsort(index, left, r, depths,task_counter, lv, indicator,cs);
					}
					else {
						/*if (r - left > 50 && (*task_counter) > 0) {
							(*task_counter)++;
							CreateThread(0, 0, qsort_launch, new Sort_Param(index, left, r, depths, lv, indicator, task_counter), 0, 0);
						}
						else*/
							qsort(index, left, r, depths, task_counter,lv, indicator,cs);

						/*if (right - l> 50 && (*task_counter) > 0) {
							(*task_counter)++;
							CreateThread(0, 0, qsort_launch, new Sort_Param(index, l, right, depths, lv, indicator, task_counter), 0, 0);
						}
						else*/
							qsort(index, l, right, depths, task_counter, lv, indicator,cs);
					}
				}
				else
					qsort(index, left, r, depths, task_counter, lv, indicator,cs);

			}
			else if (l < right)
				qsort(index, l, right, depths, task_counter, lv, indicator,cs);
		}
	}

	/**
	 * @param v      The object coordinates.
	 * @param mvp    The modelview-projection-matrix.
	 * @return       The depth belonging to the given coordinates.
	 */
	int ILUtilities::getDepth(const GLfloat *v, GLfloat *mvp)
	{
		GLfloat	depth;

		depth  = mvp[2] * v[0] + mvp[6] * v[1] + mvp[10] * v[2] + mvp[14];
		depth /= mvp[3] * v[0] + mvp[7] * v[1] + mvp[11] * v[2] + mvp[15];

		/* depth now lies in the range [-1, +1].
		 * Map it to the integer range [-INT_MAX / 4, +INT_MAX / 4]. */

		return ((int)(depth * INT_MAX) / 4);
	}
}

