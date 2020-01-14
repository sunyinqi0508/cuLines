#pragma once
#ifndef STREAMLINE_H
#define STREAMLINE_H

#include <vector>
#include <deque>
#include "Vector.h"
typedef Vector3 vec3;

template <class TA, class TB, class TC> TA clamp(const TA & x, const TB & minv, const TC & maxv) { if (x < (TA)minv) return (TA)minv; else if (x > (TA)maxv) return (TA)maxv; return x; }

struct StreamlineParam
{
	double alpha;
	double dMin;
	double dSep;
	double dSelfsep;
	double w;
	double minLen;
	double maxLen;

	int nHalfSample;
	int maxSize;

	void print()const;
};

struct Line
{
	std::vector<vec3> path;
	std::vector<vec3> pathDir;
	double getLength()const;
	static double getLength(std::deque<Vector3>& line);
	static float getLength(const Vector3* this_line, const int ptr_rear, const int ptr_front);
};

extern vec3 m_minDims;
extern vec3 m_maxDims;
extern StreamlineParam g_param;

#endif
