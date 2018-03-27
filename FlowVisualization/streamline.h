#pragma once
#ifndef STREAMLINE_H
#define STREAMLINE_H
#define QT_NO_DEBUG_OUTPUT

#include "Vector2.h"
#include "Vector3.h"
#include "Parameters.h"
#include <vector>
#include <deque>

typedef Vector3 vec3;
typedef Vector2 vec2;

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
	int N;

};



extern vec2 m_minDims;
extern vec2 m_maxDims;


extern StreamlineParam g_param;

#endif