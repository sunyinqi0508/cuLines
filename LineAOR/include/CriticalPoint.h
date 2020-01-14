#pragma once
#ifndef CRITICAL_POINT_H
#define CRITICAL_POINT_H

#include "Vector.h"
#include <vector>

enum class CriticalPointType : int {
	RepelNode = 0x0,
	RepelFocus = 0x4,
	RepelNodeSaddle = 0x1,
	RepelFocusSaddle = 0x5,
	AttractNodeSaddle = 0x3,
	AttractFocusSaddle = 0x7,
	AttractNode = 0x2,
	AttractFocus = 0x6
};

const char *getCriticalPointTypeName(CriticalPointType cp_type);

struct CriticalPoint : Vector3 {
	CriticalPointType type = CriticalPointType::RepelNode;
	float scale = 1.f;
};

std::vector<CriticalPoint> loadCriticalPoints(const char *filename);

#endif
