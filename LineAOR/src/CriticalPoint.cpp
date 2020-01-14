#include "CriticalPoint.h"

#include <fstream>

static const char *critical_point_type_names[] = {
	"RepelFocus", "RepelNodeSaddle",
	"AttractNode", "AttractNodeSaddle",
	"RepelFocus", "RepelFocusSaddle",
	"AttractFocus", "AttractFocusSaddle"
};

const char *getCriticalPointTypeName(CriticalPointType cp_type) {
	const auto index = static_cast<std::size_t>(cp_type);
	return 0 <= index && index < 8 ? critical_point_type_names[index] : "(unknown)";
}

template <typename T>
void readRaw(std::istream &in, T &x) {
	in.read(reinterpret_cast<char*>(&x), sizeof(T));
}

std::vector<CriticalPoint> loadCriticalPoints(const char *filename) {
	std::ifstream inputFile{ filename, std::ifstream::binary };
	uint32_t n_points;
	readRaw(inputFile, n_points);
	std::vector<CriticalPoint> pts{ n_points };
	for (uint32_t i = 0; i < n_points; i++)
		readRaw<Vector3>(inputFile, pts[i]);
	for (uint32_t i = 0; i < n_points; i++)
		readRaw(inputFile, pts[i].type);
	for (uint32_t i = 0; i < n_points; i++)
		readRaw(inputFile, pts[i].scale);
	return pts;
}
