#ifndef _SEEDING_3D
#define _SEEDING_3D
#include "../cuLines/LSH.h"
#include "Field2D.h"
#include <vector>

std::vector<LshFunc> create_seeding(int n_funcs, VectorField& field);

#endif
