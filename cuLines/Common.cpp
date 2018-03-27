#include "Common.h"

#define _USE_MATH_DEFINES
#include <math.h>

#include <random>
using namespace std;


random_device rd{};
mt19937_64 engine{ rd() };

float gaussianDist(float x) {
	return exp(-x * x / 2.) / sqrtf(M_PI * 2);
}


