#include "Common.h"
<<<<<<< HEAD
#define _USE_MATH_DEFINES
#include <math.h>
=======
>>>>>>> b474210fd47d6c9a4d8374701a0bf1b740dfb3d8
#include <random>
using namespace std;


random_device rd{};
mt19937_64 engine{ rd() };

<<<<<<< HEAD
float gaussianDist(float x) {
	return exp(-x * x / 2.) / sqrtf(M_PI * 2);
}
=======
>>>>>>> b474210fd47d6c9a4d8374701a0bf1b740dfb3d8
