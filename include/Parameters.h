#ifndef _H_PARAMETERS
#define _H_PARAMETERS

#if  defined(__linux__) || defined(_WIN32) || defined(__unix__) || defined(_POSIX_VERSION) ||defined (__APPLE__)
// determine if it's in host code orshader
constexpr int funcpool_size = 32, L = 1, K = 1, TABLESIZE = 10000;
constexpr float R1 = 1.f;
#include <stdint.h>
constexpr int64_t Prime = (1ll << 32) - 5;
constexpr int maxNN = 64, similarity_window= 8;
constexpr bool readVT = true;
constexpr bool allow_segs_on_same_line = false;
constexpr int downsample_pt = 1;
constexpr int omp_threads = 8;
#endif

//#if  defined(USE_PARAM1)


//UI: data from dll or file
#define DLL

//Similarity Measure: Central surrounded
#define _CENTRAL_SURROUNDED

//Shader: Direct alpha output?
#define ALPHA_DIRECT

//Applications
#define _APPLICATION_ALPHA
#define _APPLICATION_SALIENCY

#endif
