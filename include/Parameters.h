#ifndef _H_PARAMETERS
#define _H_PARAMETERS

#if  defined(__linux__) || defined(_WIN32) || defined(__unix__) || defined(_POSIX_VERSION) ||defined (__APPLE__)
// determine if it's in host code or shader
constexpr int funcpool_size = 32, L = 5, K = 4, TABLESIZE = 100;
constexpr int64_t Prime = (1ll << 32) - 5;
constexpr int maxNN = 64, similarity_window= 8;
#endif


//UI: data from dll or file
#define DLL

//Shader: Direct alpha output?
#define ALPHA_DIRECT

#endif