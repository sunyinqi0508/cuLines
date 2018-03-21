#ifndef _COMMON_H
#define _COMMON_H

#define __macro_min(a,b) ((a)<(b)?(a):(b))
#define __macro_max(a,b) ((a)>(b)?(a):(b))
#define __macro_bound(x, a, b) ((x) > (a) ? ((x) < (b) ? (x):(b)):(a))



template<typename T>
inline T constexpr cubic(const T v) noexcept { return v*v*v; }


#endif