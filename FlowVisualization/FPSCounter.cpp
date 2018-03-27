#include "FPSCounter.h"


#if defined(WIN32)
FPSCounter::fps_time FPSCounter::freq;
#endif

FPSCounter::FPSCounter()
{
#if defined(WIN32)
	QueryPerformanceFrequency(&freq);
#endif
}

FPSCounter::fps_time FPSCounter::getTime()
{
	fps_time	t;

#if defined(WIN32)
	QueryPerformanceCounter(&t);
#elif defined(LINUX) || defined(__CYGWIN__)
	gettimeofday(&t, NULL);
#endif

	return (t);
}

float FPSCounter::getFPS(const fps_time &t1, const fps_time &t2)
{
#if defined(WIN32)
	return ((float)freq.QuadPart / (t2.QuadPart - t1.QuadPart));
#elif defined(LINUX) || defined(__CYGWIN__)
	int	mu;

	mu = 1000000 * (t2.tv_sec  - t1.tv_sec)
	             + (t2.tv_usec - t1.tv_usec);

	return (1000000.0f / mu);
#endif
}

void FPSCounter::stopTime()
{
	stoppedTime = getTime();
}

float FPSCounter::getFPS() const
{
	return (getFPS(stoppedTime, getTime()));
}

