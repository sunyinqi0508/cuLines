/* forward declarations */
class FPSCounter;

#ifndef _FPSCOUNTER_H_
#define _FPSCOUNTER_H_

#include <cstdlib>

#if defined(WIN32)
#include <windows.h>
#include <winbase.h>
#elif defined(LINUX) || defined(__CYGWIN__)
#include <sys/time.h>
#endif


class FPSCounter
{
public:
#if defined(WIN32)
	typedef LARGE_INTEGER	fps_time;
#elif defined(LINUX) || defined(__CYGWIN__)
	typedef struct timeval	fps_time;
#endif

	FPSCounter();

	static fps_time getTime();
	static float getFPS(const fps_time &t1, const fps_time &t2);

	void stopTime();
	float getFPS() const;

private:
#if defined(WIN32)
	static fps_time	freq;
#endif
	fps_time	stoppedTime;
};

#endif /* _FPSCOUNTER_H_ */

