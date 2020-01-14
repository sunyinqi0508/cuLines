#include "streamline.h"

double Line::getLength() const
{
	double l = 0;
	for (int i = 0; i < path.size()-1; ++i)
	{
		vec3 delta = path[i+1] - path[i];
		l += delta.length();
	}
	return l;
}
double Line::getLength(std::deque<Vector3>& line)
{
	double l = 0;
	std::deque<Vector3>::const_iterator it, it1;
	it = line.cbegin();
	it1 = it + 1;
	while (it1 != line.cend()) {
		l += it->distance(*it1);
		it = it1;
		++it1;
	}
	return l;
}
float Line::getLength(const Vector3* this_line, const int ptr_rear, const int ptr_front) {
	float l = 0;
	for (int i = ptr_front + 2; i < ptr_rear; ++i)
		l += this_line[i].distance(this_line[i - 1]);
	return l;
}
void StreamlineParam::print() const
{
	printf("alpha:%lf\n", alpha);
	printf("dMin :%lf\n", dMin);
	printf("dSep :%lf\n", dSep);
	printf("dSelfSep:%lf\n", dSelfsep);
	printf("w    :%lf\n", w);
	printf("minLen:%lf\n", minLen);
	printf("maxLen:%lf\n", maxLen);
	printf("nHalfSample:%d\n", nHalfSample);
	printf("maxSize:%d\n", maxSize);
}
