#pragma once
#include <Vector.h>

class fColor :public Vector<> {
public:
	using Vector<>::Vector; 
	fColor(int iColor):Vector<>(((iColor >> 24) & 0xff) / 255.f,
		((iColor >> 16) & 0xff) / 255.f,
		((iColor >> 8) & 0xff) / 255.f) {}
	operator float*() {
		return Vector<>::operator float *();
	}
	static fColor fromIColors(int color, float* alpha = 0) {
		if(alpha)
			*alpha = (color & 0xff) / 255.f;

		return fColor(color);
	}
	static int toIntColor(Vector<> color, float alpha = 1.f) {
		return ((int)(color.x *255.) & 0xff) << 24 +
			((int)(color.y *255.) & 0xff) << 16 +
			((int)(color.z *255.) & 0xff) << 8 +
			((int)(255.f*alpha) & 0xff);
	}
	int toIntColor(const float alpha = 1.f) {
		return (((int)(x *255.) & 0xff) << 24) +
			(((int)(y *255.) & 0xff) << 16) +
			(((int)(z *255.) & 0xff) << 8) +
			((int)(255.f*alpha)&0xff);
	}
	int*toIntColors(){
		int *colors = new int[3];
		colors[0] = (((int)(x *255.) & 0xff) << 24);
		colors[1] = (((int)(y *255.) & 0xff) << 16);
		colors[2] = (((int)(z *255.) & 0xff) << 8);

		return colors;
	}

};
class SphereType {

public:
	struct ColorView {
		int r:8;
		int g:8;
		int b:8;
		int a:8;
		operator int&() {
			return *reinterpret_cast<int*>(this);
		}
		ColorView(int ival) {((int&)*this) = ival; }
	};
	SphereType() = default;
	SphereType(Vector3 origin, float radius) :origin(origin), radius(radius), color(0xffffffff) {}
	SphereType(Vector3 origin, float radius, int color) :origin(origin), radius(radius), color(color) {}
	SphereType(Vector3 origin, float radius, Vector3 color, float alpha) :origin(origin), color(fColor::toIntColor(color, alpha)) {}

	Vector3 origin;
	float radius;
	ColorView color;
	
};
