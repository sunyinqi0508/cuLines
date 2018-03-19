#ifndef _VECTOR_H
#define _VECTOR_H
#define _USE_MATH_DEFINES
#include <math.h>
#include <functional>

#define OP(OPERAND)	inline Vector3 operator OPERAND(const Vector3& v) const {\
		Vector3 res;\
		res.x = x OPERAND v.x;\
		res.y = y OPERAND v.y;\
		res.z = z OPERAND v.z;\
		return res;\
	}\
	inline Vector3 operator OPERAND(const float v) const {\
		Vector3 res; \
		res.x = x OPERAND v; \
		res.y = y OPERAND v; \
		res.z = z OPERAND v; \
		return res; \
	}
#define OPEQ(OPERAND)	inline Vector3 operator OPERAND(const Vector3& v) {\
		x OPERAND v.x;\
		y OPERAND v.y;\
		z OPERAND v.z;\
		return *this;\
	}\
	inline Vector3 operator OPERAND(const float v) {\
			x OPERAND v; \
			y OPERAND v; \
			z OPERAND v; \
			return *this; \
	}



#define OPERATIONS OP(+) OP(-) OP (*) OP(/)\
	OPEQ(+=) OPEQ(-=) OPEQ(*=) OPEQ(/=)

class Vector3 {
public:
	Vector3() = default;
	Vector3(float s) : x(s), y(s), z(s) {}
	Vector3(float x, float y, float z) 
		:x(x), y(y), z(z) {}
	
	inline operator float*() { return (float*)this; }
	inline operator float() { return length(); }
	void operator ()
		(float _x, float _y, float _z) { x = _x, y = _y, z = _z; }

	OPERATIONS;

	float sq() const { return x * x + y * y + z * z; }
	float length() const { return sqrt(sq()); }
	inline float l1Norm() { return abs(x) + abs(y) + abs(z); }

	inline Vector3 normalize(){ return *this / length(); }

	inline Vector3& normalized() {
		operator/= (length());
		return *this;
	}

	inline float dot(const Vector3& v) const { return v.x * x + v.y * y + v.z * z; }
	inline Vector3 cross(const Vector3& v) const {
		Vector3 res;
		res.x = y * v.z - z * v.y;
		res.y = z * v.x - x * v.z;
		res.z = x * v.y - y * v.x;
		return res;
	}
	
	float x, y, z;
};
#endif