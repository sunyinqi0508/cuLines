#ifndef _VECTOR_H
#define _VECTOR_H
#define _USE_MATH_DEFINES
#include <math.h>
#include <Common.h>
#ifdef __CUDACC__
#define _DEVICE_HOST __device__
#else
#define _DEVICE_HOST
#endif

#define OP(OPERATOR)\
_DEVICE_HOST \
inline Vector<T> operator OPERATOR(const Vector<T>& v) const {\
		Vector<T> res;\
		res.x = x OPERATOR v.x;\
		res.y = y OPERATOR v.y;\
		res.z = z OPERATOR v.z;\
		return res;\
	}\
template<class T2>\
_DEVICE_HOST \
inline Vector operator OPERATOR(const T2 v) const {\
		Vector<T> res; \
		res.x = x OPERATOR v; \
		res.y = y OPERATOR v; \
		res.z = z OPERATOR v; \
		return res; \
	}
#define OPEQ(OPERATOR)\
_DEVICE_HOST \
inline Vector<T> operator OPERATOR(const Vector<T>& v) {\
		x OPERATOR v.x;\
		y OPERATOR v.y;\
		z OPERATOR v.z;\
		return *this;\
	}\
_DEVICE_HOST \
inline Vector<T> operator OPERATOR(const T v) {\
			x OPERATOR v; \
			y OPERATOR v; \
			z OPERATOR v; \
			return *this; \
	}


#define OPERATIONS OP(+) OP(-) OP (*) OP(/)\
	OPEQ(+=) OPEQ(-=) OPEQ(*=) OPEQ(/=)

template<class T = float>
class Vector {
public:
	_DEVICE_HOST
	Vector() = default;
	_DEVICE_HOST
	constexpr Vector(T s) noexcept : x(s), y(s), z(s) {}
	_DEVICE_HOST
	constexpr Vector(T x, T y, T z) noexcept
		:x(x), y(y), z(z) {}

	template<class T2>
	_DEVICE_HOST
	constexpr Vector(const Vector<T2> &v) : x(v.x), y(v.y), z(v.z) {}
	_DEVICE_HOST
	inline operator T*() { return (T*)this; }
	_DEVICE_HOST
	inline operator const T*() const { return (const T*)this; }
	_DEVICE_HOST
	inline operator T() const { return length(); }
	_DEVICE_HOST
	void operator ()
		(T _x, T _y, T _z) { x = _x, y = _y, z = _z; }

	OPERATIONS;

	_DEVICE_HOST
	T sq() const { return x * x + y * y + z * z; }
	_DEVICE_HOST
	T length() const { return sqrt(sq()); }
	_DEVICE_HOST
	inline T l1Norm() const { return abs(x) + abs(y) + abs(z); }
	_DEVICE_HOST
	inline Vector normalize() const { return *this / length(); }
	_DEVICE_HOST
	inline Vector& normalized() {
		operator/= (length());
		return *this;
	}
	_DEVICE_HOST
	inline Vector& normalized_checked() {
		const T l = length();

		if (l)
			operator /= (l);
		else
			x = y = z = 0;

		return *this;
	}
	_DEVICE_HOST
	inline T dot(const Vector& v) const { return v.x * x + v.y * y + v.z * z; }
	_DEVICE_HOST
	inline Vector cross(const Vector& v) const {
		Vector res;
		res.x = y * v.z - z * v.y;
		res.y = z * v.x - x * v.z;
		res.z = x * v.y - y * v.x;
		return res;
	}
	_DEVICE_HOST
    inline T project(const Vector &target) const {
        return dot(target) / target.length();
    }
	_DEVICE_HOST
    inline T distance(const Vector &that) const {
        return (*this - that).length();
    }
	_DEVICE_HOST
	inline T sqDist(const Vector& target) const {
		T sqDist = 0;
		sqDist += square(x - target.x);
		sqDist += square(y - target.y);
		sqDist += square(z - target.z);
		return sqDist;
	}


	static __forceinline Vector<T> abs(const Vector<T>& val) {
		return { std::abs(val.x), std::abs(val.y), std::abs(val.z) };
	}
	static __forceinline Vector<T> max(const Vector<T>& lhs, const Vector<T>& rhs) {
		return { __macro_max(lhs.x, rhs.x), __macro_max(lhs.y, rhs.y), __macro_max(lhs.z, rhs.z) };
	}
	static __forceinline Vector<T> min(const Vector<T>& lhs, const Vector<T>& rhs) {
		return { __macro_min(lhs.x, rhs.x), __macro_min(lhs.y, rhs.y), __macro_min(lhs.z, rhs.z) };
	}
	T x, y, z;
};

using Vector3 = Vector<>;

#endif
