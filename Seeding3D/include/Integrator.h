#pragma once
#include "Field2D.h"
class RK45Integrator
{
public:
	RK45Integrator(VectorField* field = NULL);
	~RK45Integrator(void);

	void				setTolerance(double t){m_tolerance = t;}
	double				getTolerance()const{m_tolerance;}

	bool				integrate(const Vector3& pnt, Vector3& nextPnt, double stepSize = 0.1, double* nextStepSize = NULL, bool isForward = true);
private:
	VectorField*			m_field;
	double				m_tolerance;			// threshold used to determine next step size
	double				m_minStep,m_maxStep;
};
