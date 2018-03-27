#pragma once
#include "Field2D.h"
#include "streamline.h"



class RK45Integrator
{
public:

	double            steplenghth;


	RK45Integrator(VectorField2D* field = NULL);
	~RK45Integrator(void);

	void				setTolerance(double t){m_tolerance = t;}
	double				getTolerance()const{m_tolerance;}

	bool				curvature( const Vector3& pnt,double& cur);
	bool				tortuosity(std::deque<vec3>  path,int n,double& cur);
	bool				torsion( const Vector3& pnt,double& cur);






	bool				integrate(const Vector3& pnt, Vector3& nextPnt, double stepSize = 0.1, double* nextStepSize = NULL, bool isForward = true);
private:

	VectorField2D*			m_field;
	double				m_tolerance;			// threshold used to determine next step size

	
};
extern double	 m_minStep;
extern double   m_maxStep;

extern double max_ds;
