#include "Integrator.h"
using namespace std;

#include <iostream>
#include <stdio.h>      /* printf, scanf, puts, NULL */
#include <stdlib.h>     /* srand, rand */



RK45Integrator::RK45Integrator(VectorField2D* field):m_field(field), m_tolerance(1e-5)
{

}

RK45Integrator::~RK45Integrator(void)
{
}



//#define  STAY
#define NOSTAY




bool RK45Integrator::integrate( const Vector3& pnt, Vector3& nextPnt, double stepSize /*= 0.1*/, double* nextStepSize /*= NULL*/, bool isForward /*= true*/ )
{


		
	#ifdef STAY
	Vector3 d1,d2,d3,d4,d5,d6,v;

	if(!m_field->getValue(pnt, d1)) return false;
	//d1 *= h;
	if(!m_field->getValue(pnt + d1*(0.25), d2)) return false;
	//d2 *= h;
	if(!m_field->getValue(pnt + d1*(3.0/32)		+ d2*(9.0/32), d3)) return false;
	//d3 *= h;
	if(!m_field->getValue(pnt + d1*(1932.0/2197) + d2*(-7200.0/2197) + d3*(7296.0/2197), d4)) return false;
	//d4 *= h;
	if(!m_field->getValue(pnt + d1*(439.0/216)   + d2*(-8.0)         + d3*(3680.0/513)    + d4*(-845.0/4104), d5)) return false;
	//d5 *= h;
	if(!m_field->getValue(pnt + d1*(-8.0/27)     + d2*(2.0)          + d3*(-3544.0/2565)  + d4*(1859.0/4104) + d5 * (-11.0/40), d6)) return false;
	//d6 *= h;


	//Vector3 nextPnt0 = pnt + d1*(25.0/216) + d3*(1408.0/2565)  + d4*(2197.0/4104)   + d5*(-1.0/5);

	Vector3  vj=d1*(16.0/135) + d3*(6656.0/12825) + d4*(28561.0/56430) + d5*(-9.0/50) + d6*(2.0/55);


	double h = isForward ? this->steplenghth/vj.length() : -this->steplenghth/vj.length();


	Vector3 nextPnt1 = pnt + h*(d1*(16.0/135) + d3*(6656.0/12825) + d4*(28561.0/56430) + d5*(-9.0/50) + d6*(2.0/55));



	//if ((nextPnt1 - pnt).length() < 0.05)
    //return false;


	nextPnt = nextPnt1;

	* nextStepSize=h;

		return true;

#endif




#ifdef NOSTAY



double h = isForward ? stepSize : -stepSize;

	Vector3 d1,d2,d3,d4,d5,d6,v;


	if(!m_field->getValue(pnt, d1)) return false;
	d1 *= h;
	if(!m_field->getValue(pnt + d1*(0.25), d2)) return false;
	d2 *= h;
	if(!m_field->getValue(pnt + d1*(3.0/32)		+ d2*(9.0/32), d3)) return false;
	d3 *= h;
	if(!m_field->getValue(pnt + d1*(1932.0/2197) + d2*(-7200.0/2197) + d3*(7296.0/2197), d4)) return false;
	d4 *= h;
	if(!m_field->getValue(pnt + d1*(439.0/216)   + d2*(-8.0)         + d3*(3680.0/513)    + d4*(-845.0/4104), d5)) return false;
	d5 *= h;
	if(!m_field->getValue(pnt + d1*(-8.0/27)     + d2*(2.0)          + d3*(-3544.0/2565)  + d4*(1859.0/4104) + d5 * (-11.0/40), d6)) return false;
    d6 *= h;

	;

/*
	Vector2  vj=d1*(16.0/135) + d3*(6656.0/12825) + d4*(28561.0/56430) + d5*(-9.0/50) + d6*(2.0/55);
	double h = isForward ? this->steplenghth/vj.length() : -this->steplenghth/vj.length();
	Vector2 nextPnt1 = pnt + h*(d1*(16.0/135) + d3*(6656.0/12825) + d4*(28561.0/56430) + d5*(-9.0/50) + d6*(2.0/55));
	if ((nextPnt1 - pnt).length() < 0.05)
    return false;
	nextPnt = nextPnt1;
	* nextStepSize=h;
*/


	Vector3 nextPnt0 = pnt + d1*(25.0/216) + d3*(1408.0/2565)  + d4*(2197.0/4104)   + d5*(-1.0/5);
	Vector3 nextPnt1 = pnt + d1*(16.0/135) + d3*(6656.0/12825) + d4*(28561.0/56430) + d5*(-9.0/50) + d6*(2.0/55);



	//if ((nextPnt1 - pnt).length() < 0.00005)
		//return false;


	if((nextPnt1 - pnt).length()>max_ds)
		max_ds=(nextPnt1 - pnt).length();


	nextPnt = nextPnt1;

	if (nextStepSize)
	{
		Vector3 delta = nextPnt1 - nextPnt0;
		double deltaL = delta.length();
		double s = 1;
		if (deltaL >= 1e-6)
			s = pow(m_tolerance / (2 * delta.length()), 0.25);

		if (s < 0.75 && stepSize > 2 * m_minStep)
			*nextStepSize = stepSize / 2.0;

		if (s > 1.5 && stepSize < m_maxStep / 2)
			*nextStepSize = stepSize * 2.0;
		else			
			*nextStepSize = stepSize;
	}
	return true;


#endif

}



//����������Ҫ��ָ�� curvature torsion tortuosity


bool RK45Integrator::curvature( const Vector3& pnt,double& cur)
{

		Vector3  v,a,Cur;

	if(!m_field->getValue(pnt, v)) return false;

	if(!m_field->getAcceleration(pnt, a)) return false;


     Cur=(v.cross(a))/pow(v.length(),3);
	 cur=Cur.length();


	return true;
}


//torsion

bool RK45Integrator::torsion( const Vector3& pnt,double& cur)
{

		Vector3  v,a,sa;

	if(!m_field->getValue(pnt, v)) return false;

	if(!m_field->getAcceleration(pnt, a)) return false;

	if(!m_field->getSupAcceleration(pnt, sa)) return false;


	cur=abs(sa.dot(v.cross(a))/pow(v.cross(a).length(),2));


	return true;
}


//����tortuosity

bool RK45Integrator::tortuosity(std::deque<vec3> path,int n,double& cur)
{

       double l = 0;

	   if(n==0)
	   {
		   cur=1;

	   }else{

	for (int i = 0; i <n; ++i)
	{
		Vector3 delta = path[i+1] - path[i];
		l += delta.length();
	}

	Vector3 deltamu=path[n]-path[0];
	cur=l/deltamu.length();

	   }
	return true;

}