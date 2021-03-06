#include "Field2D.h"
//#include <gl/glut.h>
#include <windows.h>
#include <stdio.h>      /* printf, scanf, puts, NULL */
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include<ctime>

#include <iostream>
using namespace std;
#define PI 3.1415926f 
#define  THREE_DATA
//#define  TWO_DATA



//#define  PI_DATA
#define  NORMAL_DATA


//ABCFlow精度

int ABCdimension = 120;




VectorField2D::VectorField2D(void):m_nRows(0), m_nCols(0),m_nHeis(0)
{
}

VectorField2D::~VectorField2D(void)
{
	
}

void VectorField2D::clear()
{
	m_nRows = m_nCols = m_nHeis=0;
	m_minVal = m_maxVal = Vector3(0,0,0);
	m_data.clear();
	
	
}


void  VectorField2D:: gen_tornado( int xs, int ys, int zs, int time )
/*
 *  Gen_Tornado creates a vector field of dimension [xs,ys,zs,3] from
 *  a proceedural function. By passing in different time arguements,
 *  a slightly different and rotating field is created.
 *
 *  The magnitude of the vector field is highest at some funnel shape
 *  and values range from 0.0 to around 0.4 (I think).
 *
 *  I just wrote these comments, 8 years after I wrote the function.
 *  
 * Developed by Roger A. Crawfis, The Ohio State University
 *
 */
{
  float x, y, z;
  int ix, iy, iz;
  float r, xc, yc, scale, temp, z0;
  float r2 = 8;
  //
  float SMALL = 0.0000000000000000001;
  float xdelta = 1.0 / (xs-1.0);
  float ydelta = 1.0 / (ys-1.0);
  float zdelta = 1.0 / (zs-1.0);

  for( iz = 0; iz < zs; iz++ )
  {
     z = iz * zdelta;                        // map z to 0->1
     xc = 0.5 + 0.1*sin(0.04*time+10.0*z);   // For each z-slice, determine the spiral circle.
     yc = 0.5 + 0.1*cos(0.03*time+3.0*z);    //    (xc,yc) determine the center of the circle.
     r = 0.1 + 0.4 * z*z + 0.1 * z * sin(8.0*z); //  The radius also changes at each z-slice.
     r2 = 0.2 + 0.1*z;                           //    r is the center radius, r2 is for damping
     for( iy = 0; iy < ys; iy++ )
     {
		y = iy * ydelta;
		for( ix = 0; ix < xs; ix++ )
		{
			x = ix * xdelta;
			temp = sqrt( (y-yc)*(y-yc) + (x-xc)*(x-xc) );
			scale = fabs( r - temp );
/*
 *  I do not like this next line. It produces a discontinuity 
 *  in the magnitude. Fix it later.
 *
 */
           if ( scale > r2 )
              scale = 0.8 - scale;
           else
              scale = 1.0;
			z0 = 0.1 * (0.1 - temp*z );
		   if ( z0 < 0.0 )  z0 = 0.0;
		   temp = sqrt( temp*temp + z0*z0 );
			scale = (r + r2 - temp) * scale / (temp + SMALL);
			scale = scale / (1+z);
	   float vx = scale * (y-yc) + 0.1*(x-xc); // added by bob;
	   // printf("%f ", vx);
          // *tornado++ = scale * (y-yc) + 0.1*(x-xc);

	   float vy = scale * (y-yc) + 0.1*(x-xc); // added by bob;
	   // printf("%f ", vy);
          // *tornado++ = scale * -(x-xc) + 0.1*(y-yc);


	   float vz = scale * (y-yc) + 0.1*(x-xc); // added by bob;
	  // printf("%f ", vz);
         //  *tornado++ = scale * z0;

		   m_data.push_back(Vector3(vx,scale * -(x-xc) + 0.1*(y-yc),scale * z0));


		}
     }
  }
}



void  VectorField2D::maketornado( )
{


 int dimension = 64;

  /* size or volume of data */
  int volume = dimension * dimension * dimension * 3;

  printf("main:: volume is %d \n", volume);

  //float data[volume];

  gen_tornado(dimension, dimension, dimension, 3);


 m_nCols = dimension;
	 
 m_nRows = dimension;
 
 m_nHeis=dimension;

}



void  VectorField2D::makeABCFlow( )
{




  /* size or volume of data */
  int volume = ABCdimension * ABCdimension * ABCdimension * 3;

  printf("main:: volume is %d \n", volume);

  //float data[volume];

  
   double A=pow(3,0.5);
   double B=pow(2,0.5);
   double C=1;

  for(int k=0;k<ABCdimension;k++)
  {

	  for(int j=0;j<ABCdimension;j++)
	  {

		  for(int i=0;i<ABCdimension;i++)
		  {

			  m_data.push_back(Vector3(A*sin(k*2*PI/ABCdimension)+B*cos(j*2*PI/ABCdimension),
				  B*sin(i*2*PI/ABCdimension)+C*cos(k*2*PI/ABCdimension),
				  C*sin(j*2*PI/ABCdimension)+A*cos(i*2*PI/ABCdimension)));

		  }
	  }
  }



 m_nCols = ABCdimension;
	 
 m_nRows = ABCdimension;
 
 m_nHeis=  ABCdimension;


}













bool VectorField2D::readTnado( const char* fileName )
{

 m_nCols = 64;
	 
 m_nRows = 64;
 
 m_nHeis= 64;

 
    ifstream file(fileName);

    for(int k=0;k<m_nHeis;k++)

{

for(int i=0;i<m_nRows;i++)
{
	
	for (int j=0;j<m_nCols;j++)
	{
		
		
		double x,y,z;
		
		 float a,b,c;

          file>>a>>b>>c;

		x=(double)a;
		
		y=(double)b;

		z=(double)c;
	
		m_data.push_back(Vector3(x,y,z));
			
			//cout<<x<<" "<<y<<" "<<z<<endl;
			
	}
}

}
  
return true;
}


bool VectorField2D::readraw( const char* fileName )

{
/*
 
m_nCols = 128;

m_nRows = 32;  

 m_nHeis= 64;  

    ifstream file(fileName);

    for(int k=0;k<m_nHeis;k++)

{

for(int i=0;i<m_nRows;i++)
{
	
	for (int j=0;j<m_nCols;j++)
	{
		
		
		double  x,y,z;
		
		 char  a,b,c;


          file>>a>>b>>c;

  

		  //速度归一化处理
		  double m=10;

		x=((double)a)/m;
		
		y=((double)b)/m;

		z=((double)c)/m;
		
		
		m_data.push_back(Vector3(x,y,z));
			
		//if(k==0&&i==0&&j<20)
			//cout<<x<<" "<<y<<" "<<z<<endl;

			
	}
}

}
*/

FILE* fp = fopen(fileName, "rb");  


//fread(info, sizeof(int), 3, fp);  


 m_nCols = 256;  

 m_nRows =128;  

 m_nHeis= 64;  



char * pVector = new char  [m_nHeis*m_nRows * m_nCols *3];  

fread(pVector, sizeof(char), m_nHeis*m_nRows * m_nCols * 3, fp);  


for(int k=0;k<m_nHeis;k++)
{
for(int i=0;i<m_nRows;i++)
{
	
	for (int j=0;j<m_nCols;j++)
	{
		
		double m=100;

		double x,y,z;
		
		x=(double)pVector[k*3*m_nRows*m_nCols+3*i*m_nCols+3*j+1]/m;
		
		y=(double)pVector[k*3*m_nRows*m_nCols+3*i*m_nCols+3*j+2]/m;

		z=(double)pVector[k*3*m_nRows*m_nCols+3*i*m_nCols+3*j+3]/m;
		

		m_data.push_back(Vector3(x,y,z));
			
			//cout<<x<<" "<<y<<" "<<z<<endl;
			
	}
}

}




fclose(fp);  
fp = NULL; 





return   true;
}




bool VectorField2D::readFile( const char* fileName )
{
	

	
#ifdef   THREE_DATA


	int info[3] ;
	
FILE* fp = fopen(fileName, "rb");  


fread(info, sizeof(int), 3, fp);  

 m_nCols = info[0];  

 m_nRows =info[1];  

 m_nHeis= info[2];  


float * pVector = new float [m_nHeis*m_nRows * m_nCols *3];  

fread(pVector, sizeof(float), m_nHeis*m_nRows * m_nCols * 3, fp);  

for(int k=0;k<m_nHeis;k++)

{

for(int i=0;i<m_nRows;i++)
{
	
	for (int j=0;j<m_nCols;j++)
	{
		
		
		double x,y,z;
		
		x=(double)pVector[k*3*m_nRows*m_nCols+3*i*m_nCols+3*j+1];
		
		y=(double)pVector[k*3*m_nRows*m_nCols+3*i*m_nCols+3*j+2];

		z=(double)pVector[k*3*m_nRows*m_nCols+3*i*m_nCols+3*j+3];
		
		
		m_data.push_back(Vector3(x,y,z));
			
			//cout<<x<<" "<<y<<" "<<z<<endl;
			
	}
}

}

fclose(fp);  
fp = NULL;  

#endif
   
//////////////////////////////////////////  
// ... ...  
//////////////////////////////////////////  

	//Êý¾ÝµãµÄÁÐÊý µ±±¾³ÌÐòÎ¨Ò»ÐèÒªµ÷µÄÊý 

	/*
	int point_Cols=41;	
	int point_Rows=35;
	ifstream file(fileName);
	if (!file)
		return false;

	clear();

	int ithRow = 0;
while(true){
		string line1,line2,line3;
		
		file >> line1>>line2>>line3;
		
		if (!file)
		{
			break;
		}
			
		stringstream sx,sy,sz;
		
		sx << line1;
		
		sy <<line2;
		
		sz <<line3;
		
					
			double x,y,z;

			sx>>x;
			sy>>y;
			sz>>z;
			
			m_data.push_back(Vector3(x,y,z));
			
			//cout<<x<<" "<<y<<" "<<z<<endl;
						
	 	++ithRow;
		

}

	
	m_nCols=point_Cols;
	
	m_nRows=point_Rows;
	
	m_nHeis = ithRow/point_Cols/point_Rows;

*/


#ifdef  TWO_DATA

int point_Cols=41;	

	ifstream file(fileName);

	if (!file)
		return false;

	clear();

	int ithRow = 0;
	while (true)
	{
		

		string line1,line2;
		
		file >> line1>>line2;
	
		
		if (!file)
		{
			break;
		}
			
 	
		stringstream sx,sy;
		
		sx << line1;
		
		sy <<line2;
		
				
			double x,y;

			sx>>x;
			sy>>y;
			
			m_data.push_back(Vector3(x,y,0));
			
			//cout<<x<<" "<<y<<endl;

		++ithRow;

	}


	m_nCols=point_Cols;
	
	m_nRows = ithRow/point_Cols;

    m_nHeis=1;
	

#endif
	
	return true;
	
}



//


/* 
bool VectorField2D::writeFile( const char* fileName )
{
	ofstream file(fileName);
	if (!file)
		return false;

	for (int ithRow = 0, i = 0; ithRow < m_nRows; ++ithRow)
	{
		for (int ithCol = 0; ithCol < m_nCols; ++ithCol, ++i)
		{
			Vector2& v = m_data[i];
			file << v.x << ',' << v.y << ',';

		}
		file << endl;
	}
	file.close();
	return true;
}

*/


/*
void VectorField2D::computeBound()
{
	double min[3] = {DBL_MAX, DBL_MAX,DBL_MAX};
	double max[3] = {-DBL_MAX, -DBL_MAX, -DBL_MAX};

	for (int ithHei = 0, i = 0; ithHei < m_nHeis; ++ithHei)
	{
	for (int ithRow = 0; ithRow < m_nRows; ++ithRow)
	{
		for (int ithCol = 0; ithCol < m_nCols; ++ithCol, ++i)
		{
			Vector3& v = m_data[i];
			if (v.x == INVALID_VALUE || v.y == INVALID_VALUE||v.z == INVALID_VALUE)
				continue;
			min[0] = min[0] < v.x ? min[0] : v.x;
			min[1] = min[1] < v.y ? min[1] : v.y;
			min[2] = min[2] < v.z ? min[2] : v.z;
			
			max[0] = max[0] > v.x ? max[0] : v.x;
			max[1] = max[1] > v.y ? max[1] : v.y;
			max[2] = max[2] < v.z ? max[2] : v.z;
		}
	}
	
}
	
	
	m_minVal = Vector3(min[0], min[1],min[2]);
	m_maxVal = Vector3(max[0], max[1],max[2]);
	
}



*/


//getValueÆäÊµÊÇÖªµÀÎ»ÖÃÇó¸ÃµãµÄ²åÖµËÙ¶ÈÖµ





#ifdef PI_DATA

bool VectorField2D::getValue( const Vector3& pnt , Vector3& dir)const
{

	double row = pnt.y/(2*PI)*ABCdimension;

	double col = pnt.x/(2*PI)*ABCdimension;

	double hei=pnt.z/(2*PI)*ABCdimension;

	//判断是否出界
	
	if (row < 0 || row >= m_nRows-1 || col < 0 || col >= m_nCols-1||hei<0||hei>=m_nHeis-1)
		return false;
		
	int rowi = row, coli = col, heii=hei;
	
	double rowf = row - rowi, colf = col - coli, heif=hei-heii;
	
	const Vector3& v000 = getCell(rowi, coli,hei);
	const Vector3& v010 = getCell(rowi, coli+1,hei);
	const Vector3& v100 = getCell(rowi+1, coli,hei);
	const Vector3& v110 = getCell(rowi+1, coli+1,hei);
	
	const Vector3& v001 = getCell(rowi, coli,hei+1);
	const Vector3& v111 = getCell(rowi+1, coli+1,hei+1);
	const Vector3& v101 = getCell(rowi+1, coli,hei+1);
	const Vector3& v011 = getCell(rowi, coli+1,hei+1);
	
	
//if is valled


	// bilinear intepolation
	Vector3 v00  = v000 * (1-colf) + v010 * colf;
	Vector3 v01  = v100 * (1-colf) + v110 * colf;
	Vector3 v0 = v00 * (1-rowf) + v01 * rowf;
	
	
	Vector3 v10  = v001 * (1-colf) + v011 * colf;
	Vector3 v11  = v101 * (1-colf) + v111 * colf;
	Vector3 v1 = v10 * (1-rowf) + v01 * rowf;
	
	dir=v0*(1-heif)+v1*heif;
	
	
	return true;
}



#endif


#ifdef NORMAL_DATA

bool VectorField2D::getValue( const Vector3& pnt , Vector3& dir)const
{

	double row = pnt.y;

	double col = pnt.x;

	double hei=pnt.z;

	//判断是否出界
	
	if (row < 0 || row >= m_nRows-1 || col < 0 || col >= m_nCols-1||hei<0||hei>=m_nHeis-1)
		return false;
		
	int rowi = row, coli = col, heii=hei;
	
	double rowf = row - rowi, colf = col - coli, heif=hei-heii;
	
	const Vector3& v000 = getCell(rowi, coli,hei);
	const Vector3& v010 = getCell(rowi, coli+1,hei);
	const Vector3& v100 = getCell(rowi+1, coli,hei);
	const Vector3& v110 = getCell(rowi+1, coli+1,hei);
	
	const Vector3& v001 = getCell(rowi, coli,hei+1);
	const Vector3& v111 = getCell(rowi+1, coli+1,hei+1);
	const Vector3& v101 = getCell(rowi+1, coli,hei+1);
	const Vector3& v011 = getCell(rowi, coli+1,hei+1);
	
	
//if is valled


	// bilinear intepolation
	Vector3 v00  = v000 * (1-colf) + v010 * colf;
	Vector3 v01  = v100 * (1-colf) + v110 * colf;
	Vector3 v0 = v00 * (1-rowf) + v01 * rowf;
	
	
	Vector3 v10  = v001 * (1-colf) + v011 * colf;
	Vector3 v11  = v101 * (1-colf) + v111 * colf;
	Vector3 v1 = v10 * (1-rowf) + v01 * rowf;
	
	dir=v0*(1-heif)+v1*heif;
	
	
	return true;
}



#endif



//求场点处加速度
bool VectorField2D::getAcceleration( const Vector3& pnt , Vector3& acc)const

{
	 Vector3 dx;
	 dx.x=0.1;
	 dx.y=0;
	 dx.z=0;

	Vector3 dy;
	dy.x=0;
	dy.y=0.1;
	dy.z=0;

	Vector3 dz;
	dz.x=0;
	dz.y=0;
	dz.z=0.1;


	Vector3  v,vx,vy,vz,dv;

	if(!getValue(pnt, v)) return false;
	
	if(!getValue(pnt+dx, vx)) return false;

	if(!getValue(pnt+dy, vy)) return false;

	if(!getValue(pnt+dz, vz)) return false;

	dv.x=(vx.x-v.x)/dx.length();

	dv.y=(vy.y-v.y)/dy.length();

	dv.z=(vz.z-v.z)/dz.length();


	 acc=dv*v;


     return true;

}





bool VectorField2D::getSupAcceleration( const Vector3& pnt , Vector3& supa)const

{
  Vector3 dx;
	 dx.x=0.1;
	 dx.y=0;
	 dx.z=0;

	Vector3 dy;
	dy.x=0;
	dy.y=0.1;
	dy.z=0;

	Vector3 dz;
	dz.x=0;
	dz.y=0;
	dz.z=0.1;




	Vector3   a,ax,ay,az,da;

	if(!getAcceleration(pnt, a)) return false;
	
	if(!getAcceleration(pnt + dx, ax)) return false;

	if(!getAcceleration(pnt + dy, ay)) return false;

	if(!getAcceleration(pnt + dz, az)) return false;



	da.x=(ax.x-a.x)/dx.length();
	da.y=(ay.y-a.y)/dy.length();
	da.z=(az.z-a.z)/dz.length();


	 supa=da*a;
     return true;

}
















/*

bool VectorField2D::setToConstant( const Vector2& v, int nRows, int nCols )
{
	if (nRows <= 0 || nCols <= 0)
		return false;

	m_nRows = nRows;
	m_nCols = nCols;
	m_data.resize(m_nRows * m_nCols);
	for (int i = 0; i < m_nCols*m_nRows; ++i)
	{
		m_data[i] = v;
	}
	computeBound();
	return true;
}








bool VectorField2D::setToRadical(int nRows, int nCols)
{
	if (nRows <= 0 || nCols <= 0)
		return false;

	m_nRows = nRows;
	m_nCols = nCols;
	m_data.resize(m_nRows * m_nCols);
	int center[2] = {nCols / 2, nRows/2};
	for (int y = 0, i = 0; y < m_nRows; y++)
	{
		for (int x = 0; x < m_nCols; x++, i++)
		{
			Vector2 v(x-center[0], y-center[1]);
			m_data[i] = v.normalized();
		}
	}
	computeBound();
	return true;
}




bool VectorField2D::setToCircular(int nRows, int nCols)
{
	if (nRows <= 0 || nCols <= 0)
		return false;

	m_nRows = nRows;
	m_nCols = nCols;
	m_data.resize(m_nRows * m_nCols);
	int center[2] = {nCols / 2, nRows/2};
	for (int y = 0, i = 0; y < m_nRows; y++)
	{
		for (int x = 0; x < m_nCols; x++, i++)
		{
			Vector2 v(x-center[0], y-center[1]);
			v.normalize();
			m_data[i] = Vector2(-v.y, v.x);
		}
	}
	computeBound();
	return true;
}







bool VectorField2D::getLengthField( vector<double>& field ) const
{
	if (m_nRows <= 0 || m_nCols <= 0)
		return false;

	field.resize(m_nRows * m_nCols);
	for (int y = 0, i = 0; y < m_nRows; y++)
	{
		for (int x = 0; x < m_nCols; x++, i++)
		{
			const Vector2& cell = getCell(y, x);
			double length = isInvalid(cell) ? INVALID_VALUE : cell.length();
			field[i] = length;
		}
	}
	return true;
}





*/
















/*


ScalarField2D::ScalarField2D() :m_nRows(0), m_nCols(0), m_minVal(FLT_MAX), m_maxVal(-FLT_MAX)
{

}

void ScalarField2D::computeBound()
{
	double min = DBL_MAX;
	double max = -DBL_MAX;

	for (int ithRow = 0, i = 0; ithRow < m_nRows; ++ithRow)
	{
		for (int ithCol = 0; ithCol < m_nCols; ++ithCol, ++i)
		{
			double v = m_data[i];
			if (v == INVALID_VALUE)
				continue;
			min = min < v ? min : v;
			max = max > v ? max : v;
		}
	}
	m_minVal = min;
	m_maxVal = max;
}

bool ScalarField2D::setToRadical( int nRows, int nCols )
{
	if (nRows <= 0 || nCols <= 0)
		return false;

	m_nRows = nRows;
	m_nCols = nCols;
	m_data.resize(m_nRows * m_nCols);
	int center[2] = {nCols / 2, nRows/2};
	for (int y = 0, i = 0; y < m_nRows; y++)
	{
		for (int x = 0; x < m_nCols; x++, i++)
		{
			Vector2 v(x-center[0], y-center[1]);
			double len = v.length() / min(nRows, nCols);
			m_data[i] = exp(-len*len);
		}
	}
	computeBound();
	return true;
}








bool ScalarField2D::getValue( const Vector2& pnt , double& val ) const
{
	double row = pnt.y;
	double col = pnt.x;
	if (row < 0 || row >= m_nRows-1 || col < 0 || col >= m_nCols-1)
		return false;
	int rowi = row, coli = col;
	double rowf = row - rowi, colf = col - coli;
	const double& v00 = getCell(rowi, coli);
	const double& v01 = getCell(rowi, coli+1);
	const double& v10 = getCell(rowi+1, coli);
	const double& v11 = getCell(rowi+1, coli+1);

	if (isInvalid(v00) || isInvalid(v01) || isInvalid(v10) || isInvalid(v11))
	{
		return false;
	}

	// bilinear intepolation
	double v0  = v00 * (1-colf) + v01 * colf;
	double v1  = v10 * (1-colf) + v11 * colf;

	val = v0 * (1-rowf) + v1 * rowf;
	return true;
}

bool ScalarField2D::setFromData( vector<double>& field, int nRows, int nCols )
{
	if (nRows <= 0 || nCols <= 0 || nRows * nCols != field.size() || field.size() <= 0)
	{
		return false;
	}
	m_data = field;
	m_nRows = nRows;
	m_nCols = nCols;
	computeBound();
	return true;
}

void ScalarField2D::clear()
{
	m_data.clear();m_nCols = m_nRows = 0; m_minVal = DBL_MAX; m_maxVal = -DBL_MAX;
}

bool ScalarField2D::readFile( const char* fileName )
{
	ifstream file(fileName);
	if (!file)
		return false;

	clear();

	int ithRow = 0;
	while (true)
	{
		string line;
		file >> line;
		if (!file)
		{
			break;
		}

		stringstream ss;
		ss << line;

		int ithCol = 0;
		while (true)
		{
			char comma0;
			double x;
			ss >> x >> comma0;
			if (!ss)break;

			++ithCol;
			m_data.push_back(x);
		}

		if (m_nCols == 0)
		{
			m_nCols = ithCol;
		}
		else if (m_nCols != ithCol)
		{
			clear();
			return false;
		}
		++ithRow;
	}
	m_nRows = ithRow;
	computeBound();
	return true;
}


*/