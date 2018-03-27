#pragma once
#include <fstream>
#include <sstream>
#include <vector>
#include "Vector2.h"
#include "Vector3.h"
using namespace std;

#include <math.h>
#include <float.h>


//new





#define INVALID_VALUE		-9999

class VectorField2D
{
public:
	VectorField2D(void);
	~VectorField2D(void);

	void				clear();
	
	bool				readFile(const char* fileName);
	//bool				writeFile(const char* fileName);

//	bool				setToConstant(const Vector2& v, int nRows, int nCols);
//	bool				setToRadical(int nRows, int nCols);
//	bool				setToCircular(int nRows, int nCols);

//
	inline Vector3&		getCell(int row, int col,int hei)
	{
		return m_data[hei*(m_nRows*m_nCols)+row * m_nCols + col];
	}

	inline const Vector3& getCell(int row, int col,int hei)const
	{
		return m_data[hei*(m_nRows*m_nCols)+row * m_nCols + col];
	}

	// get value with bilinear interpolation
	// pnt.x -> row
	// pnt.y -> col
	bool				getValue( const Vector3& pnt , Vector3& dir)const;

	bool                getAcceleration( const Vector3& pnt , Vector3& acc)const;
	bool				getSupAcceleration( const Vector3& pnt , Vector3& supa)const;
    void         gen_tornado( int xs, int ys, int zs, int time );

	bool           readraw( const char* fileName );

    void                       maketornado( );
	
	void                       makeABCFlow( );

    bool    readTnado( const char* fileName );
	
	
//	bool				getLengthField(vector<double>& field)const;


	int					getNRows()const{return m_nRows;}
	int					getNCols()const{return m_nCols;}
    int				getNHeis()const{return m_nHeis;}
		
		
		
		//
	static inline bool	isInvalid(const Vector3& v)
	{
		return v.x == INVALID_VALUE || v.y == INVALID_VALUE||v.z== INVALID_VALUE;
	}
private:


	Vector3			m_minVal, m_maxVal;	// bounding rectangle
				
		//				
	int					m_nRows, m_nCols,m_nHeis;
	
	//
	vector<Vector3>		m_data;
	 int             dimension;
};





/*

class ScalarField2D
{
public:
	ScalarField2D();
	bool				readFile(const char* fileName);

	void				clear();
	bool				setToRadical(int nRows, int nCols);
	bool				setFromData(vector<double>& field, int nRows, int nCols);
	

	int					getNRows()const{return m_nRows;}
	int					getNCols()const{return m_nCols;}
	double				getMinVal()const{return m_minVal;}
	double				getMaxVal()const{return m_maxVal;}

	inline double&		getCell(int row, int col)
	{
		return m_data[row * m_nCols + col];
	}
	inline double		getCell(int row, int col)const
	{
		return m_data[row * m_nCols + col];
	}

	bool				getValue(const Vector2& pnt , double& val)const;

	static inline bool	isInvalid(const double v)
	{
		return v == INVALID_VALUE;
	}
private:
	void				computeBound();

	int					m_nRows, m_nCols;
	double				m_minVal, m_maxVal;				// bounding rectangle
	vector<double>		m_data;
};

*/