#pragma once

#include "FileIO.h"
#include "Vector.h"

#define INVALID_VALUE		NAN
class LshFunc;
class VectorField
{
public:
	VectorField(void);
	~VectorField(void);
	friend int FileIO::FieldfromAmiramesh(void*, const char*);
	friend std::vector<LshFunc> create_seeding(int n_funcs, VectorField& field);
	void				clear();
	bool readVecFile(const char * filename);
	bool readTXTFile(const char * filename);
	bool				readFile(const char* fileName);
	bool				writeFile(const char* fileName);

	bool				setToConstant(const Vector3& v, int nRows, int nCols, int nLayer);
	bool				setToRadical(int nRows, int nCols, int nLayer);
	bool				setToCircular(int nRows, int nCols, int nLayer);

	inline Vector3&		getCell(int col, int row, int layer)
	{
		return m_data[col + m_nCols * (row + m_nRows * layer)];
	}
	inline const Vector3& getCell(int col, int row, int layer)const
	{
		return m_data[col + m_nCols * (row + m_nRows * layer)];
	}

	// get value with bilinear interpolation

	bool				getValue( const Vector3& pnt , Vector3& dir)const;
	bool				getLengthField(std::vector<double>& field)const;
	Vector<int>			getDimensions() const { return m_dimensions; }
	int					getNRows()const{return m_nRows;}
	int					getNCols()const{return m_nCols;}
	int					getNLayers()const { return m_nLayers; }
	static inline bool	isInvalid(const Vector3& v)
	{
		return isnan(v.x)||isnan(v.y)||isnan(v.z);
	}
private:
	void				computeBound();

	Vector3				m_minVal, m_maxVal;				// bounding rectangle
	Vector<int>			m_dimensions;
	int					&m_nRows = m_dimensions.y, &m_nCols = m_dimensions.x, &m_nLayers = m_dimensions.z;
	std::vector<Vector3>		m_data;
};

