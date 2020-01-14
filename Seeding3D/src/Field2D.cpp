#include "Field2D.h"
#include <stdio.h>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <vector>
using namespace std;
VectorField::VectorField(void):m_dimensions(0)
{
	
}

VectorField::~VectorField(void)
{
}

void VectorField::clear()
{
	m_nRows = m_nCols = m_nLayers = 0;
	m_minVal = m_maxVal = Vector3(0,0, 0);
	m_data.clear();
}
bool VectorField::readVecFile(const char* filename) {
	FILE *fp;
	fopen_s(&fp, filename, "r");
//	char *buf = new char[256];
	fscanf_s(fp, "Gvec");
	fscanf_s(fp, "%d %d ", &m_nCols, &m_nRows);
	for (int i = 0; i < m_nCols * m_nRows; i++) {
		float _x, _y;
		fscanf_s(fp, "%f %f ", &_x, &_y);
		m_data.push_back(Vector3(_x, _y, 0));
	}

	m_nLayers = 1;
	fclose(fp);
	computeBound();
	return true;
}
bool VectorField::readTXTFile(const char* filename) {
	FILE *fp;
	fopen_s(&fp, filename, "r");
////	char *buf = new char[256];
	fscanf_s(fp, "G\n");
	fscanf_s(fp, "I=%d,J=%d\n", &m_nCols, &m_nRows);
	for (int i = 0; i < m_nCols * m_nRows; i++) {
		float _x, _y, _dump;
		fscanf_s(fp, "%f,%f,%f,%f\n", &_dump, &_dump,&_x, &_y);
		m_data.push_back(Vector3(_x, _y, 0));
	}
	m_nLayers = 1;
	fclose(fp);
	computeBound();
	return true;
}
bool VectorField::readFile( const char* fileName )
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
			char comma0, comma1;
			double x,y;
			ss >> x >> comma0 >> y >> comma1;
			if (!ss)break;

			++ithCol;
			m_data.push_back(Vector3(x,y, 0));
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
	m_nLayers = 1;

	m_nRows = ithRow;
	computeBound();
	return true;
}

bool VectorField::writeFile( const char* fileName )
{
	ofstream file(fileName);
	if (!file)
		return false;

	for (int ithRow = 0, i = 0; ithRow < m_nRows; ++ithRow)
	{
		for (int ithCol = 0; ithCol < m_nCols; ++ithCol, ++i)
		{
			Vector3& v = m_data[i];
			file << v.x << ',' << v.y << ',';

		}
		file << endl;
	}
	file.close();
	return true;
}

void VectorField::computeBound()
{
	double min[3] = {DBL_MAX, DBL_MAX, DBL_MAX };
	double max[3] = {-DBL_MAX, -DBL_MAX, -DBL_MAX};
	for(int ithLayer = 0, i = 0;  ithLayer < m_nLayers; ++ithLayer)
		for (int ithRow = 0; ithRow < m_nRows; ++ithRow)
		{
			for (int ithCol = 0; ithCol < m_nCols; ++ithCol, ++i)
			{
				Vector3& v = m_data[i];
				if (v.x == INVALID_VALUE || v.y == INVALID_VALUE)
					continue;
				min[0] = min[0] < v.x ? min[0] : v.x;
				min[1] = min[1] < v.y ? min[1] : v.y;
				min[2] = min[2] < v.z ? min[2] : v.z;

				max[0] = max[0] > v.x ? max[0] : v.x;
				max[1] = max[1] > v.y ? max[1] : v.y;
				max[2] = max[2] > v.z ? max[2] : v.z;
			}
		}
	m_minVal = Vector3(min[0], min[1], min[2]);
	m_maxVal = Vector3(max[0], max[1], max[2]);
}

bool VectorField::getValue( const Vector3& pnt , Vector3& dir)const
{
	double row = pnt.y;
	double col = pnt.x;
	double layer = pnt.z;
	if (row < 0 || row >= m_nRows || col < 0 || col >= m_nCols||layer<0||layer>=m_nLayers)
		return false;
	Vector<int> coordi(col, row, layer);
	Vector<double> coordf(pnt - coordi);
	for (int i = 0; i < 3; i++)
		coordf[i] = coordi[i] + 1 < m_dimensions[i] ? coordf[i] : 0;

	Vector3 results[2] = { 0, 0 };
	int stepsi = 1;
	if (coordf[0] > 1e-7)
		stepsi = 2;
	for (int i = 0; i < stepsi; i++)
	{
		int stepsj = 1;
		if (coordf[1] > 1e-7)
			stepsj = 2;
		Vector3 sum_l[2] = { 0, 0 };

		for (int j = 0; j < stepsj; j++)
		{
			sum_l[j] = getCell(coordi[0] + i, coordi[1] + j, coordi[2]);

			if (coordf[2] > 1e-7)
			{
				const Vector3& cell1 = getCell(coordi[0] + i, coordi[1] + j, coordi[2] + 1);
				if (!isInvalid(cell1))
					if (!isInvalid(sum_l[j]))
						sum_l[j] = sum_l[j] * (1 - coordf[2]) + cell1 * coordf[2];
					else
						sum_l[j] = cell1;
			}
		}

		results[i] = stepsj == 1 ?
			sum_l[0] :
				isInvalid(sum_l[0]) ? sum_l[1]:
					isInvalid(sum_l[1]) ? sum_l[0]:
						sum_l[0] * (1 - coordf[1]) + sum_l [1] * coordf[1];
		
	}
	
	Vector3 res =
		stepsi == 1 ? results[0] :
			isInvalid(results[0]) ? results[1]:
				isInvalid(results[1])? results[0]:
					results[0] * (1 - coordf[0]) + results[1] * coordf[0];
	if (isInvalid(res))
		return false;
	else
		dir = res;
	return true;
}

bool VectorField::setToConstant( const Vector3& v, int nRows, int nCols, int nLayers )
{
	if (nRows <= 0 || nCols <= 0|| nLayers <= 0)
		return false;

	m_nRows = nRows;
	m_nCols = nCols;
	m_nLayers = nLayers;
	m_data.resize(m_nRows * m_nCols*m_nLayers);
	for (int i = 0; i < m_nCols*m_nRows*m_nLayers; ++i)
		m_data[i] = v;
	
	computeBound();
	return true;
}

bool VectorField::setToRadical(int nRows, int nCols, int nLayers)
{
	if (nRows <= 0 || nCols <= 0||nLayers <= 0)
		return false;

	m_nRows = nRows;
	m_nCols = nCols;
	m_nLayers = nLayers;
	m_data.resize(m_nRows * m_nCols*m_nLayers);
	int center[3] = {nCols / 2, nRows/2, nLayers/2};
	for (int y = 0, i = 0; y < m_nRows; y++)
		for (int x = 0; x < m_nCols; x++)
			for (int z = 0; z < m_nCols; z++, i++)
				m_data[i] = Vector3(x-center[0], y-center[1], z - center[2]).normalized();
	computeBound();
	return true;
}

bool VectorField::setToCircular(int nRows, int nCols, int nLayers)
{
	if (nRows <= 0 || nCols <= 0||nLayers <= 0)
		return false;

	m_nRows = nRows;
	m_nCols = nCols;
	m_nLayers = nLayers;
	m_data.resize(m_nRows * m_nCols*m_nLayers);
	int center[] = {nCols / 2, nRows/2, nLayers/2};
	for (int y = 0, i = 0; y < m_nRows; y++)
		for (int x = 0; x < m_nCols; x++)
			for (int z = 0; z < m_nLayers; z++, i++)
				m_data[i] = Vector3(center[1] - y, x - center[0], z - center[2]).normalized();
	computeBound();
	return true;
}

bool VectorField::getLengthField( vector<double>& field ) const
{
	if (m_nRows <= 0 || m_nCols <= 0)
		return false;

	field.resize(m_nRows * m_nCols * m_nLayers);
	for (int y = 0, i = 0; y < m_nRows; y++)
		for (int x = 0; x < m_nCols; x++)
			for (int z = 0; z < m_nLayers; ++z, ++i)
			{
				const Vector3& cell = getCell(y, x ,z);
				double length = isInvalid(cell) ? INVALID_VALUE : cell.length();
				field[i] = length;
			}
	return true;
}
