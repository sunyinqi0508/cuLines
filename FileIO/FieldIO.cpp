#include "FileIO.h"
#include "Field2D.h"
#include <stdio.h>
#include <string.h>
namespace FileIO {



	const char* FindAndJump(const char* buffer, const char* SearchString)
	{
		const char* FoundLoc = strstr(buffer, SearchString);
		if (FoundLoc) return FoundLoc + strlen(SearchString);
		return buffer;
	}


	int FieldfromAmiramesh(void *_field, const char* FileName)
	{
		FILE* fp = 0;
		fopen_s(&fp, FileName, "rb");
		if (!fp)
			return 1;

		char buffer[2048];
		fread(buffer, sizeof(char), 2047, fp);
		buffer[2047] = '\0'; //The following string routines prefer null-terminated strings

		if (!strstr(buffer, "# AmiraMesh BINARY-LITTLE-ENDIAN 2.1"))
		{
			printf("Not a proper AmiraMesh file.\n");
			fclose(fp);
			return 1;
		}

		int xDim(0), yDim(0), zDim(0);
		sscanf(FindAndJump(buffer, "define Lattice"), "%d %d %d", &xDim, &yDim, &zDim);

		float xmin(1.0f), ymin(1.0f), zmin(1.0f);
		float xmax(-1.0f), ymax(-1.0f), zmax(-1.0f);
		sscanf(FindAndJump(buffer, "BoundingBox"), "%g %g %g %g %g %g", &xmin, &xmax, &ymin, &ymax, &zmin, &zmax);

		const bool bIsUniform = (strstr(buffer, "CoordType \"uniform\"") != NULL);

		int NumComponents(0);
		if (strstr(buffer, "Lattice { float Data }"))
			NumComponents = 1;
		else
			sscanf(FindAndJump(buffer, "Lattice { float["), "%d", &NumComponents);

		if (xDim <= 0 || yDim <= 0 || zDim <= 0
			|| xmin > xmax || ymin > ymax || zmin > zmax
			|| !bIsUniform || NumComponents <= 0)
		{
			printf("Something went wrong\n");
			fclose(fp);
			return 1;
		}

		const long idxStartData = strstr(buffer, "# Data section follows") - buffer;
		if (idxStartData > 0)
		{
			fseek(fp, idxStartData, SEEK_SET);
			fgets(buffer, 2047, fp);
			fgets(buffer, 2047, fp);

			const size_t NumToRead = xDim * yDim * zDim * NumComponents;
			float* pData = new float[NumToRead];
			if (pData)
			{
				const size_t ActRead = fread((void*)pData, sizeof(float), NumToRead, fp);

				VectorField * field = (VectorField *)_field;
				field->m_data.resize(xDim*yDim*zDim);
				field->m_dimensions(xDim, yDim, zDim);
				field->m_maxVal(xmax, ymax, zmax);
				field->m_minVal(xmin, ymin, zmin);
				std::memcpy(field->m_data.data(), pData, sizeof(Vector3) * xDim*yDim*zDim);

				if (NumToRead != ActRead)
				{
					delete[] pData;
					fclose(fp);
					return 1;
				}
			}
		}

		fclose(fp);


		return 0;
	}

}
