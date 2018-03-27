/* $Id: ShaderProgram.h,v 1.8 2005/10/17 10:12:09 ovidiom Exp $ */

/* forward declarations */
namespace ILines { class ShaderProgram; }

#ifndef _SHADERPROGRAM_H_
#define _SHADERPROGRAM_H_

#include <iostream>
#include <string>
#include <fstream>

#include "glExtensions.h"


namespace ILines
{
	class ShaderProgram
	{
	public:
		ShaderProgram(GLenum target = GL_FRAGMENT_PROGRAM);
		~ShaderProgram();

		void bind() const;
		void release() const;

		void allocate();
		void destroy();

		bool load(std::istream &is) const;
		bool load(const std::string &fn) const;
		bool loadProgram(const std::string &programString) const;

		GLuint getProgramID() const;
		void setProgramID(GLuint programID);

	private:
		GLenum	target;
		GLuint	programID;

		static PFNGLBINDPROGRAMPROC		pglBindProgram;
		static PFNGLGENPROGRAMSPROC		pglGenPrograms;
		static PFNGLDELETEPROGRAMSPROC	pglDeletePrograms;
		static PFNGLPROGRAMSTRINGPROC	pglProgramString;

		void getExtensions() const;
	};
}

#endif /* _SHADERPROGRAM_H_ */

