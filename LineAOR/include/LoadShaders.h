#pragma once
#include<stdio.h>  
#include<stdlib.h>  
#include<string.h>  
#define GLUT_DISABLE_ATEXIT_HACK
#include "glew.h"
#include "glut.h"
#include <freeglut.h>
#include <vector>
#include <string>
#include <xhash>
std::hash<std::string> hashfunc;

struct ShaderInfo
{
	GLuint Shader_Code;
	const char * path;
	int64_t hash_shader = hashfunc("");
};
GLuint Chech_n_load(ShaderInfo *_SI) {
	return 0;
}
GLuint LoadShader(const char * context, GLuint _type)
{
	GLuint shader_code = glCreateShader(_type);
	glShaderSource(shader_code, 1, &context, NULL);
	glCompileShader(shader_code);
    GLint *status = new GLint(-1);
    glGetShaderiv(shader_code, GL_COMPILE_STATUS, status);
  //  printf("Shader %d Compile Status: %d\n%s\n", shader_code, *status,context);
    int maxLength = 65500;
    std::vector<GLchar> errorLog;
    errorLog.resize(maxLength);
    //(maxLength);
    glGetShaderInfoLog(shader_code, maxLength, &maxLength, &errorLog[0]);
    printf("%s", errorLog.data());
    
    delete status;
	return shader_code;
}
char * Load_A_Shader_File(const char * _path)
{
	FILE *fp = 0;
	char * context = NULL;
	int count = 0;
	if (_path != NULL)
	{
		fp = fopen(_path, "r");
		if (fp != NULL)
		{
			fseek(fp, 0, SEEK_END);
			count = ftell(fp);
			rewind(fp);
			if (count>0)
			{
				context = (char*)malloc(sizeof(char)*(count + 1));
				count = fread(context, sizeof(char), count, fp);
				context[count] = '\0';
			}
            for(int i = 0;i<count;i++)
                if(context[i] == '\r')
                    context[i] = ' ';
			fclose(fp);
		}
		if (context == 0)
		{
			context = const_cast<char*>((new std::string("#version 460\nint main(){}\n"))->c_str());
		//	context[0] = 0;
		}
	}
	return context;
}
GLuint LoadShaders(ShaderInfo * _SI) 
{
	GLuint _p = glCreateProgram();
	while (_SI->Shader_Code!=GL_NONE)
	{
		const char * context = Load_A_Shader_File(_SI->path);
		const int64_t hashvalue = hashfunc(context);
		printf("hashvalue: %ld\n", hashvalue);
		if (_SI->hash_shader != hashvalue)
		{
			GLuint _shadercode = LoadShader(context, _SI->Shader_Code);
			glAttachShader(_p, _shadercode);
		}
		_SI++;
	}
	glLinkProgram(_p);
    GLint *status = new GLint(-1);
    glGetProgramiv(_p, GL_LINK_STATUS, status);
    printf("Program %d Link Status: %d\n", _p, *status);
    delete status;
    printf("Error: %d\n", glGetError());

	int maxLength = 65500;
	std::vector<GLchar> errorLog;
	errorLog.resize(maxLength);
	glGetProgramInfoLog(_p, maxLength, &maxLength, &errorLog[0]);
	printf("%s", errorLog.data());
	return _p;
}

