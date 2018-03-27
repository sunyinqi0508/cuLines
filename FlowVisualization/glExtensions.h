#ifndef _GLEXTENSIONS_H_
#define _GLEXTENSIONS_H_


#if defined(WIN32) || defined(__CYGWIN__)
#define WIN32_LEAN_AND_MEAN		1
#include <windows.h>
#include <GL/gl.h>
#define wglXGetProcAddress(e)	wglGetProcAddress((LPCSTR)(e))
#endif

#ifdef LINUX
#include <GL/gl.h>
#define GLX_GLXEXT_PROTOTYPES
#include <GL/glx.h>
#define wglXGetProcAddress(e)	glXGetProcAddressARB((const GLubyte *)(e))
#endif


#ifndef APIENTRY
#define APIENTRY
#endif


typedef ptrdiff_t GLsizeiptr;


/* Function pointer typedef's for accessing the OpenGL extensions. */

// GL_ARB_multitexture
typedef void(APIENTRY * PFNGLACTIVETEXTUREPROC)(GLenum);
typedef void(APIENTRY * PFNGLCLIENTACTIVETEXTUREPROC)(GLenum);

// GL_EXT_multi_draw_arrays
typedef void(APIENTRY * PFNGLMULTIDRAWARRAYSPROC)(GLenum, const GLint *, const GLsizei *, GLsizei);

// GL_ARB_vertex_buffer_object
typedef void(APIENTRY * PFNGLBINDBUFFERPROC)(GLenum, GLuint);
typedef void(APIENTRY * PFNGLDELETEBUFFERSPROC)(GLsizei, const GLuint *);
typedef void(APIENTRY * PFNGLGENBUFFERSPROC)(GLsizei, GLuint *);
typedef void *(APIENTRY * PFNGLMAPBUFFERPROC)(GLenum, GLenum);
typedef GLboolean(APIENTRY * PFNGLUNMAPBUFFERPROC)(GLenum);
typedef void(APIENTRY * PFNGLGETBUFFERPARAMETERIVPROC)(GLenum, GLenum, GLint *);
typedef void(APIENTRY * PFNGLGETBUFFERPOINTERVPROC)(GLenum, GLenum, GLvoid **);
typedef void(APIENTRY * PFNGLBUFFERDATAPROC)(GLenum, GLsizeiptr,
                                             const GLvoid *, GLenum);

// GL_ARB_vertex_program && GL_ARB_fragment_program
typedef void(APIENTRY * PFNGLPROGRAMSTRINGPROC)(GLenum, GLenum, GLsizei,
                                                const GLvoid *); 
typedef void(APIENTRY * PFNGLBINDPROGRAMPROC)(GLenum, GLuint);
typedef void(APIENTRY * PFNGLDELETEPROGRAMSPROC)(GLsizei, const GLuint *);
typedef void(APIENTRY * PFNGLGENPROGRAMSPROC)(GLsizei, GLuint *);


/* Definition of symbolic constants as specified by the extensions. */

// GL_ARB_multitexture
#define GL_TEXTURE0						0x84C0
#define GL_TEXTURE1						0x84C1
#define GL_ACTIVE_TEXTURE				0x84E0
#define GL_CLIENT_ACTIVE_TEXTURE		0x84E1

// GL_EXT_texture3D
#define GL_TEXTURE_3D					0x806F

// GL_ARB_vertex_buffer_object
#define GL_ARRAY_BUFFER					0x8892
#define GL_VERTEX_ARRAY_BUFFER_BINDING	0x8896
#define GL_READ_ONLY					0x88B8
#define GL_BUFFER_MAP_POINTER			0x88BD
#define GL_STATIC_DRAW					0x88E4

// GL_ARB_vertex_program && GL_ARB_fragment_program
#define GL_VERTEX_PROGRAM				0x8620
#define GL_FRAGMENT_PROGRAM				0x8804
#define GL_PROGRAM_FORMAT_ASCII			0x8875


#endif /* _GLEXTENSIONS_H_ */

