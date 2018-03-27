#
# $Id: Makefile,v 1.3 2005/10/18 14:05:51 ovidiom Exp $
#

CC = g++
CFLAGS = -ansi -pedantic -Wall -O2 -fPIC

HEADERS = \
	glExtensions.h \
	ILLightingModel.h \
	ILRender.h \
	ILTexture.h \
	ILUtilities.h \
	ShaderProgram.h \
	Vector.h
SOURCES = \
	ILRender.cpp \
	ILTexture.cpp \
	ILUtilities.cpp \
	ShaderProgram.cpp \
	Vector.cpp \
	shader_programs/IL_cylinder_blinn_fp.cpp \
	shader_programs/IL_cylinder_blinn_tangent_vp.cpp \
	shader_programs/IL_cylinder_blinn_vp.cpp
OBJECTS = $(SOURCES:.cpp=.o)

LIB_NAME = ILines
LIB_MAJOR = 1
LIB_MINOR = 0


ifneq ($(shell uname | grep -i Cygwin), )
  OS = Cygwin
else # assume Linux
  OS = Linux
endif

ifeq ($(OS), Linux)
  CFLAGS += -DLINUX
  LIB_SHARED_NAME = lib$(LIB_NAME).so
  LIB_SHARED_SONAME = $(LIB_SHARED_NAME).$(LIB_MAJOR)
  LIB_SHARED_REALNAME = $(LIB_SHARED_SONAME).$(LIB_MINOR)
  LIB_STATIC_NAME = lib$(LIB_NAME).a
else # Cygwin
  CFLAGS += -D__CYGWIN__
  LIBS = -lstdc++ -lm -lglu32 -lopengl32
  LIB_SHARED_NAME = lib$(LIB_NAME).dll.a
  LIB_DLL_NAME = cyg$(LIB_NAME).dll
  LIB_STATIC_NAME = lib$(LIB_NAME).a
endif

INSTALL_DIR ?= /usr/local
DLL_DIR ?= /bin


help:
	@echo "User-defined Makefile Variables                                       "
	@echo "===============================                                       "
	@echo "                                                                      "
	@echo "INSTALL_DIR    :  Defaults to /usr/local/.                            "
	@echo "                  Base directory where to install the libraries and   "
	@echo "                  header files. The header files are installed in     "
	@echo "                  $(INSTALL_DIR)/include and the libraries are        "
	@echo "                  installed in $(INSTALL_DIR)/lib.                    "
	@echo "DLL_DIR        :  [Cygwin ONLY] Defaults to /bin/.                    "
	@echo "                  Directory where to install the DLL file.            "
	@echo "                                                                      "
	@echo "                                                                      "
	@echo "Makefile Targets                                                      "
	@echo "================                                                      "
	@echo "                                                                      "
	@echo "all:              Builds the static and shared libraries.             "
	@echo "shared:           Builds the shared library.                          "
	@echo "static:           Builds the static library.                          "
	@echo "install:          Builds and installs the static and shared libraries."
	@echo "install_shared:   Builds and installs the shared library.             "
	@echo "install_static:   Builds and installs the static library.             "
	@echo "install_headers:  Installs the header files.                          "
	@echo "clean:            Clean up the current directory.                     "

.cpp.o:
	$(CC) $(CFLAGS) -c -o $@ $<

all: shared static

shared: shared_$(OS)

shared_Linux: $(OBJECTS)
	gcc -shared -Wl,-soname,$(LIB_SHARED_SONAME) -o $(LIB_SHARED_REALNAME) $(OBJECTS)

shared_Cygwin: $(OBJECTS)
	gcc -shared -o $(LIB_DLL_NAME) \
		-Wl,--out-implib=$(LIB_SHARED_NAME) \
		-Wl,--export-all-symbols \
		-Wl,--enable-auto-import \
		-Wl,--whole-archive $(OBJECTS) \
		-Wl,--no-whole-archive $(LIBS)

static: $(OBJECTS)
	ar rcs $(LIB_STATIC_NAME) $(OBJECTS)

install: install_headers install_shared install_static

install_headers:
	mkdir -p -m 755 $(INSTALL_DIR)/include/$(LIB_NAME)
	install -D -m 644 $(HEADERS) $(INSTALL_DIR)/include/$(LIB_NAME)

install_shared: install_shared_$(OS)

install_shared_Linux: shared_Linux
	install -D -m 644 $(LIB_SHARED_REALNAME) $(INSTALL_DIR)/lib/$(LIB_SHARED_REALNAME)
	( cd $(INSTALL_DIR)/lib ; ln -f -s $(LIB_SHARED_REALNAME) $(LIB_SHARED_SONAME) )
	( cd $(INSTALL_DIR)/lib ; ln -f -s $(LIB_SHARED_SONAME) $(LIB_SHARED_NAME) )

install_shared_Cygwin: shared_Cygwin
	install -D -m 644 $(LIB_SHARED_NAME) $(INSTALL_DIR)/lib/$(LIB_SHARED_NAME)
	install -D -m 644 $(LIB_DLL_NAME) $(DLL_DIR)/$(LIB_DLL_NAME)

install_static: static
	install -D -m 644 $(LIB_STATIC_NAME) $(INSTALL_DIR)/lib/$(LIB_STATIC_NAME)

clean:
	rm -f $(OBJECTS) $(LIB_SHARED_REALNAME) $(LIB_STATIC_NAME) $(LIB_SHARED_NAME) $(LIB_DLL_NAME)

