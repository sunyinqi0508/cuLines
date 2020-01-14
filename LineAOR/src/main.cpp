#include "LineAO.h"
#include "Common.h"
#include "CommandParser.h"
#include "png.h"
//#include "Vector3.h"
#include <comdef.h>
#include <ctime>
#include <chrono>
#include <Windows.h>
#include <fstream>
#include <iostream>
#include <cstring>
#include <sstream>
#include <vector>
#include <map>
#include <unordered_map>
#include <thread>
#include <iterator>
#include <mutex>
#include <gdiplus.h>
#include <ShellScalingApi.h>
#include <xhash>
#include "ILLightingModel.h"
#include "ILTexture.h"
#include "LoadShaders.h"
#include "vmath.h"
#include "fileio.h"
#include "../include/CriticalPoint.h"

#define BUFFER_OFFSET(offset) ((GLvoid*) (NULL + offset))
#define ERR(x) printf("\nERROR %d: %d\n", x, glGetError());

constexpr int texDim = 256;
namespace GLParams {
	int  m_width = 1080;
	int m_height = 1080;
	vmath::vec3 cameraposition;
	vmath::vec3 sceneposition;
	vmath::vec3 up;
	vmath::vec4 lightcol = { 1.0f,1.0f,1.0f,1.0f };
	GLfloat lightpos[] = { 0.f,0.f,3.f,1.f };
	vmath::vec4 line_color = { 1.0f,1.0f,1.0f,1.0f };
	vmath::mat4 mv_matrix;
	vmath::mat4 mvp_matrix;
	vmath::mat4 proj_matrix;
	vmath::mat4 rotation_matrix = vmath::rotate(0.f,0.f,0.f);
	vmath::mat4 translate_matrix = vmath::translate(0.f, 0.f, 0.f);
	Vector3 bg_colors[4] = { {1.f,1.f,1.f },{ 1.f,1.f,1.f } ,{ 1.f,1.f,1.f } ,{ 1.f,1.f,1.f } };
	GLfloat line_width= 1.f;
}

namespace GLVars {
	GLuint Phong_Render = 0; //LineAO Renderer;
	GLuint Sphere_Renderer = 0; //Sphere renderer with phong-blinn shading;
	GLuint light_pos;
	GLuint light_color;
	GLuint line_color_index;
	GLuint mvp_loc, mv_loc, p_loc;
	GLuint VAO[3] = { -1 };
	GLuint VBO[3];
	GLuint Texture_Depth[1];
	GLuint Texture_SphereInfo[3];
	GLuint Texture_Spec[1], Texture_Diff[1];

	GLuint texture_depth_loc, texture_sphereinfo_loc,
		texture_spec_loc, texture_diff_loc;

	GLuint sphere_light_pos_loc;
	GLuint sphere_mv_matrix_loc;
	GLuint sphere_mvp_matrix_loc;

	GLuint aoswitch_loc;
	GLuint shader_type_loc;
	GLuint DEPTHFBO[1];
	GLuint cam_loc;
	GLuint size_loc;
	GLuint plane_loc;
	GLuint line_width_index;
	GLint is_rendering_bg_loc;
	ShaderInfo shader_info_Phong[] =
	{
		{ GL_VERTEX_SHADER, "d:/flow_shaders/Phong.vert" },
	{ GL_FRAGMENT_SHADER, "d:/flow_shaders/Phong.frag" },
	{ GL_NONE, NULL }
	};
	ShaderInfo shader_info_Sphere[] = {
		{ GL_VERTEX_SHADER,"d:/flow_shaders/Sphere_Phong.vert" },
	{ GL_FRAGMENT_SHADER, "d:/flow_shaders/Sphere_Phong.frag" },
	{ GL_NONE, NULL }
	};
}

namespace Status {
	int old_x;
	int old_y;
	bool updatePending = false, daemon_running = true, main_thread_running = true;
	bool operation_change_light_dir = false, blend = false;
	float fRotateAngle = 0.0f;
	float angle_x = 0, light_angle_x = 0;
	float angle_y = 0, light_angle_y = 0;
	float light_distance = 3.f;
	float aspect =GLParams::m_width / GLParams::m_height;
	float _scale = 1.f, lastscale = 1.f, thisscale = 0.f;
	float scalex, scaley;
	float transform_x = 0, transform_y = 0;
	bool is_zooming = false, is_rotating = false, is_transforming = false;
	bool ao_switch = true;
	bool bgcolor = true, antialaising = false;
	LSH_Application curr_application = LSH_None;
	int reduced;
	float lsh_radius = 9.948f/172.f;
	bool screenshot = false;
	float plane_pos = 1.f;
	bool fastrender = true;
}


namespace Globals {
	struct SphereInfo {
		Vector3 Colors;
		Vector3 Pos;
		float alpha;
		SphereInfo(Vector3 Colors, Vector3 Pos, float alpha) : Colors(Colors), Pos(Pos), alpha(alpha) {}
	};

	float *texDiff;
	float *texSpec;
	Vector3* tangents = 0;
	float *alpha;
	Vector3 *colors;

	int sphereCount = 0;
	int sphereVertCnt = 0;
	int* sphereOffsets;
	int* sphereSizes;
	SphereInfo* sphereInfo;
	std::string filename = "";
	std::string imagepath = "d:/flow_screenshots/";
	std::unordered_map<std::string, int> commands;
	std::unordered_map<std::string, int> lsh_application;

}

using namespace std;
using namespace vmath;
using namespace FileIO;
using namespace Status;
using namespace GLVars;
using namespace Globals;
using namespace GLParams;



//namespace std {
//	template<>
//	struct hash<StrKey> {
//		size_t operator()(StrKey const& v) const {
//			return (*((hash<string>*)(this))).operator()(v);
//			string a;
//		}
//	};
//}
inline char* str_from_wstr(const WCHAR* _w) {
	int len = lstrlenW(_w);
	char* ret = new char[len + 1];
	for (int i = 0; i < len; i++)
		ret[i] = _w[i];
	ret[len] = 0;
	return ret;
}
bool save_png_libpng(const char *filename, uint8_t *pixels, int w, int h)
{
	
	png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
	if (!png)
		return false;

	png_infop info = png_create_info_struct(png);
	if (!info) {
		png_destroy_write_struct(&png, &info);
		return false;
	}

	FILE *fp = fopen(filename, "wb");
	if (!fp) {
		png_destroy_write_struct(&png, &info);
		return false;
	}
	png_init_io(png, fp);
	png_set_IHDR(png, info, w, h, 8 /* depth */, PNG_COLOR_TYPE_RGBA, PNG_INTERLACE_NONE,
		PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);
	png_colorp palette = (png_colorp)png_malloc(png, PNG_MAX_PALETTE_LENGTH * sizeof(png_color));
	if (!palette) {
		fclose(fp);
		png_destroy_write_struct(&png, &info);
		return false;
	}
	png_set_PLTE(png, info, palette, PNG_MAX_PALETTE_LENGTH);
	png_write_info(png, info);
	png_set_packing(png);

	png_bytepp rows = (png_bytepp)png_malloc(png, h * sizeof(png_bytep));
	for (int i = 0; i < h; ++i)
		rows[i] = (png_bytep)(pixels + (h - i - 1) * w * 4);

	png_write_image(png, rows);
	png_write_end(png, info);
	png_free(png, palette);
	png_destroy_write_struct(&png, &info);

	fclose(fp);
	delete[] rows;
	return true;
}
void MessageCallback(GLenum source,
	GLenum type,
	GLuint id,
	GLenum severity,
	GLsizei length,
	const GLchar* message,
	const void* userParam)
{
	fprintf(stderr, "GL CALLBACK: %s type = 0x%x, severity = 0x%x, message = %s\n",
		(type == GL_DEBUG_TYPE_ERROR ? "** GL ERROR **" : ""),
		type, severity, message);
}

void CommandPattleDaemon() {
	char *curr_command = new char[1048576];
	vector<string>* params = new vector<string>;
	uint64_t curr_cycle = 0;
	printf("Command parser daemon %x started.\n", std::this_thread::get_id());
	printf("\n>");
//	glewInit();
	while (Status::daemon_running) {
		if (curr_cycle % 10 == 0)
		{
			
		}
		cin.getline(curr_command, 1024000);
		//printf_s("%s %d %d\n", curr_command, strnlen_s(curr_command, 1024000), *commands.find(StrKey("fu")));
		int command_len = strnlen_s(curr_command, 1024000);
		while (command_len > 0&& curr_command[command_len-1] == '*') 
			command_len--;

		curr_command[command_len] = 0;
		istringstream iss(curr_command);
		istream_iterator<string> isi(iss);
		string this_command = *(isi++);
		params->clear();
		std::copy(isi, istream_iterator<string>(), back_inserter(*params));
		if (command_len && commands.find((this_command)) != commands.end()) {
			const int command_num = commands[this_command];

			if (command_num > 64)
				switch (command_num) {
				case 65:
					glutExit();
					main_thread_running = false;
					Status::daemon_running = false;
					
					break;
				case 66:
				{
					printf("hello!\n");
					break;
				}

				default:
					;
				}

			else if (command_num < command_callbacks.size())
			{
				command_callbacks[command_num](params);
				continue;
			}
		}
		
		printf("\n>");
		curr_cycle++;
	}
	
	printf("Deiniting\n\n\n");
	delete[] curr_command;
	delete params;
}


void getTangent() {
	if (!tangents)
		delete[]tangents;

	tangents = new Vector3[n_points];
	FileIO::_getTangent(tangents);//new Vector3[n_points];

	//int idx_pt = 0;
	//Vector3 *tmp = new Vector3[Streamline::max_size()];
	//for (size_t i = 0; i <n_streamlines; i++)
	//{
	//	for (size_t j = 0; j < Streamline::size(i); j++)
	//	{
	//		int idx1 = j + 1, idx2 = j - 1;
	//		idx2 = idx2 < 0 ? 0 : idx2;
	//		idx1 = idx1 >= Streamline::size(i)? Streamline::size(i)-1: idx1;
	//		tangents[idx_pt + j] = (f_streamlines[i][idx1] - f_streamlines[i][idx2]).normalized();
	//	}
	//	//for(int k = 0; k <2; k++)
	//	//	for (size_t j = 0; j < Streamline::size(i); j++)
	//	//	{
	//	//		int idx1 = j + 1, idx2 = j - 1;
	//	//		idx2 = idx2 < 0 ? 0 : idx2;
	//	//		idx1 = idx1 >= Streamline::size(i) ? Streamline::size(i) - 1 : idx1;
	//	//		tmp[j] = (tmp[idx1] + tmp[idx2]) / 8.f + 3*tmp[j]/4.f;
	//	//	}
	//	/*for (size_t j = 0; j < Streamline::size(i); j++)
	//	{
	//		int idx1 = j + 1, idx2 = j - 1;
	//		idx2 = idx2 < 0 ? 0 : idx2;
	//		idx1 = idx1 >= Streamline::size(i) ? Streamline::size(i) - 1 : idx1;
	//		tangents[idx_pt + j] = (tmp[idx1] + tmp[idx2]) / 8.f + 3 * tmp[j] / 4.f;
	//	}*/
	//	idx_pt += Streamline::size(i);
	//}
	//delete[] tmp;
}

void InitConig()
{
	cameraposition = vmath::vec3{ 0.f , 0.f, 2.f};//2maxx, 2maxy, maxz + minz
	sceneposition = vmath::vec3{-0.,-0.f,-0.f};//means
	up = vmath::vec3{ 0.0f,1.0f,0.0f };
	line_color = { 0.7f, (float)150.5 / 255, 0.0f, 1.0f };
}
void InitTexture()
{

	glGenTextures(1, Texture_Depth);
	glBindTexture(GL_TEXTURE_2D, Texture_Depth[0]);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, m_width, m_height,
		0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
	//glTexParameteri(GL_TEXTURE_2D, GL_GENERATE_MIPMAP, GL_TRUE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); 
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);// _MIPMAP_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glGenTextures(1, Texture_Diff);
    glBindTexture(GL_TEXTURE_2D, Texture_Diff[0]);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R16, texDim, texDim, 0,
                 GL_RED, GL_FLOAT, texSpec);

	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glGenTextures(1, Texture_Spec);
    glBindTexture(GL_TEXTURE_2D, Texture_Spec[0]);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R16, texDim, texDim, 0, GL_RED, GL_FLOAT, texSpec);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	string a;
	auto b = "" + a;
	printf("12: %d\n", glGetError());
}
void InitFBO()
{
	glGenFramebuffers(1, DEPTHFBO);
	glBindFramebuffer(GL_FRAMEBUFFER, DEPTHFBO[0]);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, Texture_Depth[0], 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}
void Mouse(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN)
	{
		if (button == GLUT_LEFT_BUTTON)
		{
			is_rotating = true;
			old_x = x, old_y = y;
		}
		else if (button == GLUT_RIGHT_BUTTON)
		{
			is_zooming = true;
			lastscale = _scale;
			thisscale = 0.f;
			scalex = x;
			scaley = y;
		}
	}
	else if (state == GLUT_UP)
	{
		if (button == GLUT_LEFT_BUTTON)
			is_rotating = false;
		else if (button == GLUT_RIGHT_BUTTON)
			is_zooming = false;
		//is_transforming = false;
	}
}
void onMouseMove(int x, int y)
{
	//cout << x << ' ' << y << endl;
	if (is_rotating || is_transforming)
	{
		if (is_transforming)
		{

		}
		else if (operation_change_light_dir) {
			light_angle_x -= (x - old_x)/150.f;
			light_angle_y -= (y - old_y)/150.f;
			old_x = x, old_y = y;
		}
		else {
			rotation_matrix=vmath::rotate((float)(y - old_y), (float)(x - old_x), 0.f)*rotation_matrix;
			old_x = x, old_y = y;
		}
	}
	else if (is_zooming) {
		vec2 dir = vec2(scalex - m_width / 2.f, scaley - m_height / 2.f);
		vec2 move = vec2(scalex - x, scaley - y);
		if (dot(dir, move) < 0)
			thisscale = sqrt(dot(move, move)) / 1000.f;
		else
			thisscale = -sqrt(dot(move, move)) / 1000.f;
		_scale = lastscale + thisscale;
	}
	else
		return;
	glutPostRedisplay();
}
void InitShader()
{

	if (Phong_Render > 0)
		glDeleteProgram(Phong_Render);
	if (Sphere_Renderer > 0)
		glDeleteProgram(Sphere_Renderer);


	Phong_Render = LoadShaders(shader_info_Phong);
	glUseProgram(Phong_Render);
	light_pos = glGetUniformLocation(Phong_Render, "lightpos");
	light_color = glGetUniformLocation(Phong_Render, "lightcolor");
	line_color_index = glGetUniformLocation(Phong_Render, "linecolor");
	mvp_loc = glGetUniformLocation(Phong_Render, "mvp_matrix");
	mv_loc = glGetUniformLocation(Phong_Render, "mv_matrix");
	p_loc = glGetUniformLocation(Phong_Render, "proj_matrix");
	plane_loc = glGetUniformLocation(Phong_Render, "z_filter");

	texture_depth_loc = glGetUniformLocation(Phong_Render,"Depth");
	aoswitch_loc = glGetUniformLocation(Phong_Render, "enable_AO");
	shader_type_loc = glGetUniformLocation(Phong_Render,"shader_type");
	line_width_index = glGetUniformLocation(Phong_Render, "line_width");
	size_loc = glGetUniformLocation(Phong_Render, "window_size");
	is_rendering_bg_loc = glGetUniformLocation(Phong_Render, "is_rendering_background");

	glUniform1f(line_width_index, line_width);
	glUniform1i(aoswitch_loc, 1);
	glUniform2f(size_loc, m_width, m_height);
	glUniform1f(plane_loc, plane_pos);

	glUniform4f(light_color, lightcol[0], lightcol[1], lightcol[2], lightcol[3]);
	glUniform4f(line_color_index, line_color[0], line_color[1], line_color[2], line_color[3]);
	glLineWidth(line_width);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, Texture_Depth[0]);
	glUniform1i(texture_depth_loc, 0);


	Sphere_Renderer = LoadShaders(shader_info_Sphere);
	glUseProgram(Sphere_Renderer);
	sphere_light_pos_loc = glGetUniformLocation(Sphere_Renderer, "lightpos");
	sphere_mv_matrix_loc = glGetUniformLocation(Sphere_Renderer, "mv_matrix");
	sphere_mvp_matrix_loc = glGetUniformLocation(Sphere_Renderer, "mvp_matrix");
	texture_sphereinfo_loc = glGetUniformLocation(Sphere_Renderer, "texSphereInfo");

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_1D, Texture_SphereInfo[0]);
	glUniform1i(texture_sphereinfo_loc, 1);


}
enum InitType{ InitFromDll, InitFromObj, InitFromBSL, InitCpData, InitNULL };

void InitData(const InitType type) {

	FileIO::reinitData();
	switch (type) {
	case InitFromDll:
	{
		HINSTANCE similarity_dll = LoadLibrary(L"UIBridge.dll");

		void(_stdcall *transfer)(Communicator*) =
			(void(*)(Communicator*)) GetProcAddress(similarity_dll, "transfer");

		printf("Proc %x in Module %x Loaded with error code: %d\n", transfer, similarity_dll, GetLastError());

		Communicator *comm = new Communicator();
		/* Setting up paras*/
		//comm->filename = "d:/flow_results/5adaa397.bsl";
		comm->filename = filename.c_str();
		comm->n_streamlines = reduced;
		comm->application = curr_application;
		comm->lsh_radius = lsh_radius;

		transfer(comm);
		printf("Transferred pointer: %x\n", comm->f_streamlines);

		if (comm->AdditionalParas)
			Streamline::initFromStreamlineData(
				reinterpret_cast<Streamline::Streamline_data*>(comm->AdditionalParas)
			);
		else {
			f_streamlines = (Vector3**)comm->f_streamlines;
			n_streamlines = comm->n_streamlines;
			n_points = comm->n_points;
			//Streamline::sizes = comm->sizes;
			Streamline::reinit();
		}
		if (comm->alpha == 0) {
			alpha = new float[n_points];
			std::fill(alpha, alpha + n_points, 1.f);
		}
		else
			alpha = comm->alpha;
		if (comm->colors == 0) {
			colors = new Vector3[n_points];
			for (int i = 0; i < n_points; i++)
				colors[i] = Vector3(0x49 / 255.f, 0xb9 / 255.f, 0xf9 / 255.f);
		}
		else
		{
			colors = new Vector3[n_points];
			for (int i = 0; i < n_points; i++)
			{
				int this_color = comm->colors[i];
				colors[i].x = ((this_color >> 24) & 0xff) / 255.f;
				colors[i].y = ((this_color >> 16) & 0xff) / 255.f;
				colors[i].z = ((this_color >> 8) & 0xff) / 255.f;
			}
		}
	}
		break;
	case InitFromObj:
		FileIO::LoadWaveFrontObject(filename.c_str());
		FileIO::toFStreamlines();
		FileIO::normalize(1.f, true, Format::STREAMLINE_ARRAY, 0, true);
		alpha = new float[n_points];
		std::fill(alpha, alpha + n_points, 1.f);
		colors = new Vector3[n_points];
		for (int i = 0; i < n_points; i++)
			colors[i] = Vector3(0x49 / 255.f, 0xb9 / 255.f, 0xf9 / 255.f);
		break;
	case InitFromBSL:
		FileIO::ReadBSL(filename.c_str());
		FileIO::normalize(1.f, false, Format::STREAMLINE_ARRAY, 0, true);
		alpha = new float[n_points];
		std::fill(alpha, alpha + n_points, 1.f);
		colors = new Vector3[n_points];
		for (int i = 0; i < n_points; i++)
			colors[i] = Vector3(0x49 / 255.f, 0xb9 / 255.f, 0xf9 / 255.f);
		break;
		break;
	}
	getTangent();

	glGenBuffers(1, VBO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO[0]);
	glBufferData(GL_ARRAY_BUFFER, 10 * n_points * sizeof(float), NULL, GL_STATIC_DRAW);
	if (n_points > 0)
	{
		glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(Vector3)*n_points, f_streamlines[0]);
		glBufferSubData(GL_ARRAY_BUFFER, sizeof(Vector3)*n_points, sizeof(Vector3)*n_points, tangents);
		glBufferSubData(GL_ARRAY_BUFFER, sizeof(Vector3)*n_points + sizeof(Vector3)*n_points, sizeof(float)*n_points, alpha);
		glBufferSubData(GL_ARRAY_BUFFER, sizeof(Vector3)*n_points + sizeof(Vector3)*n_points + sizeof(float) * n_points, sizeof(Vector3)*n_points, colors);
	}
	glGenVertexArrays(1, &VAO[0]);
}
void InitSphere(SphereType *sp, int length) {
	std::vector<Vector3> sphereCoords;
	std::vector< int> *spherePtIDs = new vector< int>;
	GLfloat x, y, z;
	Globals::sphereInfo = (SphereInfo*) malloc (sizeof(SphereInfo) * length);
	Globals::sphereOffsets = new int[length];
	Globals::sphereSizes = new int[length];
	Globals::sphereCount = length;
	for (int i = 0; i < length; i++)
	{

		GLfloat alpha, beta, div = 16.f;
		const GLfloat radius = sp[i].radius;
		sphereOffsets[i] = (sphereCoords.size());
		for (alpha = 0.f; alpha < M_PI; alpha += M_PI / div)
		{
			for (beta = 0; beta < 2.01*M_PI; beta += M_PI / div)
			{
				x = radius * cos(beta)*sin(alpha);
				y = radius * sin(beta)*sin(alpha);
				z = radius * cos(alpha);
				sphereCoords.emplace_back(x, y, z);
				sphereCoords.back() += sp[i].origin;
				spherePtIDs->emplace_back(i);
				x = radius * cos(beta)*sin(alpha + M_PI / div);
				y = radius * sin(beta)*sin(alpha + M_PI / div);
				z = radius * cos(alpha + M_PI / div);
				sphereCoords.emplace_back(x, y, z);
				sphereCoords.back() += sp[i].origin;
				spherePtIDs->emplace_back(i);
			}
		}
		float sp_alpha = 1.f;
		sphereSizes[i] = (sphereCoords.size() - sphereOffsets[i]);
		sphereInfo[i] = std::move(SphereInfo(fColor::fromIColors(sp[i].color, &sp_alpha), sp[i].origin, sp_alpha));
	}

	sphereVertCnt = sphereCoords.size();
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	glGenTextures(1, Texture_SphereInfo);

	glBindTexture(GL_TEXTURE_1D, Texture_SphereInfo[0]);
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_BASE_LEVEL, 0);
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAX_LEVEL, 0);
	glTexImage1D(GL_TEXTURE_1D, 0, GL_R16F, (3+3+1)*length, 0, GL_RED, GL_FLOAT,  (void*)sphereInfo);

	glGenBuffers(1, VBO + 1);
	glBindBuffer(GL_ARRAY_BUFFER, VBO[1]);
	glBufferData(GL_ARRAY_BUFFER, (sizeof(Vector3) + sizeof(int))*sphereCoords.size(), 0, GL_STATIC_DRAW);
	glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(Vector3)*sphereCoords.size(), sphereCoords.data());
	glBufferSubData(GL_ARRAY_BUFFER, sizeof(Vector3) * sphereCoords.size(), spherePtIDs->size()*sizeof(int), spherePtIDs->data());


	glGenVertexArrays(1, VAO + 1);

}
void InitBackground() {
	Vector3 bg_vertices[] = {
		{-40.f,40.f,-20.f},
		{ -40.f,-40.f,-20.f },
		{ 40.f,40.f,-20.f },
		{ 40.f,-40.f,-20.f }
	};
	Vector3 bg_tangents[4] = { {0.f,0.f,1.f},{ 0.f,0.f,1.f },{ 0.f,0.f,1.f },{ 0.f,0.f,1.f } };// normal
	float bg_alphas[4] = { 1.f ,1.f,1.f,1.f };
	
	glGenBuffers(1, VBO + 2);
	glBindBuffer(GL_ARRAY_BUFFER, VBO[2]);
	glBufferData(GL_ARRAY_BUFFER, (sizeof(Vector3)*3 + sizeof(float))*4, 0, GL_STATIC_DRAW);
	glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(Vector3)*4, bg_vertices);
	glBufferSubData(GL_ARRAY_BUFFER, sizeof(Vector3)*4, sizeof(Vector3)*4, bg_tangents);
	glBufferSubData(GL_ARRAY_BUFFER, sizeof(Vector3)*4 + sizeof(Vector3)*4, sizeof(float)*4, bg_alphas);
	glBufferSubData(GL_ARRAY_BUFFER, sizeof(Vector3)*4 + sizeof(Vector3)*4 + sizeof(float) * 4, sizeof(Vector3)*4, bg_colors);

	glGenVertexArrays(1, VAO + 2);
}
void InitVAO() {

	if (VAO[1] > 0)
	{
		glUseProgram(Sphere_Renderer);
		glBindVertexArray(VAO[1]);
		glBindBuffer(GL_ARRAY_BUFFER, VBO[1]);
		GLuint svapos = glGetAttribLocation(Sphere_Renderer, "in_pos");
		GLuint svaid = glGetAttribLocation(Sphere_Renderer, "in_id");

		glVertexAttribPointer(svapos, 3, GL_FLOAT, GL_FALSE, sizeof(Vector3), BUFFER_OFFSET(0));
		glVertexAttribIPointer(svaid, 1, GL_INT, sizeof(int), (void*)(sizeof(Vector3)* sphereVertCnt));

		glEnableVertexAttribArray(svapos);
		glEnableVertexAttribArray(svaid);
	}

	glUseProgram(Phong_Render);
	glBindVertexArray(VAO[2]);
	glBindBuffer(GL_ARRAY_BUFFER, VBO[2]);
	GLuint vapos = glGetAttribLocation(Phong_Render, "position");
	GLuint vatan = glGetAttribLocation(Phong_Render, "in_tangent");
	GLuint vaalp = glGetAttribLocation(Phong_Render, "in_alpha");
	GLuint vacol = glGetAttribLocation(Phong_Render, "in_color");
	glVertexAttribPointer(vapos, 3, GL_FLOAT, GL_FALSE, sizeof(Vector3), BUFFER_OFFSET(0));
	glVertexAttribPointer(vatan, 3, GL_FLOAT, GL_FALSE, sizeof(Vector3), BUFFER_OFFSET(sizeof(Vector3)*4));
	glVertexAttribPointer(vaalp, 1, GL_FLOAT, GL_FALSE, sizeof(float), BUFFER_OFFSET(2 * sizeof(Vector3)*4));
	glVertexAttribPointer(vacol, 3, GL_FLOAT, GL_TRUE, sizeof(Vector3), BUFFER_OFFSET((2 * sizeof(Vector3) + sizeof(float))*4));

	glEnableVertexAttribArray(vapos);
	glEnableVertexAttribArray(vatan);
	glEnableVertexAttribArray(vaalp);
	glEnableVertexAttribArray(vacol);


	glBindVertexArray(VAO[0]);
	glBindBuffer(GL_ARRAY_BUFFER, VBO[0]);
	glVertexAttribPointer(vapos, 3, GL_FLOAT, GL_FALSE, sizeof(Vector3), BUFFER_OFFSET(0));
	glVertexAttribPointer(vatan, 3, GL_FLOAT, GL_FALSE, sizeof(Vector3), BUFFER_OFFSET(sizeof(Vector3)*n_points));
	glVertexAttribPointer(vaalp, 1, GL_FLOAT, GL_FALSE, sizeof(float), BUFFER_OFFSET(2 * sizeof(Vector3)*n_points));
	glVertexAttribPointer(vacol, 3, GL_FLOAT, GL_TRUE, sizeof(Vector3), BUFFER_OFFSET((2 * sizeof(Vector3) + sizeof(float))*n_points));

	glEnableVertexAttribArray(vapos);
	glEnableVertexAttribArray(vatan);
	glEnableVertexAttribArray(vaalp);
	glEnableVertexAttribArray(vacol);

}

std::vector<SphereType> spheres;
NormalizeParameter normalize_param;

void loadSelectionResult(const char *filename) {
	LoadWaveFrontObject(filename);
	normalize_param = normalize(.5f, true, STREAMLINE_VECTOR, 0, true);
	toFStreamlines();
	alpha = new float[n_points];
	std::fill(alpha, alpha + n_points, .8f);
	colors = new Vector3[n_points];
	for (int i = 0; i < n_points; i++)
		colors[i] = Vector3(0xff / 255.f, 0x22 / 255.f, 0x1e / 255.f);
}

inline unsigned int makeFilter(CriticalPointType cp_type) {
	return 1 << static_cast<unsigned int>(cp_type);
}

void loadCriticalPointsAsSpheres(const char *filename, unsigned int filter = 0b11111111) {
	std::unordered_map<CriticalPointType, std::size_t> cp_type_counter;
	auto points = loadCriticalPoints(filename);
	for (const auto &pt : points)
		cp_type_counter[pt.type]++;
	std::cout << "Critical Point Type Statistics\n";
	for (const auto &kv : cp_type_counter)
		std::cout << '\t' << getCriticalPointTypeName(kv.first) << '\t' << kv.second << '\n';
	std::cout << points.size() << " critical points loaded.\n";
	for (const auto &pt : points) {
		if (filter & makeFilter(pt.type) == 0)
			continue;
		auto position = (pt - normalize_param.offset) / normalize_param.multiplier;
		spheres.emplace_back(position, .015f, 0xffff05ff);
	}
	InitSphere(spheres.data(), static_cast<int>(spheres.size()));
}
void InitRenderer() {
	tangents = new Vector3[n_points];
	getTangent();


	InitConig();

	InitBackground();

	InitTexture();

	InitFBO();

	InitShader();

	//loadCriticalPointsAsSpheres(cp_filename, makeFilter(CriticalPointType::RepelFocus));

	InitVAO();
}
void Init()
{
	// change input filename here
	//auto model_filename = "d:/flow_data/cp/5cp.result.obj";
	//auto cp_filename = "d:/flow_data/cp/5cp.cp";

	//loadSelectionResult(model_filename);
	//glGenBuffers(1, VBO);
	//glBindBuffer(GL_ARRAY_BUFFER, VBO[0]);
	//glBufferData(GL_ARRAY_BUFFER, 10 * n_points * sizeof(float), NULL, GL_DYNAMIC_DRAW);

	//glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(Vector3)*n_points, f_streamlines[0]);
	//glBufferSubData(GL_ARRAY_BUFFER, sizeof(Vector3)*n_points, sizeof(Vector3)*n_points, tangents);
	//glBufferSubData(GL_ARRAY_BUFFER, sizeof(Vector3)*n_points + sizeof(Vector3)*n_points, sizeof(float)*n_points, alpha);
	//glBufferSubData(GL_ARRAY_BUFFER, sizeof(Vector3)*n_points + sizeof(Vector3)*n_points + sizeof(float) * n_points, sizeof(Vector3)*n_points, colors);

	//glGenVertexArrays(1, &VAO[0]);

	//InitData(InitType::InitFromDll);
	//InitData(InitType::InitFromObj);
	int _dot_pos = filename.find_last_of('.');
	string extension = filename.substr(_dot_pos>=0? _dot_pos: 0);
	if (curr_application == LSH_None)
	{
		if (extension == ".bsl")
			InitData(InitType::InitFromBSL);
		else if (extension == ".obj")
			InitData(InitType::InitFromObj);
		else
			InitData(InitType::InitNULL);
	}
	else if(curr_application == LSH_Contraction||LSH_Alpha) {
		InitData(InitType::InitFromDll);
	}
	else InitData(InitType::InitNULL); //

	InitRenderer();

	lsh_application["contraction"] = LSH_Application::LSH_Contraction;
	lsh_application["contract"] = LSH_Application::LSH_Contraction;
	lsh_application["con"] = LSH_Application::LSH_Contraction;
	lsh_application["c"] = LSH_Application::LSH_Contraction;

	lsh_application["alpha"] = LSH_Application::LSH_Alpha;
	lsh_application["transparency"] = LSH_Application::LSH_Alpha;
	lsh_application["alp"] = LSH_Application::LSH_Alpha;
	lsh_application["a"] = LSH_Application::LSH_Alpha;


}

void deinit() {
	delete[] tangents;
	delete[] colors;
	/*delete[] texDiff;
	delete[] texSpec;*/
	delete[] alpha;
}
void Render_Object(int type)
{
	glUseProgram(Phong_Render);
	glUniform1i(shader_type_loc, type);
	glUniformMatrix4fv(mvp_loc, 1, GL_FALSE, mvp_matrix);
	glUniformMatrix4fv(mv_loc, 1, GL_FALSE, mv_matrix);
	glUniformMatrix4fv(p_loc, 1, GL_FALSE, proj_matrix);
	glBindVertexArray(VAO[2]);
	glUniform1i(is_rendering_bg_loc, 1);
	glBindBuffer(GL_ARRAY_BUFFER, VBO[2]);
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
	glUniform1i(is_rendering_bg_loc, 0);
	glBindVertexArray(VAO[0]);
	glBindBuffer(GL_ARRAY_BUFFER, VBO[0]);
	glMultiDrawArrays(GL_LINE_STRIP, Streamline::offsets, Streamline::sizes, n_streamlines);
}

LARGE_INTEGER freq;
void display() 
{
	//LARGE_INTEGER begin, end;
	//QueryPerformanceCounter(&begin);


	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	proj_matrix = vmath::mat4(vmath::frustum(-1.f, 1.f, -1, 1, 1.f, 50000.0f));
	mv_matrix = vmath::lookat(cameraposition, sceneposition, up)*(translate_matrix)*vmath::scale(_scale > 1 ? _scale : (1.f / (2.f - _scale)))*rotation_matrix;
	mvp_matrix = proj_matrix * mv_matrix;
	lightpos[0] = light_distance*cos(light_angle_x) * sin(light_angle_y);
	lightpos[1] = light_distance * sin(light_angle_x) * sin(light_angle_y);
	lightpos[2] = light_distance * cos(light_angle_y);
	glUseProgram(Phong_Render);
	glUniform3f(light_pos, lightpos[0], lightpos[1], lightpos[2]);
	if (fastrender) {
		Render_Object(1);
	}
	else {
		glBindFramebuffer(GL_FRAMEBUFFER, DEPTHFBO[0]);
		glDepthMask(GL_TRUE);
		glEnable(GL_DEPTH_TEST);
		glDepthFunc(GL_LESS);
		glClear(GL_DEPTH_BUFFER_BIT);
		Render_Object(1);
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		glDepthMask(GL_TRUE);
		glEnable(GL_DEPTH_TEST);
		glDepthFunc(GL_LESS);
		Render_Object(0);

		if (VAO[1] > 0)
		{
			glUseProgram(Sphere_Renderer);
			glUniform3f(sphere_light_pos_loc, lightpos[0], lightpos[1], lightpos[2]);
			glUniformMatrix4fv(sphere_mvp_matrix_loc, 1, GL_FALSE, mvp_matrix);
			glUniformMatrix4fv(sphere_mv_matrix_loc, 1, GL_FALSE, mv_matrix);
			glBindVertexArray(VAO[1]);
			glMultiDrawArrays(GL_TRIANGLE_STRIP, Globals::sphereOffsets, sphereSizes, sphereCount);
		}
	}
	//if (operation_change_light_dir)
	//{
	//	glUseProgram(Sphere_Renderer);
	//	glBindVertexArray(0);
	//	//glDisable(GL_TEXTURE_2D);
	//	Vector3 data[2] = {
	//		{ 0,0,0 },{ Vector3(lightpos[0], lightpos[1],lightpos[2]) }
	//	}; 
	//	GLuint this_vbo, this_vao;
	//	glGenBuffers(1, &this_vbo);
	//	glBindBuffer(GL_ARRAY_BUFFER, this_vbo);
	//	glBufferData(GL_ARRAY_BUFFER, (sizeof(Vector3)) * 2, 0, GL_STATIC_DRAW);
	//	glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(Vector3) * 2, data);
	//	glGenVertexArrays(1, &this_vao);
	//	glBindVertexArray(this_vao);
	//	glBindBuffer(GL_ARRAY_BUFFER, this_vbo);

	//	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vector3), BUFFER_OFFSET(0));
	//	glEnableVertexAttribArray(0);

	//	glDrawArrays(GL_LINES, 0, 2);

	//}
	//glMapBuffer(GL_FRAMEBUFFER, GL_READ_ONLY);
	if (screenshot) {
		uint8_t *pixels = new uint8_t[m_width * m_height * 4];

		glReadPixels(0, 0, m_width, m_height, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
		auto now = std::chrono::system_clock::now();
		auto tt = chrono::system_clock::to_time_t(now);
		char *buf = new char[40];
		sprintf_s(buf, 39, "%x", tt);
		string datafilename = filename.substr(filename.find_last_of('/') + 1);
		datafilename = imagepath + datafilename + '_' + buf + ".png";
		save_png_libpng(datafilename.c_str(), pixels, m_width, m_height);
		delete[]buf;
		screenshot = !screenshot;
		delete[] pixels;
	}


	glutSwapBuffers();
	//QueryPerformanceCounter(&end);
	//printf("%f fps ", freq.QuadPart/(double)(end.QuadPart - begin.QuadPart));
}
void myReshape(int w, int h)
{
	m_width = w, m_height = h;
	const long max_len = __macro_max(w, h);
	const long compensation = (__macro_min(w, h) - max_len)/2;
	bool wide = true;
	if (w < h)
		wide = false;
	else
		wide = true;

	glBindTexture(GL_TEXTURE_2D, Texture_Depth[0]);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, m_width, m_height,
		0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);// _MIPMAP_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	aspect = GLParams::m_width / GLParams::m_height;
	glUseProgram(Phong_Render);

	glUniform2f(size_loc, m_width, m_height);
	if(wide)
		glViewport(0, compensation, max_len, max_len);
	else
		glViewport(compensation, 0, max_len, max_len);
}
void Idle()
{
	std::this_thread::sleep_for(std::chrono::milliseconds::duration(1));

	if (message_mutex.try_lock())
	{
		for (const int& evid : message_queue) {
			command_callbacks[evid](0);
		}
		message_queue.clear();
		message_mutex.unlock();
	}
}
void processNormalKeys(unsigned char key, int x, int y)
{
	if (key == 27)
	{
		//Status::deamon_running = false;
		//Globals::command_parser_daemon.join();
		if (MessageBox(NULL, L"Sure to quit?", L"LineAO Test", MB_YESNO) == IDYES) {
			glutExit();
		}
		//exit(0);
	}
	//else if (key == 'Q' || key == 'q') {
	//	GLfloat radius = vmath::distance(cameraposition, sceneposition);
	//	fRotateAngle += 0.5;

	//	GLfloat camX = sin(fRotateAngle) * radius + sceneposition[0];
	//	GLfloat camZ = cos(fRotateAngle) * radius + sceneposition[2];

	//	cameraposition = vmath::vec3(camX, 0.0, camZ);
	//}
	//else if (key == 'E' || key == 'e') {
	//	GLfloat radius = vmath::distance(cameraposition, sceneposition);
	//	fRotateAngle -= 0.5;

	//	GLfloat camX = sin(fRotateAngle) * radius + sceneposition[0];
	//	GLfloat camZ = cos(fRotateAngle) * radius + sceneposition[2];

	//	cameraposition = vmath::vec3(camX, 0.0, camZ);
	//	printf("\ncam: %f, %f, %f\n", cameraposition[0], cameraposition[1], cameraposition[2]);
	//}
	//else if (key == 'X' || key == 'x') {
	//	printf("\nreset: %f %f %f %f %f %f %f %f %f\n", cameraposition[0], cameraposition[1], cameraposition[2], sceneposition[0], sceneposition[1], sceneposition[2], up[0], up[1], up[2]);
	//	cameraposition = vmath::vec3(2.f, 2.f, 1.f);
	//	sceneposition = vmath::vec3(0.f, 0.f, 0.f);
	//	up = vmath::vec3(0.0f, 0.0f, 1.0f);
	//}

	//else if (key == 't' || key == 'T')
	//{
	//	is_transforming = !is_transforming;
	//}
	else if (key == 's' || key == 'S') {
		if (antialaising)
		{
			glDisable(GL_LINE_SMOOTH);
			glDisable(GL_POLYGON_SMOOTH);
			glDisable(GL_MULTISAMPLE);
			glDisable(GL_BLEND);
		}
		else
		{
			glEnable(GL_MULTISAMPLE);
			if (key == 'S') {

				glEnable(GL_LINE_SMOOTH);
				glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
				glEnable(GL_POLYGON_SMOOTH);
				glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);
			}
		}
		antialaising = !antialaising;
	}
	else if (key == 'c' || key == 'C')
	{
		if (bgcolor)
			glClearColor(0, 0, 0, 1);
		else
			glClearColor(1, 1, 1, 1);
		bgcolor = !bgcolor;
	}
	else if (key == 'w') {
		FILE *fp;
		fopen_s(&fp, "d:/flow_config/ao_config.cfg", "w");
		fprintf_s(fp, "%f\n", _scale);
		for (int i = 0; i < 4; i++)
			for (int j = 0; j < 4; j++)
				fprintf_s(fp, "%f ", rotation_matrix[i][j]);
		for (int i = 0; i < 4; i++)
			for (int j = 0; j < 4; j++)
				fprintf_s(fp, "%f ", translate_matrix[i][j]);
		
		fprintf_s(fp, "%f %f ", light_angle_x, light_angle_y);

		fclose(fp);
	}
	else if (key == 'W') {
		auto now = std::chrono::system_clock::now();
		auto tt = chrono::system_clock::to_time_t(now);
		string filename = "d:/flow_results/";
		char *buf = new char[40];
		sprintf_s(buf, 39, "%x", tt);
		buf[8] = 0;
		filename = filename +buf + ".bsl";
		FileIO::OutputBSL(filename.c_str(), Format::STREAMLINE_ARRAY);
		delete[] buf;
	}
	else if (key == 'R'||key == 'r')
	{
		FILE *fp;
		fopen_s(&fp, "d:/flow_config/ao_config.cfg", "r");
		if (fp)
		{
			fscanf_s(fp, "%f\n", &_scale);
			for (int i = 0; i < 4; i++)
				for (int j = 0; j < 4; j++)
					fscanf_s(fp, "%f", &rotation_matrix[i][j]);
			for (int i = 0; i < 4; i++)
				for (int j = 0; j < 4; j++)
					fscanf_s(fp, "%f", &translate_matrix[i][j]);
			fprintf_s(fp, "%f,%f", &light_angle_x, &light_angle_y);


			fclose(fp);
		}
		else
			return;
	}
	else if (key == 'a') {
		ao_switch = !ao_switch;
		glUseProgram(Phong_Render);
		glUniform1i(aoswitch_loc, (GLuint)ao_switch);
	}
	else if (key == 'l'||key == 'L') {
		operation_change_light_dir = !operation_change_light_dir;
	}
	else if (key == '-')
	{
		translate_matrix = vmath::translate(0.f, 0.f, -0.05f)*translate_matrix;

		/*if (plane_pos <-2)
			plane_pos = -2;
*/
	}
	else if (key == '+')
	{
		translate_matrix = vmath::translate(0.f,0.f,0.05f)*translate_matrix;
	}
	else if (key == 'b')
	{
		if (blend)
			glDisable(GL_BLEND);
		else {
			glEnable(GL_BLEND);
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		}
		blend = !blend;
	}
	else
		return;
	glutPostRedisplay();
}
void processSpecialKeys(int key, int x, int y){
	// 物体到相机的单位向量
	vmath::vec3 direction = vmath::normalize(sceneposition - cameraposition);
	// 物体到相机的单位向量 与 相机的向上向量相乘,得到垂直向量,即平移向量
	vmath::vec3 vertical = vmath::normalize(vmath::cross(direction, up));
	switch(key){
	//case GLUT_KEY_LEFT:
	//	vertical *= 0.5;
	//	cameraposition += vertical;  // 移动相机位置
	//	sceneposition += vertical; //相机的指向位置也一起平移(不平移则以原来的目标转圈)
	//	break;
	//case GLUT_KEY_RIGHT:
	//	vertical *= 0.5;
	//	cameraposition -= vertical;  // 移动相机位置
	//	sceneposition -= vertical; //相机的指向位置也一起平移(不平移则以原来的目标转圈)
	//	break;
	//case GLUT_KEY_UP:
	//	direction *= 0.5;   // 移动0.2个单位向量
	//	cameraposition += direction;
	//	break;
	//case GLUT_KEY_DOWN:
	//	direction *= 0.5;
	//	cameraposition -= direction;
	//	break;
	case GLUT_KEY_F11:
		glutFullScreenToggle();
		break;
	case GLUT_KEY_F10:
		screenshot = true;
		break;
	case GLUT_KEY_UP:
		translate_matrix = vmath::translate(.0f, 0.01f, 0.f) * translate_matrix;
		//rotation_matrix = vmath::rotate(-1.f, 0.f, 0.f) * rotation_matrix;
		break;
	case GLUT_KEY_DOWN:
		translate_matrix = vmath::translate(.0f, -0.01f, 0.f) * translate_matrix;
		//rotation_matrix = vmath::rotate(1.f, 0.f, 0.f) * rotation_matrix;
		break;
	case GLUT_KEY_LEFT:
		translate_matrix = vmath::translate(-0.01f, 0.f, 0.f) * translate_matrix;
		//rotation_matrix = vmath::rotate(0.f, 1.f, 0.f) * rotation_matrix;
		break;
	case GLUT_KEY_RIGHT:
		translate_matrix = vmath::translate(0.01f, 0.0f, 0.f) * translate_matrix;
		//rotion_matrix = vmath::rotate(1.f, -1.f, 0.f) * rotation_matrix;
		break;

	default:
		return;
	}
	glutPostRedisplay();
}
int main(int argc, char ** argv)
{
//	SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_SYSTEM_AWARE);
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA|GLUT_MULTISAMPLE);
	glutInitWindowSize(m_width, m_height);
	glutInitContextVersion(4, 6);
//	glutInitContextFlags(GLUT_FORWARD_COMPATIBLE);
	glutCreateWindow("LineAO Test");

    glewExperimental = GL_TRUE;
	if (glewInit())
	{
		cerr << "Unable to initialize GLEW ... exiting" << endl;
		exit(EXIT_FAILURE);
	}

    
    
    ILines::ILTexture texture;

    texDiff = new float[texDim * texDim];
    texSpec = new float[texDim * texDim];
    texture.computeTextures(0.05, 0.8, 1.0, 10.f*4.f, texDim, NULL, texDiff, texSpec, ILines::ILLightingModel::IL_CYLINDER_BLINN);
    
	Init();
	glEnable(GL_DEBUG_OUTPUT);
	glDebugMessageCallback((GLDEBUGPROC)MessageCallback, 0);
	QueryPerformanceFrequency(&freq);



	
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	GLfloat lineWidthRange[2] = { 0.0f, 0.0f };
	glGetFloatv(GL_ALIASED_LINE_WIDTH_RANGE, lineWidthRange);
	cout << lineWidthRange[0] << ' ' << lineWidthRange[1] << endl;

    glDisable(GL_BLEND);
 //   glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
 //   glEnable(GL_LINE_SMOOTH);
 //   glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
	//glEnable(GL_POLYGON_SMOOTH);
	//glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);
 //   glEnable(GL_MULTISAMPLE);
	//glHint(GL_MULTISAMPLE_FILTER_HINT_NV, GL_NICEST);
	GLint  iMultiSample = 0;
	GLint  iNumSamples = 0;
	glGetIntegerv(GL_SAMPLE_BUFFERS, &iMultiSample);
	glGetIntegerv(GL_SAMPLES, &iNumSamples);
	printf("Multisample support: %d %d \n", iMultiSample, iNumSamples);

	glutMouseFunc(Mouse);
	glutMotionFunc(onMouseMove);
	glutKeyboardFunc(processNormalKeys);
	glutSpecialFunc(processSpecialKeys);
	glutDisplayFunc(display);
	glutReshapeFunc(myReshape);
	glutIdleFunc(Idle);

	command_parser_daemon = std::thread(CommandPattleDaemon);
	commands["exit"] = 65;
	commands["sayhello"] = 66;

	commands["fuck"] = command_callbacks.size();
	command_callbacks.emplace_back([](vector<string> *params) {

		printf("fuck\n");
		for (const string& this_param : *params) {
			printf("%s\n", this_param.c_str());
		}

	});
	commands["reload"] = command_callbacks.size();
	command_callbacks.emplace_back([](vector<string> *params) {
		const int eventid = 1;
		if (std::this_thread::get_id() != main_thread_id) {
			message_mutex.lock();
			message_queue.emplace_back(eventid);
			message_mutex.unlock();

		}
		else
		{
			InitShader();
			InitVAO();
			glutPostRedisplay();
			printf("\n>");

		}
	});

	commands["data"] = command_callbacks.size();
	command_callbacks.emplace_back([](vector<string> *params) {
		const int eventid = 2;

		if (std::this_thread::get_id() != main_thread_id) {
			if (params && params->size() > 0)
			{
				filename = (*params)[0];
				FILE *test_fp;
				fopen_s(&test_fp, filename.c_str() , "r");
				if (!test_fp)
					filename = "d:/flow_data/bsldatanormalized/" + filename;
				fopen_s(&test_fp, filename.c_str(), "r");
				if(!test_fp)
					filename = filename + ".bsl";
				reduced = -1;
				curr_application = LSH_None;

				if (params->size() > 1)
				{
					char* cstr_1 = const_cast<char*> ((*params)[1].c_str());
					if (lsh_application.find((*params)[1]) != lsh_application.end())
						curr_application = (LSH_Application)lsh_application[(*params)[1]];
					else if (my_atof(cstr_1) >= 0)
						curr_application = static_cast<LSH_Application>((int)(my_atof(cstr_1)));
					else
						curr_application = LSH_Application::LSH_None;

					if (params->size() > 2)
					{
						char* cstr_2 = const_cast<char*> ((*params)[2].c_str());
						reduced = (int)my_atof(cstr_2);
						if (params->size() > 3) {
							char* cstr_3 = const_cast<char*> ((*params)[3].c_str());
							lsh_radius = my_atof(cstr_3);
						}
					}
				}
			}
			message_mutex.lock();
			message_queue.emplace_back(eventid);
			message_mutex.unlock();
		}
		else {

			deinit();
			Init();
			glutPostRedisplay();
			printf("\n>");

		}

	});

	commands["error"] = command_callbacks.size();
	command_callbacks.emplace_back([](vector<string> *params) {
		const int eventid = 3;

		if (std::this_thread::get_id() != main_thread_id) {
			message_mutex.lock();
			message_queue.emplace_back(eventid);
			message_mutex.unlock();
		}
		else {
			static int err_cnt = 0;
			ERR(err_cnt++);
			printf("\n>");

		}

	});

	commands["repaint"] = command_callbacks.size();
	command_callbacks.emplace_back([](vector<string> *params) {
		const int eventid = 4;

		if (std::this_thread::get_id() != main_thread_id) { 
			message_mutex.lock();
			message_queue.emplace_back(eventid);
			message_mutex.unlock();
		}
		else {
			glutPostRedisplay();
			printf("\n>");

		}

	});
	commands["color"] = command_callbacks.size();
	command_callbacks.emplace_back([](vector<string> *params) {
		const int eventid = 5;

		if (std::this_thread::get_id() != main_thread_id) {
			int cnt_component = 0;
			if (params && params->size() > 0)
				for (string color_component : *params)
				{
					if (params->size() < 12)
					{
						for (int i = 0; i < 4; i++)
						{
							char* cstr = const_cast<char *>(color_component.c_str());
							bg_colors[i][cnt_component] = my_atof(cstr);
						}
						if (cnt_component >= 2)
							break;
					}
					else
					{
						char* cstr = const_cast<char *>(color_component.c_str());
						bg_colors[cnt_component / 3][cnt_component % 3] = my_atof(cstr);
						if (cnt_component >= 11)
							break;
					}
					cnt_component++;
				}

			message_mutex.lock();
			message_queue.emplace_back(eventid);
			message_mutex.unlock();
		}
		else {
			//printf("!colors_running\n");
			glBindBuffer(GL_ARRAY_BUFFER, VBO[2]);
			glBufferSubData(GL_ARRAY_BUFFER, sizeof(Vector3) * 4 + sizeof(Vector3) * 4 + sizeof(float) * 4, sizeof(Vector3) * 4, bg_colors);
			glutPostRedisplay();
			printf("\n>");

		}

	});
	commands["linewidth"] = command_callbacks.size();
	command_callbacks.emplace_back([](vector<string>* params) {
		const int eventid = 6;

		if (std::this_thread::get_id() != main_thread_id) {
			if (params && params->size() > 0)
			{
				char* cstr = const_cast<char *>((*params)[0].c_str());
				const float linewidth = my_atof(cstr);
				if (linewidth<=10.f && linewidth >= 1.f)
				{
					line_width = linewidth;
				}
			}

			message_mutex.lock();
			message_queue.emplace_back(eventid);
			message_mutex.unlock();
		}
		else
		{

			glLineWidth(line_width);
			glUseProgram(Phong_Render);
			glUniform1f(line_width_index, line_width);
			glutPostRedisplay();
			printf("\n>");
			
		}
	}

	);
	commands["filepicker"] = command_callbacks.size();
	command_callbacks.emplace_back([](vector<string>* params) {
		const int eventid = 7;

		if (std::this_thread::get_id() != main_thread_id) {
			message_mutex.lock();
			message_queue.emplace_back(eventid);
			message_mutex.unlock();
		}
		else
		{
			HDC curr_dc = wglGetCurrentDC();
			HWND curr_hwnd = WindowFromDC(curr_dc);
			OPENFILENAME ofn;       // common dialog box structure
			TCHAR szFile[260] = { 0 };       // if using TCHAR macros
			ZeroMemory(&ofn, sizeof(ofn));
			ofn.lStructSize = sizeof(ofn);
			ofn.hwndOwner = curr_hwnd;
			ofn.lpstrFile = szFile;
			ofn.nMaxFile = sizeof(szFile);
			ofn.lpstrFilter = L"All\0*.*\0Wave Front Object\0*.obj\0Binary Streamlines\0*.bsl\0";
			ofn.nFilterIndex = 1;
			ofn.lpstrFileTitle = NULL;
			ofn.nMaxFileTitle = 0;
			ofn.lpstrInitialDir = L"d:/flow_Data/bsldatanormalized/";
			ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;
			if (GetOpenFileName(&ofn) == TRUE)
				filename = str_from_wstr(ofn.lpstrFile);
			deinit();
			Init();
			glutPostRedisplay();
			printf("\n>");

		}
	}

	);
	commands["fullscreen"] = command_callbacks.size();
	command_callbacks.emplace_back([](vector<string>* params) {
		const int eventid = 8;

		if (std::this_thread::get_id() != main_thread_id) {

			message_mutex.lock();
			message_queue.emplace_back(eventid);
			message_mutex.unlock();
		}
		else
		{
			HDC curr_dc = wglGetCurrentDC();
			HWND curr_hwnd = WindowFromDC(curr_dc);
			SetActiveWindow(curr_hwnd);
			glutFullScreenToggle();
			printf("\n>");
		}
	}

	);

	commands["smooth"] = command_callbacks.size();
	command_callbacks.emplace_back([](vector<string>* param) {
		const int eventid = 9;
		if (std::this_thread::get_id() != main_thread_id) {
			gaussianSmooth();

			message_mutex.lock();
			message_queue.emplace_back(eventid);
			message_mutex.unlock();
		}
		else
		{

			InitRenderer();
			glutPostRedisplay();
			printf("\n>");
		}
	});

	commands["fastrender"] = command_callbacks.size();
	command_callbacks.emplace_back([](vector<string>* param) {
		const int eventid = 10;
		if (std::this_thread::get_id() != main_thread_id) {
			Status::fastrender = !fastrender;
			message_mutex.lock();
			message_queue.emplace_back(eventid);
			message_mutex.unlock();
		}
		else
		{
			glutPostRedisplay();
			printf("\n>");
		}
	});

	main_thread_id = std::this_thread::get_id();
	glutMainLoop();
	//while (1)
	//	glutMainLoopEvent();
	glutExit();
	command_parser_daemon.join();

	return 0;
}
