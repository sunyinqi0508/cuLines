#ifndef VIEWER_H
#define VIEWER_H

#include <QOpenGLWidget>

#include <qcoreapplication.h>
#include <sstream>
#include <math.h>
#include <gl\GL.h>
#include <gl\GLU.h>
#include <QTouchEvent>
#include <qdebug.h>
#include <qtimer.h>
#include <time.h>

#include <stdarg.h>
#include <cmath>
#include <iostream>
#include <string>


#include "glExtensions.h"
#include <fstream>

#include "ILines/Vector.h"

#include "ILines/ILRender.h"

//#include "ILines/ILLightingModel.h"

#include "FPSCounter.h"

#include <map>
#include <windows.h>
#include <stdio.h>      /* printf, scanf, puts, NULL */
#include <stdlib.h>     /* srand, rand */
#include <atomic>
#include "streamline.h"
#include "Field2D.h"
#include "Integrator.h"
#include "SimTester.h"
#include "Vector3.h"
#include <iostream>
#include <algorithm>
#include <queue>
#include <mutex>
#include <map>
#include <set>
#include <qelapsedtimer.h>
#include <QMetaType>
#include <QKeyEvent>

#include "CudaVariableStruct.h"
#include "CalcLineOrderThread.h"
#include "LightParameters.h"
#include <Parameters.h>
using namespace std;
using namespace ILines;

struct GLfloatPoint
{
	GLfloat x, y;
};
const int BUFSIZE = 50 << 20;


class Viewer : public QOpenGLWidget
{
	Q_OBJECT

public:
	//multi-touch 
	static Viewer *This;
	static bool running;
	static clock_t last_update, update_started;
	Viewer(QWidget *parent = 0);
	~Viewer();
	GLuint text_list;
	void selectFont(int size, int charset, const char* face);
	void drawString(float x, float y, string str);

	//flow visualization
	float max_k_to_split_lines = 1.08;
	vector<Line> streamlines;
	bool doColors = false;// paint pattern,true:paint by points,false:paint by lines
	int old_x = 0;
	int old_y = 0;
	int rotate_x = 0;
	int rotate_y = 0;
	int zoom_Z = 0;
	int cur_line_num = 0;//current can delete lines num
	int dim_x, dim_y, dim_z;	// dimension of flow field
	float max_x = -999999, max_y = -999999, max_z = -999999;
	float min_x = 999999, min_y = 999999, min_z = 999999;
	float _orig_minx = 0, _orig_miny = 0, _orig_minz = 0;
	float k_x = 1.8, k_y = 0.4, k_z = 0.3;
	float ka = 0.05f, kd = 0.8f, ks = 1.0f, gloss = 10.0f;
	bool dataHasColors = false;
	bool extVertexBufferObject = false;
	bool extMultiDrawArrays = false;
	FPSCounter fpsCounter;
	ILRender maximumPhongIL;
	ILRender cylinderPhongIL;
	ILRender cylinderBlinnIL;
	ILRender *curIL;
	bool maximumPhongSupported;
	bool cylinderBlinnSupported;
	bool cylinderPhongSupported;
	PFNGLBINDBUFFERPROC pglBindBuffer = NULL;
	PFNGLBUFFERDATAPROC pglBufferData = NULL;
	PFNGLGENBUFFERSPROC pglGenBuffers = NULL;
	PFNGLMULTIDRAWARRAYSPROC pglMultiDrawArrays = NULL;
	GLuint vboVertID;
	GLuint vboColID;
	ILRender::ILIdentifier ilID_solid_lines = ILRender::IL_INVALID_IDENTIFIER;
	ILRender::ILIdentifier ilID_deleted_lines = ILRender::IL_INVALID_IDENTIFIER;
	ILRender::ILIdentifier oneline_tmp = ILRender::IL_INVALID_IDENTIFIER;
	ILLightingModel::Model lightingModel;
	Vector3f X, Y, Z;
	//solid lines
	int lineCount;
	int *first;
	int *vertCount;
	//deleted lines
	int lineCount_deleted_lines;
	int *first_deleted_lines;
	int *vertCount_deleted_lines;
	//selected one line
	int  oneline_lineCount;
	int	*oneline_first;
	int	*oneline_vertCount;
	QSlider *slider = 0;

	float *vertices = NULL;
	float *colors = NULL;
	int totalSize;
	int N_p = -1;
	int texDim = 256;
	int transX = 0, transY = 0;
	char Buf[BUFSIZE + 1], *buf = Buf;
	float afasim = 0.9;
	float setdSef = 0.2;
	int HalfSample = 5;
	int *linearmn, *linearmx, *linearch, *kdtree_id, n_backup;
	vector<int> *line_division;
	int *d_id, /*host */*linearKD_id;
	int *d_lkd, *d_buckets_g;
	//int *p_left[BUFFERS], currBuffer = BUFFERS / 2; 
	//vector<float> Sum_Similarity;
	int *linearKd;
	//vector<vector<float>> Simimarity;
	vector<vector<int>> Nj;
	float Steplenghth = 2;
	double alpha_value = 1.5;
	float light_x = 0, light_y = 0, light_z = 0;
	CalcLineOrderThread *calLineOrderThread;

	void makeConfigFile();
	bool read(char *&d);
	void makeData();
	void makeData2();
	void readfromBin(int & N);
	void readTrajectoryData();
	void readFromTxt();
	void init();
	void initGLExtensions();
	void initGL();
	void initIL();
	void display();
	void reshape(int width, int height);
	void cleanup();
	void die();
	//repaint
	void repaint();

	void update();

	void errorCallbackIL(ILRender *ilRender);
	vector<int> getSelectedLinesIndex(float r0, Vector3 pt);
	void prepareLinesToDraw(double x, double y, double z, double r0);
	void displayScene();
	dsim min_dsim(Point a);
	vector<int> getLocalLineOrder(set<int> *local, int del);
	vector<int> selected_lines_index_list;
	vector<int> deleted_lines_index_list;
	bool is_zoom_sign = false;
	bool is_translate_sign = false;
	//float **val = new float*[BUFFERS], **variation = new float*[BUFFERS];
	DevicePointers* device_pointers;
	bool is_change_mode_sign = false;
	vector<float> opacity_point;
	void changeDeletedLinesArray(vector<int> &arr);
	void getLineToDeleteLinesNum();
	void setModeAction(int mode);
	bool is_light_move_action = false;
	float light_move_x = 0;
	float light_move_y = 0;
	float light_move_z = 0;
	bool is_get_light_pos = false;
	int light_mode = 0;
	int material_mode = -1;
	int change_light_para = 1;
	float line_width;

	//light parameters
	float color_r = 0;
	float color_g = 0;
	float color_b = 0;

	//void readGermandata(vector<Line> &streamlines0, string data);// , int inputlines);
	int readBenddata(vector<Line> &streamlines0, const char*data);
	int readGuoningdata(vector<Line> &streamlines0, const char*data);
	void setLight();
	void printLightParameters();

signals:
	void sendTotalLineNum(int num1, int num2);
	void sendLineToDeleteNum(int num);
	void sendUnfinishSign(bool sign);//if global line order size < line to delete num
	void sendSelectedLinesIndex(vector<int> vec1, vector<int> vec2);
	void sendSelectedLines(vector<int> line_order);
	void reset();
	void sendDeletedLinesIndexList(vector<bool> deleted_lines_index_list);
	void sendPosZSlider(int max_z);
	void sendTop();

private slots:
	void scaleTimeout();
	void startMove();
	void receiveLineOrder(vector<int> order, int sign);
	void getLineToDelete(int num);
	void getSelectedOneLine(vector<int> line_id_list);
	void getSelectedLinesIndexList(vector<int> list);
	void getParameters(vector<int> *line_division, int *linearKd, int *linearKD_id);
	void clearLineOrderPartList();
	//void getValVariation(void * device_pointers);
	void getDevicePointers(void *dp);
	void alphaChanged(int value);

	void modeChangeAction();
	void selectAreaAction();
	void keyPressEvent(QKeyEvent *key);
	void moveLightAction();
	void getLightParameters(int id, float value);
	void getColor(QColor c);
	void resetAction();
	void saveSettings();
	void loadSettings();
protected:
	void initializeGL() Q_DECL_OVERRIDE;
	void paintGL() Q_DECL_OVERRIDE;
	void resizeGL(int width, int height) Q_DECL_OVERRIDE;
	bool event(QEvent *event) Q_DECL_OVERRIDE;
	void touchEvent(QTouchEvent *ev);
	void mouseEvent(QMouseEvent*ev);
	void keyboardEvent(QKeyEvent * event);
private:
	HANDLE working_thread = 0;
	void *thread_data = 0;
	int widgetWidth;
	int widgetHeight;
	GLfloat preDis;//之前的距离
	GLfloat preDis_x;//之前的距离
	GLfloat preDis_y;//之前的距离
	GLfloat preX0 = 0;
	GLfloat preY0 = 0;
	GLfloat preX1 = 0;
	GLfloat preY1 = 0;
	bool light_pos_moving = false;
	bool is_global_sign = true;//global delete or local delete lines
	bool is_reset_sign = false;//confirm the reset action just do once
	bool is_ball_change_sign = false;//select local line sign
	bool start_move = true;//two fingers start move together in the same direction
	bool finishedcal_sign = false;//line order to delete do not finish calculate
	bool clicked = false, zoomed = false;
	float init_zoom = 0;

	int finger_num = 0;
	int listArray[10];//（手指个数，出现次数），保证不同手指数目不会混乱
	GLfloat radius = 10;
	GLfloat circle_x = 0, circle_y = 0, circle_z = 0, circle_r = 0;
	int selected_one_line_index = 0;
	int previous_deleted_lines_num = -1;
	vector<int> selected_one_line_index_list;
	clock_t doubleclick_start = 0, doubleclick_end = 0;
	int line_to_delete_num = 0;//需要删除的线的数目
	int deleted_lines_num = 0;//被删除的线的数目
	int translateX = 0;
	int translateY = 0;
	int translateZ = 0;
	int zoom_speed = 5;
	vector<int> lineOrder_global;//calc every selected one line
	vector<int> lineOrder_part;
	bool *deleted_lines_arr;
	FILE *cfg = 0;
	bool rec = false, res = false;
	void zoomTheObject(GLfloat x0, GLfloat y0, GLfloat x1, GLfloat y1);
};


#endif // VIEWER_H
