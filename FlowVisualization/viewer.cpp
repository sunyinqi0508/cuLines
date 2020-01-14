#include "viewer.h"
#include "Common.h"
#include <qfiledialog.h>

//#include <glut>

extern void cavg(float *similarity, int n, float *avg, float *max);
extern void cuda(float *similarity, float *distance, float *output, int n, float avg, float alpha, float _max);

float uniformColor[4] = { 1.0f, 0.5f, 0.0f, 0.f };
GLfloat lightDirection[4] = { 0.0f, 0.0f, 1.0f };  
GLfloat lightPosition[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
Vector3f cameraPosition = Vector3f(0.0f, 0.0f, 0.0f);
Vector3f sceneCenter = Vector3f(0.0f, 0.0f, -1.0f);
Vector3f cameraUp = Vector3f(0.0f, 1.0f, 0.0f);
GLfloat	cameraPerspective[4] = { 45.0, 1.0, 1.f, 1000.0 };
StreamlineParam g_param;
double m_minStep = 1;
double m_maxStep = 4;
double max_ds;
Viewer* Viewer::This = 0;
bool Viewer::running = true;
clock_t Viewer::last_update = 0;
clock_t Viewer::update_started = 0;
bool hold = false, hit = true;
DWORD WINAPI force_update_deamon(LPVOID param) {
	
	while (Viewer::running) {
		Sleep(100);
		if(Viewer::last_update)
			if (clock() - Viewer::last_update > 300 && clock() - Viewer::update_started > 300) {
				hit = true;
				if (!hold)
				{
					hold = true;
					hit = false;
					ILUtilities::repaint_pending = false;
					Viewer::This->update();
				}
			}
			else if(hit){
				hold = false;
			}

	}
	
	return 0;

}
Viewer::Viewer(QWidget *parent) : QOpenGLWidget(parent)
{
	this->This = this;
	setFocusPolicy(Qt::ClickFocus);
	setAttribute(Qt::WA_AcceptTouchEvents, true);//设置这个为了接收触摸事件
	setAttribute(Qt::WA_TouchPadAcceptSingleTouchEvents);

	widgetWidth = this->width();
	widgetHeight = this->height();

	preDis = 0;
	finger_num = 0;
	radius = 0;
	deleted_lines_num = 0;

	for (int i = 0; i < 10; i++)
	{
		listArray[i] = 0;
	}
	//CreateThread(0, 0, force_update_deamon, 0, 0, 0);
	/*QElapsedTimer et;
	et.start();
	while (et.elapsed()<300)
		QCoreApplication::processEvents();*/
}

Viewer::~Viewer()
{
	makeCurrent();
}

void Viewer::selectFont(int size, int charset, const char* face) {
	HFONT hFont = CreateFontA(size, 0, 0, 0, FW_MEDIUM, 0, 0, 0,
		charset, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
		DEFAULT_QUALITY, DEFAULT_PITCH | FF_SWISS, face);
	HFONT hOldFont = (HFONT)SelectObject(wglGetCurrentDC(), hFont);
	DeleteObject(hOldFont);
}
void Viewer::drawString(float x, float y, string str){
	glRasterPos2f(x, y);
	for (int i = 0; i < str.size(); i++){
		glCallList(text_list + str[i]);
	}
}
void Viewer::initializeGL()
{
	selectFont(48, ANSI_CHARSET, "Comic Sans MS");
	text_list = glGenLists(MAX_CHAR);
	wglUseFontBitmaps(wglGetCurrentDC(), 0, MAX_CHAR, text_list);
	clock_t start, finish;
	start = clock();
	makeData();
	finish = clock();
	qDebug() << "make data time:" << (finish - start) / CLOCKS_PER_SEC << " (s) ";
	makeConfigFile();
	emit sendTotalLineNum(streamlines.size(), 0);
	init();
	qDebug() << "initial";
}

void Viewer::paintGL()
{
	//ILines::ILUtilities::repaint_pending++;
	
	clock_t start, finish;
	start = clock();
	update_started = start;
	display();
	ILUtilities::repaint_pending = true;
	finish = clock();
	last_update = finish;
	//qDebug() << "display time:" << (finish - start);// / CLOCKS_PER_SEC << " (s) ";
	/*qDebug() << "lineOrder_part size:" << lineOrder_part.size();
	qDebug() << "select one line is " << selected_one_line_index;
	qDebug() << "is global sign:" << is_global_sign;
	qDebug() << "deleted lines num:" << deleted_lines_num;
	qDebug() << "selected lines num:" << selected_lines_index_list.size();
	qDebug() << "===================================";*/
}

void Viewer::resizeGL(int width, int height)
{
	reshape(width, height);
}

float getLineWidth(int line_num){
	//return 15;
	if (line_num > 0 && line_num <= 50) {
		return 3.0;
	}
	else if (line_num <= 100)
		return 2.6;
	else if (line_num > 100 && line_num <= 200){
		return 2.4;
	}
	else if (line_num > 200 && line_num <= 400){
		return 2.2;
	}
	else if (line_num > 400 && line_num <= 800){
		return 2.f;
	}
	else if (line_num > 800 && line_num <= 1600){
		return 1.5;
	}
	else
	{
		return 1.f;
	}
}

struct ChangeDeletedLinesArrayAsync {
	vector<int>* arr;
	int deleted_lines_num;
	bool* deleted_lines_arr;
	clock_t last_time = 0;
	bool dispatched = false;
	ChangeDeletedLinesArrayAsync* waitingQueue = 0;
	Viewer* this_ptr;
	int streamline_size;
	std::mutex mtx;
};
DWORD WINAPI changeDeletedLinesArrayAsync(LPVOID param) {
	ChangeDeletedLinesArrayAsync* p = (ChangeDeletedLinesArrayAsync*)(param);
	int *_arr = p->arr->data();
	int size = p->arr->size();

	bool * deleted_lines_arr = p->deleted_lines_arr;
	int deleted_lines_num = p->deleted_lines_num;
	bool ctn = true;
	do {
		for (int i = 0; i < size; i++)
			if (i < deleted_lines_num)
				deleted_lines_arr[_arr[i]] = true;
			else
				deleted_lines_arr[_arr[i]] = false;
		vector<bool> tmp_deleted_lines_arr;
		tmp_deleted_lines_arr = vector<bool>(deleted_lines_arr, deleted_lines_arr + p->streamline_size);
		/*for (int i = 0; i < streamlines.size(); i++){
		tmp_deleted_lines_arr.push_back(deleted_lines_arr[i]);
		}*/
		emit p->this_ptr->sendDeletedLinesIndexList(tmp_deleted_lines_arr);
		if (clock() - p->last_time > p->this_ptr->streamlines.size()/20.0)
		{
			p->this_ptr->update();
			p->last_time = clock();
		}
		//Sleep(200);
		p->mtx.lock();
		if (p->dispatched && p->waitingQueue) {

				deleted_lines_num = p->waitingQueue->deleted_lines_num;
				_arr = p->waitingQueue->arr->data();
				size = p->waitingQueue->arr->size();

				//delete p->waitingQueue;
				p->waitingQueue = NULL;
				p->dispatched = false;
				ctn = true;
		}
		else {
			ctn = false;
		}
		p->mtx.unlock();
	} while (ctn);
	//p->this_ptr->update();
	return 0;
}
void Viewer::changeDeletedLinesArray(vector<int> &arr) {
	if (this->thread_data == 0) {
		//cout << 1 << endl;
		thread_data = new ChangeDeletedLinesArrayAsync();
		((ChangeDeletedLinesArrayAsync*)thread_data)->deleted_lines_arr = this->deleted_lines_arr;
		((ChangeDeletedLinesArrayAsync*)thread_data)->streamline_size = streamlines.size();
		((ChangeDeletedLinesArrayAsync*)thread_data)->this_ptr = this;
	}
	if (working_thread == 0 || WaitForSingleObject(this->working_thread, 0) == WAIT_OBJECT_0)
	{

		((ChangeDeletedLinesArrayAsync*)thread_data)->deleted_lines_num = deleted_lines_num;
		((ChangeDeletedLinesArrayAsync*)thread_data)->arr = &arr;
		if (((ChangeDeletedLinesArrayAsync*)thread_data)->dispatched && ((ChangeDeletedLinesArrayAsync*)thread_data)->waitingQueue) {
			//delete ((ChangeDeletedLinesArrayAsync*)thread_data)->waitingQueue;
		}
		((ChangeDeletedLinesArrayAsync*)thread_data)->dispatched = false;
		((ChangeDeletedLinesArrayAsync*)thread_data)->waitingQueue = NULL;
		//cout << 2 << endl;
		working_thread = CreateThread(0, 0, changeDeletedLinesArrayAsync, thread_data, 0, 0);
		WaitForSingleObject(working_thread,INFINITE);
	}
	else {
		cout << 3 << endl;

		//ChangeDeletedLinesArrayAsync* appendices = new ChangeDeletedLinesArrayAsync();
		//((ChangeDeletedLinesArrayAsync*)appendices)->deleted_lines_num = deleted_lines_num;
		//((ChangeDeletedLinesArrayAsync*)appendices)->arr = &arr;

		//((ChangeDeletedLinesArrayAsync*)thread_data)->mtx.lock();

		//if (((ChangeDeletedLinesArrayAsync*)thread_data)->dispatched && ((ChangeDeletedLinesArrayAsync*)thread_data)->waitingQueue){
		//	//delete ((ChangeDeletedLinesArrayAsync*)thread_data)->waitingQueue;
		//}

		//((ChangeDeletedLinesArrayAsync*)thread_data)->dispatched = true;
		//((ChangeDeletedLinesArrayAsync*)thread_data)->waitingQueue = appendices;

		//((ChangeDeletedLinesArrayAsync*)thread_data)->mtx.unlock();

	}

}
/*
void Viewer::changeDeletedLinesArray(vector<int> &arr){
	for (int i = 0; i < arr.size(); i++){
		if (i < deleted_lines_num){
			deleted_lines_arr[arr[i]] = true;
		}
		else
		{
			deleted_lines_arr[arr[i]] = false;
		}
	}
}*/

void Viewer::getLineToDeleteLinesNum(){
	if (line_to_delete_num <= cur_line_num && line_to_delete_num >= 0)
	{
		if (line_to_delete_num <= lineOrder_global.size() && is_global_sign){
			deleted_lines_num = line_to_delete_num;
			changeDeletedLinesArray(lineOrder_global);
			finishedcal_sign = true;
		}
		else if (line_to_delete_num <= lineOrder_part.size() && !is_global_sign){
			deleted_lines_num = line_to_delete_num;
			changeDeletedLinesArray(lineOrder_part);
			finishedcal_sign = true;
		}
		else
		{
			/*for (int i = 0; i < streamlines.size(); i++){
			tmp_deleted_lines_arr.push_back(deleted_lines_arr[i]);
			}*/
			deleted_lines_num = lineOrder_global.size();// -1;
			changeDeletedLinesArray(lineOrder_global);
			finishedcal_sign = false;
		}

		emit sendTotalLineNum(cur_line_num, deleted_lines_num);
		emit sendLineToDeleteNum(line_to_delete_num);
		emit sendUnfinishSign(finishedcal_sign);
	}
	else
	{
		if (line_to_delete_num < 0){
			line_to_delete_num = 0;
		}
		else if (line_to_delete_num > cur_line_num){
			//line_to_delete_num = cur_line_num;
			vector<bool> tmp_deleted_lines_arr;
			tmp_deleted_lines_arr = vector<bool>(deleted_lines_arr, deleted_lines_arr + streamlines.size());
			/*for (int i = 0; i < streamlines.size(); i++){
			tmp_deleted_lines_arr.push_back(deleted_lines_arr[i]);
			}*/
			emit sendDeletedLinesIndexList(tmp_deleted_lines_arr);
			emit sendTotalLineNum(cur_line_num, deleted_lines_num);
			emit sendLineToDeleteNum(line_to_delete_num);
			emit sendUnfinishSign(finishedcal_sign);
		}
	}
}
bool isin(vector<int> Ndelete, int a)
{
	bool b = false;

	for (int i = 0; i < Ndelete.size(); i++)
	{
		if (Ndelete.at(i) == a)
		{
			b = true;
			break;
		}
	}
	return b;

}
void Viewer::touchEvent(QTouchEvent *ev)
{
	switch (ev->type())
	{
	case QEvent::TouchBegin:
		doubleclick_start = clock();
		if (abs(doubleclick_end - doubleclick_start) < 150) {
			//doColors = !doColors;//change paint pattern between line and points
		}
		old_x = ev->touchPoints()[0].pos().x();
		old_y = ev->touchPoints()[0].pos().y();
		/*	light_move_x = old_x;
			light_move_y = old_y;*/
		preX0 = old_x;
		preX1 = old_x;
		preY0 = old_y;
		preY1 = old_y;
		break;
	case QEvent::TouchEnd:
		doubleclick_end = clock();
		is_reset_sign = false;
		zoom_speed = 5;
		start_move = true;
		if (is_ball_change_sign && selected_lines_index_list.size() > 0) {
			line_to_delete_num = deleted_lines_index_list.size();
			getLineToDeleteLinesNum();
			vector<bool> tmp_deleted_lines_arr;
			for (int i = 0; i < streamlines.size(); i++) {
				tmp_deleted_lines_arr.push_back(deleted_lines_arr[i]);
			}
			emit sendDeletedLinesIndexList(tmp_deleted_lines_arr);
			emit sendSelectedLines(selected_lines_index_list);
			for (int i = 0; i < streamlines.size(); i++) {
				if (!isin(selected_lines_index_list, i)) {
					deleted_lines_arr[i] = true;
				}
			}
		}

		//get local line order
		if (is_ball_change_sign && !is_global_sign) {//finish with line selected
			if (selected_one_line_index_list.size() == 0 && lineOrder_global.size() > selected_lines_index_list.size())//local of mode 1
			{
				if (selected_lines_index_list.size() > 0) {
					lineOrder_part.clear();
					set<int> tmp_Ndelete = set<int>(selected_lines_index_list.begin(), selected_lines_index_list.end());
					lineOrder_part = getLocalLineOrder(&tmp_Ndelete, selected_lines_index_list.size());
				}
			}
		}

		is_ball_change_sign = false;
		is_zoom_sign = false;
		is_translate_sign = false;
		is_light_move_action = false;
		is_get_light_pos = false;
		update();

		break;
	case QEvent::TouchUpdate:
		QList<QTouchEvent::TouchPoint> list = ev->touchPoints();
		if (list.length() == 1) //rotate
		{
			if (finger_num == 2)
			{
				QTimer::singleShot(100, this, SLOT(scaleTimeout()));
			}
			else {
				if (is_ball_change_sign) {
					GLfloat updateDis = sqrt((list[0].pos().x() - old_x)*(list[0].pos().x() - old_x) + (list[0].pos().y() - old_y)*(list[0].pos().y() - old_y));
					if (updateDis > preDis)
						radius++;
					else if (updateDis < preDis)
						radius--;
					preDis = updateDis;
				}
				else
				{
					is_ball_change_sign = false;
					int x = list[0].pos().x();
					int y = list[0].pos().y();
					if (!is_light_move_action) {
						rotate_y = (x - old_x);
						old_x = x;
						rotate_x = (y - old_y);
						old_y = y;
					}
					else
					{
						light_move_x += (float)(x - preX0) / 10;
						light_move_y += -(float)(y - preY0) / 10;
						preX0 = x;
						preY0 = y;
						lightPosition[0] = light_x;
						lightPosition[1] = light_y;
						lightPosition[0] += light_move_x;
						lightPosition[1] += light_move_y;
					}
				}
			}
		}
		else if (list.length() == 2)
		{
			if (finger_num == 3)
			{
				QTimer::singleShot(100, this, SLOT(scaleTimeout()));
			}
			else {
				if (list[0].state() == Qt::TouchPointStationary && list[1].state() == Qt::TouchPointStationary) {

				}
				else
				{
					//move in the same direction:translate
					if (preX0 <= list[0].pos().x() && preX1 <= list[1].pos().x() && preY0 > list[0].pos().y() && preY1 > list[1].pos().y() ||
						preX0 >= list[0].pos().x() && preX1 >= list[1].pos().x() && preY0 < list[0].pos().y() && preY1 < list[1].pos().y() ||
						preX0 > list[0].pos().x() && preX1 > list[1].pos().x() && preY0 >= list[0].pos().y() && preY1 >= list[1].pos().y() ||
						preX0 < list[0].pos().x() && preX1 < list[1].pos().x() && preY0 <= list[0].pos().y() && preY1 <= list[1].pos().y()
						) {
						if (!is_light_move_action) {
							is_translate_sign = true;
							translateX += (preX0 - list[0].pos().x()) / 10;
							translateY += (list[0].pos().y() - preY0) / 10;
						}
					}
					else if (!is_light_move_action && !is_translate_sign)//move in the different direction at same time:scale
					{
						zoom_speed += 2;
						//is_zoom_sign = true;
						zoomTheObject(list[0].pos().x(), list[0].pos().y(), list[1].pos().x(), list[1].pos().y());
					}
					preX0 = list[0].pos().x();
					preX1 = list[1].pos().x();
					preY0 = list[0].pos().y();
					preY1 = list[1].pos().y();
				}
			}
			finger_num = 2;
		}
		else if (list.length() == 3) //delete line
		{
			is_ball_change_sign = false;
			//a finger is fixed,the others are moving:choose a line
			if (list[0].state() == Qt::TouchPointStationary && list[1].state() == Qt::TouchPointMoved && list[2].state() == Qt::TouchPointMoved) {
				if (list[1].pos().y() > preDis)
				{
					//selected_one_line_index -= 1;
				}
				else if (list[1].pos().y() < preDis) {
					//selected_one_line_index += 1;
				}
				preDis = list[1].pos().y();
			}
			//two fingers are fixed,the other is moving:draw cicle or other size
			else if (list[0].state() == Qt::TouchPointStationary && list[1].state() == Qt::TouchPointStationary && list[2].state() == Qt::TouchPointMoved) {

			}
			//all move in the same direction:delete line
			else
			{
				if (list[0].pos().y() > preDis)
				{
					line_to_delete_num--;
				}
				else if (list[0].pos().y() < preDis) {
					line_to_delete_num++;
				}
				getLineToDeleteLinesNum();
				preDis = list[0].pos().y();
			}
			finger_num = 3;
		}
		else if (list.length() == 5) //back to origin
		{
			GLfloat updateDis = sqrt((list[0].pos().x() - list[4].pos().x())*(list[0].pos().x() - list[4].pos().x()) + (list[0].pos().y() - list[4].pos().y())*(list[0].pos().y() - list[4].pos().y()));
			if (updateDis < preDis && !is_reset_sign) {
				is_reset_sign = true;
				circle_x = circle_y = circle_z = circle_r = 0;
				is_global_sign = true;
				is_ball_change_sign = false;
				finishedcal_sign = false;
				cur_line_num = streamlines.size();
				emit sendTotalLineNum(cur_line_num, 0);
				emit sendLineToDeleteNum(0);
				deleted_lines_num = 0;
				previous_deleted_lines_num = -1;
				line_to_delete_num = 0;
				selected_one_line_index = 0;
				selected_one_line_index_list.clear();
				radius = 0;
				fill(deleted_lines_arr, deleted_lines_arr + streamlines.size(), 0);
				translateX = 0;
				translateY = 0;
				rotate_y = 0;
				rotate_x = 0;
				zoom_Z = 0;
				light_move_x = 0;
				light_move_y = 0;
				light_move_z = 0;
				lightPosition[0] = light_x;
				lightPosition[1] = light_y;
				lightPosition[2] = light_z;
				color_r = (float)255 / 255;
				color_g = (float)215 / 255;
				color_b = (float)0 / 255;

				emit reset();
			}
			preDis = updateDis;
			finger_num = 5;
		}
		update();
		break;
	}
}

void Viewer::resetAction() {
	lightPosition[0] = light_x;
	lightPosition[1] = light_y;
	lightPosition[2] = light_z;
	color_r = (float)255 / 255;
	color_g = (float)215 / 255;
	color_b = (float)0 / 255;

	update();
}

void Viewer::saveSettings()
{
	FILE *fp;
	fopen_s(&fp, "d:/Flow_config/config.cfg", "w");
	this->cfg = fp;
	this->rec = true;

	fprintf_s(fp, "%d %d\n", rotate_x, rotate_y);
	fprintf_s(fp, "%d %d %d\n", translateX, translateY, translateZ);
	fprintf_s(fp, "%f %f %f\n", X.x, X.y, X.z);
	fprintf_s(fp, "%f %f %f\n", Y.x, Y.y, Y.z);
	fprintf_s(fp, "%f %f %f\n", Z.x, Z.y, Z.z);
	fprintf_s(fp, "%d\n", zoom_Z);
	update();

}
void Viewer::loadSettings()
{
	FILE *fp;
	fopen_s(&fp, "d:/Flow_Config/config.cfg", "r");
	fscanf_s(fp, "%d %d\n", &rotate_x, &rotate_y);
	fscanf_s(fp, "%d %d %d\n", &translateX, &translateY, &translateZ);
	fscanf_s(fp, "%f %f %f\n", &X.x, &X.y, &X.z);
	fscanf_s(fp, "%f %f %f\n", &Y.x, &Y.y, &Y.z);
	fscanf_s(fp, "%f %f %f\n", &Z.x, &Z.y, &Z.z);
	fscanf_s(fp, "%d\n", &zoom_Z);

	this->cfg = fp;
	this->res = true;
	update();
	
}
void Viewer::scaleTimeout(){
	finger_num = 0;
}
void Viewer::startMove(){
	start_move = false;
}

void Viewer::mouseEvent(QMouseEvent * ev)
{
	switch (ev->type()) {
	case QEvent::MouseButtonPress:
		if (ev->button() == Qt::MouseButton::RightButton)
		{
			this->zoomed = true;

			int x = ev->pos().x() - this->widgetHeight / 2.f;
			int y = ev->pos().y() - this->widgetWidth / 2.f;

			//float init_zoom_x = ev->pos().x();
			//float init_zoom_y = ev->pos().y();
			init_zoom = sqrt(x*x + y * y); 
			printf("%f\n", init_zoom);
		}
		else
			this->clicked = true;

		break;
	case QEvent::MouseButtonRelease:
		if (ev->button() == Qt::MouseButton::RightButton)
			this->zoomed = false;
		else
			this->clicked = false;
		update();
		break;
	case QEvent::MouseMove:
	{
		if (clicked) {
			
			QPoint pos = ev->pos();
			if (is_ball_change_sign) {
				GLfloat updateDis = sqrt((pos.x() - old_x)*(pos.x() - old_x) + (pos.y() - old_y)*(pos.y() - old_y));
				if (updateDis > preDis)
					radius++;
				else if (updateDis < preDis)
					radius--;
				preDis = updateDis;
			}
			else
			{
				is_ball_change_sign = false;
				int x = pos.x();
				int y = pos.y();
				if (!light_pos_moving) {
					rotate_y = (x - old_x);
					old_x = x;
					rotate_x = (y - old_y);
					old_y = y;
				}
				else
				{
					light_move_x += (float)(x - preX0) / 10;
					light_move_y += -(float)(y - preY0) / 10;
					preX0 = x;
					preY0 = y;
					lightPosition[0] = light_x;
					lightPosition[1] = light_y;
					lightPosition[0] += light_move_x;
					lightPosition[1] += light_move_y;
				}
			}
			update();
		}
		else if (zoomed) {
			QPoint pos = ev->pos();
			int x = pos.x() - this->widgetHeight/2.f;
			int y = pos.y() - this->widgetWidth/2.f;
			float this_zoom = sqrt(x*x+ y*y);
			float d_zoom = (-this_zoom + init_zoom) *.4;
			init_zoom = this_zoom;
			zoom_Z += d_zoom;
			update();
		}
		break;
	}
	}
}


void Viewer::keyboardEvent(QKeyEvent *event) {
	int kc = event->key();
	if (kc == Qt::Key::Key_Control) 
		switch (event->type())
		{
		case QKeyEvent::KeyPress:
			light_pos_moving = true;
			break;
		case QKeyEvent::KeyRelease:
			light_pos_moving = false;
			break;
		}
}
bool Viewer::event(QEvent *event) {
	switch (event->type())
	{
	case QEvent::TouchBegin:
	case QEvent::TouchEnd:
	case QEvent::TouchCancel:
	case QEvent::TouchUpdate:
		event->accept();
		touchEvent((QTouchEvent*)event);
		return true;
	case QEvent::MouseButtonPress:
	case QEvent::MouseButtonRelease:
	case QEvent::MouseMove:
		event->accept();
		mouseEvent((QMouseEvent*)event);
		return true;

	case QEvent::KeyPress:
	case QEvent::KeyRelease:
		keyboardEvent((QKeyEvent *)event);
		return true;
	default:
		return QWidget::event(event);
		//event->
	}
}

void Viewer::zoomTheObject(GLfloat x0, GLfloat y0, GLfloat x1, GLfloat y1){

	int x = (x0 + x1) / 2;
	int y = (y0 + y1) / 2;
	GLfloat updateDis = sqrt((x0 - x1)*(x0 - x1) + (y0 - y1)*(y0 - y1));
	if (updateDis > preDis)//enlarge
	{
		zoom_Z += zoom_speed;
		if (line_width < 2.5){
			line_width += 0.1;
		}
	}
	else if (updateDis < preDis)//narrow
	{
		zoom_Z -= zoom_speed;
		if (line_width > 0.1){
			line_width -= 0.1;
		}
	}

	preDis = updateDis;

	//update();
}
bool Viewer::read(char *&d){
	for (; *buf&&*buf == ' ' || *buf == '\n'; buf++);
	if (*buf == '\0' || *buf == EOF) return false;
	d = buf;
	for (; *buf != ' '&&*buf != '\n'&&*buf != '\0'&&*buf != EOF; buf++);
	*buf = '\0'; buf++;
	return true;
}

dsim Viewer::min_dsim(Point a)
{
	dsim b;
	if (a.N == 1)
	{
		if (a.data.empty())
		{
			b.nj = -1;

			b.term1 = 3 * setdSef* min(dim_x, dim_y, dim_z);
			b.term2 = 3 * setdSef* min(dim_x, dim_y, dim_z);
			return b;
		}
		else
		{

			float  min_term1 = 9999999, term2;
			int nj;
			for (int i = 0; i < a.data.size(); i++)
			{
				if (a.data[i].term1 < min_term1)
				{
					min_term1 = a.data[i].term1;
					term2 = a.data[i].term2;
					nj = a.data[i].nj;
				}
			}
			b.nj = nj;
			b.term1 = min_term1;
			b.term2 = term2;

			return b;
		}
	}
	else
	{
		b.nj = -1;
		b.term1 = 0;
		b.term2 = 0;
		return b;
	}

}

int Viewer::readBenddata(vector<Line> &streamlines0, const char*data)
{
	ifstream inf;
	inf.open(data);

	int Nlines;
	inf >> Nlines;

	streamlines0.resize(Nlines);

	int sum_points = 0;
	vector<int> length_lines;
	length_lines.resize(Nlines);

	for (int i = 0; i < Nlines; i++)
	{
		int Npoint;
		inf >> Npoint;
		length_lines[i] = Npoint;

		sum_points += Npoint;
	}
	cout << sum_points << endl;

	int all_points = 0;
	for (int i = 0; i < Nlines; i++)
	{
		for (int j = 0; j < length_lines[i]; j++)
		{

			double x, y, z;
			inf >> x >> y >> z;

			if (x > max_x)
				max_x = x;
			if (x < min_x)
				min_x = x;
			if (y>max_y)
				max_y = y;
			if (y < min_y)
				min_y = y;
			if (z>max_z)
				max_z = z;
			if (z < min_z)
				min_z = z;

			Vector3 point;
			point.x = x;
			point.y = y;
			point.z = z;
			streamlines0[i].path.push_back(point);



			all_points++;
		}
	}

	cout << all_points << endl;

	inf.close();

	//设置参数
	dim_x = (int)(max_x - min_x);
	dim_y = (int)(max_y - min_y);
	dim_z = (int)(max_z - min_z);

	//alpha为方差项权重 取0～1
	g_param.alpha = afasim;
	g_param.dSep = setdSef* min(dim_x, dim_y, dim_z);
	g_param.w = 2.0f *0.01*min(dim_x, dim_y);
	int WLENTH = g_param.w / Steplenghth;

	//太小可能做不出图
	g_param.dSelfsep = 0.001* min(dim_x, dim_y, dim_z);
	g_param.dMin = 10.0f*g_param.dSelfsep;

	//设置最短线
	g_param.minLen = 5.0 * g_param.w;
	g_param.maxLen = 30000 * g_param.minLen;
	g_param.maxSize = 5000000000000;
	g_param.nHalfSample = HalfSample;

	//数一共输入的线数
	int sum = 0;
	for (int i = 0; i < streamlines0.size(); i++)
	{
		sum = sum + streamlines0[i].path.size();
	}
	cout << "number of all lines: " << streamlines0.size() << "   points: " << sum << endl;
	return streamlines0.size();

}

int Viewer::readGuoningdata(vector<Line> &streamlines0, const char*data)
{

	std::ifstream inf(data, std::ios::in);

	streamlines0.resize(50000);
	char  line[50000] = { 0 };

	int N_line = 0;
	while (inf.getline(line, sizeof(line)) )
	{
		std::stringstream word(line);

		while (word)
		{
			float x, y, z;
			word >> x;
			word >> y;
			word >> z;

			if (x > max_x)
				max_x = x;
			if (x < min_x)
				min_x = x;
			if (y>max_y)
				max_y = y;
			if (y < min_y)
				min_y = y;
			if (z>max_z)
				max_z = z;
			if (z < min_z)
				min_z = z;

			Vector3 p(x, y, z);
			streamlines0[N_line].path.push_back(p);
		}

		N_line++;
	}

	streamlines0.resize(N_line);
	inf.close();


	//设置参数
	dim_x = (int)(max_x - min_x);
	dim_y = (int)(max_y - min_y);
	dim_z = (int)(max_z - min_z);


	//alpha为方差项权重 取0～1
	g_param.alpha = afasim;
	g_param.dSep = setdSef* min(dim_x, dim_y, dim_z);
	g_param.w = 2.0f *0.01*min(dim_x, dim_y);
	int WLENTH = g_param.w / Steplenghth;

	//太小可能做不出图
	g_param.dSelfsep = 0.001* min(dim_x, dim_y, dim_z);
	g_param.dMin = 10.0f*g_param.dSelfsep;

	//设置最短线
	g_param.minLen = 5.0 * g_param.w;
	g_param.maxLen = 30000 * g_param.minLen;
	g_param.maxSize = 5000000000000;
	g_param.nHalfSample = HalfSample;

	//数一共输入的线数
	int sum = 0;
	for (int i = 0; i < streamlines0.size(); i++)
	{
		sum = sum + streamlines0[i].path.size();
	}
	cout << "number of all lines: " << streamlines0.size() << "      points: " << sum << endl;
	return N_line;
}

void  Viewer::makeData2()
{

	float ds;
	VectorField2D field;
	SimTester SimTester;

	float x, y, z;
	float value = 0;

	int num = 3000;
	char *s;
	int n = 0;
	int N = 0;

	vector<Line> streamlines0;
	
	int N_lines = readGuoningdata(streamlines0, "data/data/streamlines_cylinder_18432/streamlines_cylinder_18432.txt");
	//int N_lines = readBenddata(streamlines0, "data/data/oneDrive_data/brain1.txt");
	qDebug() << "read end";
	int N_new = 0;
	streamlines.resize(50000);

	for (int i = 0; i < streamlines0.size(); i++)
	{
		//if (streamlines0[i].getLength() >= min_length* min(dim_x, dim_y, dim_z))
		if (streamlines0[i].getLength() >= 0)
		{
			streamlines[N_new].path.resize(streamlines0[i].path.size());
			for (int j = 0; j < streamlines0[i].path.size(); j++)
			{
				streamlines[N_new].path.at(j).x = streamlines0[i].path.at(j).x;
				streamlines[N_new].path.at(j).y = streamlines0[i].path.at(j).y;
				streamlines[N_new].path.at(j).z = streamlines0[i].path.at(j).z;
			}
			N_new++;
		}
	}
	streamlines.resize(N_new);

	//start Replentish
	qDebug() << "new thread";
	qRegisterMetaType<vector<int>>("vector<int>");
	calLineOrderThread = new CalcLineOrderThread(streamlines);
	connect(this, SIGNAL(sendSelectedLinesIndex(vector<int>, vector<int>)), calLineOrderThread, SLOT(calcCurrLineOrder(vector<int>, vector<int>)));
	connect(calLineOrderThread, SIGNAL(sendLineOrder(vector<int>, int)), this, SLOT(receiveLineOrder(vector<int>, int)));
	connect(calLineOrderThread, SIGNAL(sendParameters(vector<int>*, int*, int*)), this, SLOT(getParameters(vector<int>*, int*, int*)));
	connect(calLineOrderThread, SIGNAL(sendDevicePointers(void*)), this, SLOT(getDevicePointers(void *)));
	calLineOrderThread->start();
	qDebug() << "continue";

	int N_LINES = streamlines.size();
	lineCount = N_LINES;
	first = new int[lineCount];
	vertCount = new int[lineCount];
	oneline_first = new int[1];

	lineCount_deleted_lines = 0;
	first_deleted_lines = new int[lineCount_deleted_lines];
	vertCount_deleted_lines = new int[lineCount_deleted_lines];
	oneline_vertCount = new int[1];
	int I = 0;
	int Icore = 0;
	totalSize = 0;
	vector<float> point;

	for (int i = 0; i < streamlines.size(); i++)
	{

		vertCount[I] = streamlines[i].path.size();
		first[I] = totalSize;
		I++;
		totalSize += streamlines[i].path.size();

		for (int j = 0; j < streamlines[i].path.size(); j++)
		{
			point.push_back((float)streamlines[i].path.at(j).x);
			point.push_back((float)streamlines[i].path.at(j).y);
			point.push_back((float)streamlines[i].path.at(j).z);
		}
	}

	vertices = new float[3 * totalSize];
	for (int i = 0; i < 3 * totalSize; i++)
	{
		vertices[i] = point.at(i);
	}
	colors = new float[4 * totalSize];
	N_p = totalSize;

	/*QString str_t = "";
	for (int i = 0; i < lineOrder_global.size(); i++)
	{
	str_t.append(QString::number(lineOrder_global[i]) + ",");
	}
	qDebug() << str_t;*/

	deleted_lines_arr = new bool[streamlines.size()];
	fill(deleted_lines_arr, deleted_lines_arr + streamlines.size(), 0);
}
void Viewer::readfromBin(int &N) {
	FILE *fp;
	fopen_s(&fp, "opt.txt", "r");
	int *i_data = new int;
	float *f_data = new float[2];
	//fread_s(i_data, sizeof(int), sizeof(int), 1, fp);
	//int offset = sizeof(int);
	//N = *i_data;
	fscanf_s(fp, "%d",&N);
	streamlines.resize(N);
	//Vector2 *buf = new Vector2[6553600];
	auto _Data = streamlines.data();
	max_y = max_z = max_x = INT_MIN;
	min_x = min_y = min_z = INT_MAX;
	for (int i = 0; i < N; i++) {
	//	fread_s(i_data, sizeof(int), sizeof(int), 1, fp);
		fscanf_s(fp, "%d", i_data);
		//_Data[i].path.resize(*i_data);
		//auto path_data = _Data[i].path._Get_data()
		if (*i_data > 6550000) {
			cout << "buffer overrun " << *i_data << endl;
			exit(58);
		}
		for (int j = 0; j < *i_data; j++)
		{
			//fread_s(f_data, 2*sizeof(float), sizeof(float), 2, fp);
			fscanf_s(fp, "%f %f", f_data, f_data + 1);
			f_data[0] /= 20.f;
			f_data[1] /= 20.f;
			max_x = max(max_x, f_data[0]);
			max_y = max(max_y, f_data[1]);
			float z = rand() % 20 - 10;
			max_z = max(max_z, z);

			min_x = min(min_x, f_data[0]);
			min_y = min(min_y, f_data[1]);
			min_z = min(min_z, z);


			_Data[i].path.push_back(Vector3((double)f_data[0], (double)f_data[1], 0));
			
		}
		//
		//_Data[i].path.insert(_Data[i].path.begin(), buf, buf + *i_data);

	}
//	delete[] buf;
	fclose(fp);
}   

void Viewer::readTrajectoryData() {
	FILE *fp; 
	fopen_s(&fp, "D:/flow_data/ParisFewHours.txt", "r");
	int hh, mm;
	float ss;
	float x, y, z;
	int d, l;
	int ch;
	streamlines.resize(10000);
	auto lines = streamlines.data();
	max_y = max_x = INT_MIN;
	min_x = min_y = INT_MAX;
	max_z = INT_MIN; min_z = INT_MAX;
	int nlines = 0;
	while ((ch = fgetc(fp)) != EOF && ch != '\n');
	int last = -1;
	const int skip = 1;
	int i = 0;
	while (fscanf_s(fp, "%d;%d:%d:%f;%f;%f;%f;", &l, &hh, &mm, &ss, &x, &y, &z) != EOF) {
		while ((ch = fgetc(fp)) != EOF && ch != '\n'); //skipping to the EOL
		x /= 100;
		y /= 100;
		z /= 100;
		if (last != l) {
			i = 0;
			last = l;
			nlines++;
		}
		if (nlines > 9999)
		{
			nlines--;
			break;
		}
		if (nlines < skip)
			continue;

#ifdef DOWNSAMPLE		
		if (!(i % 5))
#endif
		{
			max_x = max(max_x, x);
			max_y = max(max_y, y);
			max_z = max(max_z, z);
			min_x = min(min_x, x);
			min_y = min(min_y, y);
			min_z = min(min_z, z);

			lines[nlines - skip].path.push_back(vec3(x, y, z));
		}
		i++;
	}


	streamlines.resize(nlines - skip + 1);
	if (min_x < 0) {
		for (int i = 0; i < streamlines.size(); i++)
		{
			for (int j = 0; j < streamlines[i].path.size(); j++)
				streamlines[i].path[j].x -= min_x;
		}
		_orig_minx = min_x;
		min_x = 0;
		max_x -= min_x;
	}
	//if (min_y < 0) {
	//	for (int i = 0; i < streamlines.size(); i++)
	//	{
	//		for (int j = 0; j < streamlines[i].path.size(); j++)
	//			streamlines[i].path[j].y -= min_y;
	//	}
	//	_orig_miny = min_y;
	//	min_y = 0;
	//	max_y -= min_y;
	//}
	//if (min_z < 0) {
	//	for (int i = 0; i < streamlines.size(); i++)
	//	{
	//		for (int j = 0; j < streamlines[i].path.size(); j++)
	//			streamlines[i].path[j].z -= min_z;
	//	}
	//	_orig_minz = min_z;
	//	min_z = 0;
	//	max_z -= min_z;
	//}
	fclose(fp);
}


void Viewer::readFromTxt() {
	FILE *fp;
	fopen_s(&fp, "D:/flow_data/US-unbunbledClustered.txt", "r");
	float a, b, c;
	int d, l;
	int ch;
	streamlines.resize(10000);
	auto lines = streamlines.data();
	max_y = max_x = INT_MIN;
	min_x = min_y = INT_MAX;
	max_z = .5; min_z = -.5;
	int nlines = 0;
	while ((ch = fgetc(fp)) != EOF && ch != '\n') {

	}
	ch = -1;
	const int skip = 1;
	int i = 0;
	while (fscanf_s(fp, "%d;%f;%f;%f;%d;%d\n", &l,&a,&b,&c,&d,&d) != EOF) {
		l--;
		
		if (ch != l) {
			i = 0;
			ch = l;
			nlines++;
		}
		if (nlines > 9999)
		{
			nlines--;
			break;
		}
		if (nlines < skip)
			continue;
#ifdef DOWNSAMPLE		
		if (!(i % 5))
#endif
		{
			max_x = max(max_x, a);
			max_y = max(max_y, b);

			min_x = min(min_x, a);
			min_y = min(min_y, b);
			lines[l - skip + 1].path.push_back(vec3(a, b, 0));
		}
		i++;
	}
	

	streamlines.resize(nlines - skip+ 1);
	if (min_x < 0) {
		for (int i = 0; i < streamlines.size(); i++)
		{
			for (int j = 0; j < streamlines[i].path.size(); j++)
				streamlines[i].path[j].x -= min_x;
		}
		_orig_minx = min_x;
		min_x = 0;
		max_x -= min_x;
	}
	if (min_y < 0) {
		for (int i = 0; i < streamlines.size(); i++)
		{
			for (int j = 0; j < streamlines[i].path.size(); j++)
				streamlines[i].path[j].y -= min_y;
		}
		_orig_miny = min_y;
		min_y = 0;
		max_y -= min_y;
	}
	fclose(fp);
	
}

const char* configlist[] = { "reset", "" };
const int expression[] = {0, 1};
void ReadConfigFiles() {

	ifstream stream("config.cfg");
	char *buf = new char[65536];
	int buf_cnt = 65500;



	while (!stream.eof())
	{
		stream.getline(buf, buf_cnt);
		int len = strlen(buf);
		int i = 0;
		while (i < len&& buf[i++] != '=');

	}
}
void parse(char* str) {
		
}
struct CtrlParams {
	bool ret : 1, stop : 1;
};
DWORD WINAPI consoleCtrl(LPVOID *param) {
	bool *stop = (bool*)param;
	char* str = new char[65536];
	while (*stop) {
		cin.getline(str, 65500);
		parse(str);
	}
	delete[] str;
	return 0;
}
void  Viewer::makeData()
{
	string datapath = "d:/flow_data/bsldatanormalized/";
	FILE *stream = 0;
	char _buffer[32767];
	while (!stream) {
		string dataset = "aneurysm.bsl";
		std::cin.getline(_buffer, 32766);
		if (strlen(_buffer) > 0 && strlen(_buffer) < 32766)
			dataset = _buffer;
		if (dataset == "")
		{
			dataset = "aneurysm.bsl";
		}
		datapath += dataset;
		stream = fopen(datapath.c_str(), "rb");
		if (!stream) {
			datapath += ".bsl";
			stream = fopen(datapath.c_str(), "rb");
			if (stream)
				break;
		}
		else break;
		datapath = "d:/flow_data/bsldatanormalized/";
	}
	fclose(stream);


	HINSTANCE similarity_dll = LoadLibrary(L"UIBridge.dll");

	void(_stdcall *transfer)(Communicator*) =
		(void(*)(Communicator*)) GetProcAddress(similarity_dll, "transfer");

	printf("Proc %x in Module %x Loaded with error code: %d\n", transfer, similarity_dll, GetLastError());

	Communicator *comm = new Communicator();
	/* Setting up paras*/
	
	comm->filename = datapath.c_str();// "d:/flow_data/2.obj";

	transfer(comm);
	printf("Transferred pointer: %x\n", comm->f_streamlines);
	float **f_streamlines = comm ->f_streamlines;
	int *i_colors = comm->colors;
	float *alpha = comm->alpha;
	//N = ;
	Vector3 _max (INT_MIN, INT_MIN, INT_MIN), _min(INT_MAX, INT_MAX, INT_MAX);
	for (int i = 0; i < comm->n_streamlines; i++) {
		streamlines.push_back(Line());
		for (int j = 0; j < comm->sizes[i]; j++)
		{
			streamlines.back().path.push_back(Vector3());
			for (int _dim = 0; _dim < 3; _dim++)
			{
				streamlines.back().path[j][_dim] = f_streamlines[i][j * 3 + _dim];
				_min[_dim] = __macro_min(f_streamlines[i][j * 3 + _dim], _min[_dim]);
				_max[_dim] = __macro_max(f_streamlines[i][j * 3 + _dim], _max[_dim]);
			}
		}
	}

	/*dim_x = (int)(_max[0] - _min[0]);
	dim_y = (int)(_max[1] - _min[1]);
	dim_z = (int)(_max[2] - _min[2]);
*/
	max_x = _max.x;
	max_y = _max.y;
	max_z = _max.z;
	min_x = _min.x;
	min_y = _min.y;
	min_z = _min.z;



	float ds;
	VectorField2D field;
	SimTester SimTester;

	float x, y, z;
	float value = 0;

	int num = 10000;
	char *s;
	int n = 0;
	int N = 0;
	
	
	
	goto skip;

	//readfromBin(N);
#ifdef TXT
	readFromTxt();
	//readTrajectoryData();
	goto skip;
#else
	vector<Line> streamlines0;
	streamlines0.resize(20000);
	streamlines0[N].path.resize(num);
#endif

//	FILE *stream = 0;
//	char _buffer[32767];
	while (!stream) {
		string dataset = "aneurysm.obj";
		std::cin.getline(_buffer, 32766);
		if (strlen(_buffer) > 0 && strlen(_buffer) < 32766)
			dataset = _buffer;
		if (dataset == "")
		{
			dataset = "aneurysm.obj";
		}
		datapath += dataset;
		stream = fopen(datapath.c_str(), "rb");
		if (!stream) {
			datapath += ".obj";
			stream = fopen(datapath.c_str(), "rb");
		}
		datapath = "d:/flow_data/";
	}
	fread(Buf, 1, BUFSIZE, stream);
	std::cin.getline(_buffer, 32766);
	int N_offset = 0;

	if (strlen(_buffer) < 0 && strlen(_buffer) > 10)
		N_offset = 0;
	N_offset = atoi(_buffer);
	if (N_offset <= 0) {
		N_offset = 0;
	}

	N = 0;
	while (N++ < N_offset)
	{
		if (!read(s)) break;//s being the return of index

		if (*(s + 1) == '\0') {
			if (*s == 'v')
			{
				read(s); x = atof(s);
				read(s); y = atof(s);
				read(s); z = atof(s);
			}
		}
	} 
//	N = 0;
	N--;
	std::cin.getline(_buffer, 32766);
	int N_max = 0;

	if (strlen(_buffer) < 0 && strlen(_buffer) > 10)
		N_max = 0;
	N_max = atoi(_buffer);
	if (N_max <= 0) {
		N_max = INT_MAX;
	}
	//fread(Buf, 1, BUFSIZE, stream);
	//while (N++ < 200){
	//	if (!read(s)) break;

	//	if (*(s + 1) == '\0'){
	//		if (*s == 'v')
	//		{
	//			read(s); x = atof(s);
	//			read(s); y = atof(s);
	//			read(s); z = atof(s);
	//		}
	//	}   
	//}
	//N = 0;
	
	N_max -= N_offset;
	N = 0;
	
#ifndef TXT

	streamlines0[N].path.resize(num);
	int _n = 0;
	while (true&&N<N_max)
	{
		if (!read(s)) break;//s being the return of index
		
		if (*(s + 1) == '\0'){
			if (*s == 'v')
			{
				read(s); x = atof(s);
				read(s); y = atof(s);
				read(s); z = atof(s);

				if (x > max_x)
					max_x = x;
				if (x < min_x)
					min_x = x;
				if (y > max_y)
					max_y = y;
				if (y < min_y)
					min_y = y;
				if (z > max_z)
					max_z = z;
				if (z < min_z)
					min_z = z;

				streamlines0[N].path.at(n).x = x;
				streamlines0[N].path.at(n).y = y;
				streamlines0[N].path.at(n).z = z;
#ifdef DOWNSAMPLE
				if (!(_n % 20))
					n++;
				_n++;
#else
				n++;
#endif // DOWNSAMPLE
					

				
				if (n > 1)
				{
					ds = length(streamlines0[N].path.at(n - 1) - streamlines0[N].path.at(n - 2));
					if (ds > max_ds)
						max_ds = ds;
				}

			}
			else if (*s == 'g' || *s == '#'){

				streamlines0[N].path.resize(n);
				N = N + 1;
				streamlines0[N].path.resize(num);
				n = 0;
			}
		}
	}
	streamlines0.resize(N);
	Vector3 vmin(min(min_x, 0), min(min_y, 0), min(min_z, 0));
	
	min_x -= vmin.x;
	min_y -= vmin.y;
	min_z -= vmin.z;
	max_x -= vmin.x;
	max_y -= vmin.y;
	max_z -= vmin.z;
	for (auto &line : streamlines0) 
		for (auto &vertex : line.path) 
			vertex -= vmin;

	 

#endif
skip:
	dim_x = (int)(max_x - min_x);
	dim_y = (int)(max_y - min_y);
	dim_z = (int)(max_z - min_z);
	g_param.alpha = afasim;
	g_param.dSep = setdSef* min(dim_x, dim_y, dim_z);
	g_param.w = 2.0f *0.01*min(dim_x, dim_y);
	int WLENTH = g_param.w / Steplenghth;

	g_param.dSelfsep = 0.001* min(dim_x, dim_y, dim_z);
	g_param.dMin = 10.0f*g_param.dSelfsep;
	g_param.minLen = 5.0 * g_param.w;
	g_param.maxLen = 30000 * g_param.minLen;
	g_param.maxSize = 5000000000000;
	g_param.nHalfSample = HalfSample;

	int sum = 0;
	for (int i = 0; i < streamlines.size(); i++)
		sum = sum + streamlines[i].path.size();
	
	int N_new = 0;

#ifndef TXT
	streamlines.resize(50000);

	for (int i = 0; i < streamlines0.size(); i++)
	{
		//if (streamlines0[i].getLength() >= min_length* min(dim_x, dim_y, dim_z))
		if (streamlines0[i].getLength() >= 0)
		{
			streamlines[N_new].path.resize(streamlines0[i].path.size());
			for (int j = 0; j < streamlines0[i].path.size(); j++)
			{
				//if (streamlines0[i].path.at(j).x < 0 || streamlines0[i].path.at(j).y < 0 || streamlines0[i].path.at(j).z < 0)
				//	printf("!");
				streamlines[N_new].path.at(j).x = streamlines0[i].path.at(j).x;
				streamlines[N_new].path.at(j).y = streamlines0[i].path.at(j).y;
				streamlines[N_new].path.at(j).z = streamlines0[i].path.at(j).z;
			}
			N_new++;
		}
	}
	//printf("succeed");
	streamlines.resize(N_new);

#endif
//	(MainWindow *)parent();
	//start Replentish
	this->slider->setRange(0, streamlines.size());
	qDebug() << "new thread";
	qRegisterMetaType<vector<int>>("vector<int>");
	calLineOrderThread = new CalcLineOrderThread(streamlines);
	calLineOrderThread->orig_minx = _orig_minx;
	calLineOrderThread->orig_miny = _orig_miny;	
	connect(this, SIGNAL(sendSelectedLinesIndex(vector<int>, vector<int>)), calLineOrderThread, SLOT(calcCurrLineOrder(vector<int>, vector<int>)));
	connect(calLineOrderThread, SIGNAL(sendLineOrder(vector<int>, int)), this, SLOT(receiveLineOrder(vector<int>, int)));
	connect(calLineOrderThread, SIGNAL(sendParameters(vector<int>*, int*, int*)), this, SLOT(getParameters(vector<int>*, int*, int*)));
	connect(calLineOrderThread, SIGNAL(sendDevicePointers(void *)), this, SLOT(getDevicePointers(void*)));
	calLineOrderThread->start();
	qDebug() << "continue";

	int N_LINES = streamlines.size();
	lineCount = N_LINES;
	first = new int[lineCount];
	vertCount = new int[lineCount];
	oneline_first = new int[1];

	lineCount_deleted_lines = 0;
	first_deleted_lines = new int[lineCount_deleted_lines];
	vertCount_deleted_lines = new int[lineCount_deleted_lines];
	oneline_vertCount = new int[1];
	int I = 0;
	int Icore = 0;
	totalSize = 0;
	vector<float> point;
	for (int i = 0; i < streamlines.size(); i++)
	{

		vertCount[I] = streamlines[i].path.size();
		first[I] = totalSize;
		I++;
		totalSize += streamlines[i].path.size();

		for (int j = 0; j < streamlines[i].path.size(); j++)
		{
			point.push_back((float)streamlines[i].path.at(j).x);
			point.push_back((float)streamlines[i].path.at(j).y);
			point.push_back((float)streamlines[i].path.at(j).z);
		}
	}

	vertices = new float[3 * totalSize];
	for (int i = 0; i < 3 * totalSize; i++)
	{
		vertices[i] = point.at(i);
	}
	colors = new float[4 * totalSize];
	for (int i = 0; i < totalSize; i++) {
		if (i_colors) {
			colors[i * 4 + 0] = ((i_colors[i] >> 24) & 0xff) / 255.f;
			colors[i * 4 + 1] = ((i_colors[i] >> 16) & 0xff) / 255.f;
			colors[i * 4 + 2] = ((i_colors[i] >> 8) & 0xff) / 255.f;
		}
		else {
			colors[i * 4 + 0] = (0xff) / 255.f;
			colors[i * 4 + 1] = (0xa9) / 255.f;
			colors[i * 4 + 2] = (0x00) / 255.f;
			/*
			colors[i * 4 + 0] = (0x39) / 255.f;
			colors[i * 4 + 1] = (0xb9) / 255.f;
			colors[i * 4 + 2] = (0xf9) / 255.f;*/
		}
		if (0)
			colors[i * 4 + 3] = alpha[i];
		else 
			colors[i * 4 + 3] = 1.f;
	}
	N_p = totalSize;
	printf("Points on lines %d\n", N_p);
	/*QString str_t = "";
	for (int i = 0; i < lineOrder_global.size(); i++)
	{
		str_t.append(QString::number(lineOrder_global[i]) + ",");
	}
	qDebug() << str_t;*/

	deleted_lines_arr = new bool[streamlines.size()];
	fill(deleted_lines_arr, deleted_lines_arr + streamlines.size(), 0);

	line_width = getLineWidth(streamlines.size());
}
void Viewer::makeConfigFile()
{

	cameraPosition.x = (max_x + min_x) / 2;
	cameraPosition.y = (max_y + min_y) / 2;
	cameraPosition.z = max_z * 5;

	sceneCenter.x = (max_x + min_x) / 2;
	sceneCenter.y = (max_y + min_y) / 2;
	sceneCenter.z = (max_z + min_z) / 2;

//	emit sendPosZSlider(cameraPosition.z);

	cameraUp.x = 0;
	cameraUp.y = 1;
	cameraUp.z = 0;

	cameraPerspective[0] = 60;//foxy
	cameraPerspective[1] = 1;//aspect
	cameraPerspective[2] = .1f;
	//cameraPerspective[3] = max_z * 10;//far
	qDebug() << "cameraPerspective:" << cameraPerspective[0] << "," << cameraPerspective[1] << "," << cameraPerspective[2] << "," << cameraPerspective[3] << ",dim_x:" << dim_x;

	lightPosition[0] = cameraPosition.x;
	lightPosition[1] = max_y;
	lightPosition[2] = cameraPosition.z;

	lightDirection[0] = sceneCenter.x - lightPosition[0];
	lightDirection[1] = sceneCenter.y - lightPosition[1];
	lightDirection[2] = sceneCenter.z - lightPosition[2];

	light_x = lightPosition[0];
	light_y = lightPosition[1];
	light_z = lightPosition[2];
	qDebug() << "lightPosition:" << lightPosition[0] << "," << lightPosition[1] << "," << lightPosition[2];

	ka = 0.05;
	kd = 0.8;
	ks = 1;
	gloss = 10;

	uniformColor[0] = (float)255 / 255;
	uniformColor[1] = (float)215 / 255;
	uniformColor[2] = (float)0 / 255;
	uniformColor[3] = 0.03;

	color_r = uniformColor[0];
	color_g = uniformColor[1];
	color_b = uniformColor[2];

	dataHasColors = true;

}

bool isExtensionSupported(const char *extension)
{
	char	*extString, *match;

	extString = (char *)glGetString(GL_EXTENSIONS);
	if (extString == NULL)
		return (false);

	match = strstr(extString, extension);
	if (match == NULL)
		return (false);

	if (match[strlen(extension)] != ' ')
		return (false);
	if ((match == extString) || (match[-1] == ' '))
		return (true);
	return (false);
}

void Viewer::initGLExtensions()
{
	extVertexBufferObject = isExtensionSupported("GL_ARB_vertex_buffer_object");
	extMultiDrawArrays = isExtensionSupported("GL_EXT_multi_draw_arrays");

	if (extVertexBufferObject)
		cout << "Extensions : GL_ARB_vertex_buffer_object supported." << endl;
	else
		cout << "Extensions : GL_ARB_vertex_buffer_object NOT supported." << endl;
	if (extMultiDrawArrays)
		cout << "Extensions : GL_ARB_multi_draw_arrays supported." << endl;
	else
		cout << "Extensions : GL_ARB_multi_draw_arrays NOT supported." << endl;

	pglGenBuffers = (PFNGLGENBUFFERSPROC)wglXGetProcAddress("glGenBuffersARB");
	pglBindBuffer = (PFNGLBINDBUFFERPROC)wglXGetProcAddress("glBindBufferARB");
	pglBufferData = (PFNGLBUFFERDATAPROC)wglXGetProcAddress("glBufferDataARB");
	pglMultiDrawArrays = (PFNGLMULTIDRAWARRAYSPROC)wglXGetProcAddress("glMultiDrawArraysEXT");
}
void Viewer::initGL()
{
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glEnable(GL_DEPTH_TEST);

	initGLExtensions();

	if (extVertexBufferObject)
	{
		pglGenBuffers(1, &vboVertID);
		pglBindBuffer(GL_ARRAY_BUFFER, vboVertID);
		pglBufferData(GL_ARRAY_BUFFER, 3 * sizeof(vertices[0]) * totalSize, vertices, GL_STATIC_DRAW);

		if (dataHasColors)
		{
			pglGenBuffers(1, &vboColID);
			pglBindBuffer(GL_ARRAY_BUFFER, vboColID);
			pglBufferData(GL_ARRAY_BUFFER, 4 * sizeof(colors[0]) * totalSize, colors, GL_STATIC_DRAW);
		}
	}
}

void Viewer::cleanup()
{
	ILRender::deleteIdentifier(ilID_solid_lines);
	ILRender::deleteIdentifier(ilID_deleted_lines);
	 
	delete[] vertices;
	delete[] colors;
}
void Viewer::die()
{
	cleanup();
	//system("pause");
	exit(1);
}
void Viewer::repaint()
{
	//ILines::ILUtilities::repaint_pending++;
	printf("repaint0");
	QOpenGLWidget::repaint();
	//ILines::ILUtilities::repaint_pending--;
}
void Viewer::update() {
	//ILines::ILUtilities::repaint_pending++;
	QOpenGLWidget::update();
}
void Viewer::errorCallbackIL(ILRender *ilRender)
{
	ILRender::ILError err;

	err = ilRender->getError();

	cout << "IL Error : " << ILRender::errorString(err) << endl;

	if (err == ILRender::IL_GL_ERROR)
		cout << "GL Error : " << gluErrorString(ilRender->getGLError()) << endl;

	die();
}
void Viewer::initIL()
{
	maximumPhongSupported = ILRender::isLightingModelSupported(ILLightingModel::IL_MAXIMUM_PHONG);
	cylinderBlinnSupported = ILRender::isLightingModelSupported(ILLightingModel::IL_CYLINDER_BLINN);
	cylinderPhongSupported = ILRender::isLightingModelSupported(ILLightingModel::IL_CYLINDER_PHONG);
/*
	maximumPhongIL.setErrorCallback(errorCallbackIL);
	cylinderBlinnIL.setErrorCallback(errorCallbackIL);
	cylinderPhongIL.setErrorCallback(errorCallbackIL);*/

	if (maximumPhongSupported)
	{
		maximumPhongIL.setupTextures(ka, 0.6f * kd, 0.3f * ks, gloss, texDim,
			ILLightingModel::IL_MAXIMUM_PHONG, false,
			lightDirection);
	}

	if (cylinderBlinnSupported)
	{
		cylinderBlinnIL.setupTextures(ka, kd, ks, 4.0f * gloss, texDim,
			ILLightingModel::IL_CYLINDER_BLINN, false);
	}

	if (cylinderPhongSupported)
	{
		cylinderPhongIL.setupTextures(ka, kd, ks, gloss, texDim,
			ILLightingModel::IL_CYLINDER_PHONG, false,
			lightDirection);
	}

	if (cylinderPhongSupported)
	{
		curIL = &cylinderPhongIL;
		lightingModel = ILLightingModel::IL_CYLINDER_PHONG;
	}
	else if (cylinderBlinnSupported)
	{
		curIL = &cylinderBlinnIL;
		lightingModel = ILLightingModel::IL_CYLINDER_BLINN;
	}
	else if (maximumPhongSupported)
	{
		curIL = &maximumPhongIL;
		lightingModel = ILLightingModel::IL_MAXIMUM_PHONG;
	}

	if (curIL == NULL)
	{

		cout << "CURIL == NULL" << endl;
		die();
	}
}

void Viewer::init()
{
	initGL();
	initIL();

	Y = normalize(cameraUp);
	Z = normalize(cameraPosition - sceneCenter);
	X = normalize(cross(Y, Z));
}

GLdouble findMaxZ(vector<GLdouble> a)
{ //求向量最大值  
	GLdouble maxdata = a[0];
	int len = a.size(), i;  //a.size() 求得向量当前存储数量
	for (i = 1; i<len; i++)
	{
		if (a[i]>maxdata)
			maxdata = a[i];
	}
	return maxdata;
}
GLdouble findMinZ(vector<GLdouble> a)
{ //求向量最小值  
	GLdouble mindata = a[0];
	int len = a.size(), i;
	for (i = 1; i<len; i++)
	{
		if (a[i]<mindata)
			mindata = a[i];
	}
	return mindata;
}

void drawSphere(GLfloat xx, GLfloat yy, GLfloat zz, GLfloat radius)
{
	double c = PI / 180.0;
	double a = 1.0;
	float theta, phir, phi, thetar, phir20;
	float x, y, z;
	for (phi = -90.0; phi <= 90.0; phi += a)
	{
		phir = c*phi;
		phir20 = c*(phi + a);
		glPolygonMode(GL_FRONT, GL_LINE);
		glPolygonMode(GL_BACK, GL_LINE);
		glFrontFace(GL_CCW); //逆时针为正面
		glColor4f(1, 0, 0, 0.5);
		//glColor3f(1, 0, 0);
		glBegin(GL_TRIANGLE_STRIP);
		for (theta = -180.0; theta <= 180.0; theta += a)
		{
			thetar = c*theta;
			x = radius*sin(thetar)*cos(phir);
			y = radius*cos(thetar)*cos(phir);
			z = radius*sin(phir);
			glVertex3d(xx + x, yy + y, zz + z);
			//glVertex3f(x,y,z);
			x = radius*sin(thetar)*cos(phir20);
			y = radius*cos(thetar)*cos(phir20);
			z = radius*sin(phir20);
			glVertex3d(xx + x, yy + y, zz + z);
			//glVertex3f(x,y,z);
		}

		glEnd();
	}
}
/*
*draw cylinder
*/
GLvoid DrawCircleArea(float cx, float cy, float cz, float r, int num_segments)
{
	GLfloat vertex[4];
	int M_PI = 3.1415926;
	const GLfloat delta_angle = 2.0*M_PI / num_segments;
	glBegin(GL_TRIANGLE_FAN);

	vertex[0] = cx;
	vertex[1] = cy;
	vertex[2] = cz;
	vertex[3] = 1.0;
	glVertex4fv(vertex);

	//draw the vertex on the contour of the circle   
	for (int i = 0; i < num_segments; i++)
	{
		vertex[0] = std::cos(delta_angle*i) * r + cx;
		vertex[1] = std::sin(delta_angle*i) * r + cy;
		vertex[2] = cz;
		vertex[3] = 1.0;
		glVertex4fv(vertex);
	}

	vertex[0] = 1.0 * r + cx;
	vertex[1] = 0.0 * r + cy;
	vertex[2] = cz;
	vertex[3] = 1.0;
	glVertex4fv(vertex);
	glEnd();
}
void RenderBone(float x0, float y0, float z0, float x1, float y1, float z1)
{
	GLdouble  dir_x = x1 - x0;
	GLdouble  dir_y = y1 - y0;
	GLdouble  dir_z = z1 - z0;
	GLdouble  bone_length = sqrt(dir_x*dir_x + dir_y*dir_y + dir_z*dir_z);
	static GLUquadricObj *  quad_obj = NULL;
	if (quad_obj == NULL)
		quad_obj = gluNewQuadric();
	gluQuadricDrawStyle(quad_obj, GLU_FILL);
	gluQuadricNormals(quad_obj, GLU_SMOOTH);
	glPushMatrix();
	// 平移到起始点  
	glTranslated(x0, y0, z0);
	// 计算长度  
	double  length;
	length = sqrt(dir_x*dir_x + dir_y*dir_y + dir_z*dir_z);
	if (length < 0.0001) {
		dir_x = 0.0; dir_y = 0.0; dir_z = 1.0;  length = 1.0;
	}
	dir_x /= length;  dir_y /= length;  dir_z /= length;
	GLdouble  up_x, up_y, up_z;
	up_x = 0.0;
	up_y = 1.0;
	up_z = 0.0;
	double  side_x, side_y, side_z;
	side_x = up_y * dir_z - up_z * dir_y;
	side_y = up_z * dir_x - up_x * dir_z;
	side_z = up_x * dir_y - up_y * dir_x;
	length = sqrt(side_x*side_x + side_y*side_y + side_z*side_z);
	if (length < 0.0001) {
		side_x = 1.0; side_y = 0.0; side_z = 0.0;  length = 1.0;
	}
	side_x /= length;  side_y /= length;  side_z /= length;
	up_x = dir_y * side_z - dir_z * side_y;
	up_y = dir_z * side_x - dir_x * side_z;
	up_z = dir_x * side_y - dir_y * side_x;
	// 计算变换矩阵  
	GLdouble  m[16] = { side_x, side_y, side_z, 0.0,
		up_x,   up_y,   up_z,   0.0,
		dir_x,  dir_y,  dir_z,  0.0,
		0.0,    0.0,    0.0,    1.0 };
	glMultMatrixd(m);
	// 圆柱体参数  
	GLdouble radius = 0.05;        // 半径  
	GLdouble slices = 32;      //  段数  
	GLdouble stack = 32.0;       // 递归次数  
	DrawCircleArea(x0, y0, z0, radius, slices);
	DrawCircleArea(x1, y1, z1, radius, slices);
	gluCylinder(quad_obj, radius, radius, bone_length, slices, stack);
	glPopMatrix();
}
bool texSet = false;
void Viewer::setLight() {
	if (!texSet) {
		cylinderBlinnIL.setupTextures(ka, kd, ks, 4.0f * gloss, texDim,
			ILLightingModel::IL_CYLINDER_BLINN, false);
		curIL = &cylinderBlinnIL;
		texSet = true;
	}
	lightingModel = ILLightingModel::IL_CYLINDER_BLINN;

	glEnable(GL_DEPTH_TEST);
	glShadeModel(GL_SMOOTH);
	glEnable(GL_LIGHT0);
	//set light component
	//set global ambient 
	GLfloat gAmbient[] = { 0.4, 0,4, 0,4, 1.0 };
	GLfloat local_view[] = { 0.0 };
	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, gAmbient);
	glLightModelfv(GL_LIGHT_MODEL_LOCAL_VIEWER, local_view);

	lightDirection[0] = sceneCenter.x + translateX - lightPosition[0];
	lightDirection[1] = sceneCenter.y + translateY - lightPosition[1];
	lightDirection[2] = sceneCenter.z + translateZ - lightPosition[2];
	//set light postion and direction
	if (light_mode == 0) {
		GLfloat lightPosition_parall[] = { sceneCenter.x + translateX - lightPosition[0], sceneCenter.y + translateY - lightPosition[1], sceneCenter.z + translateZ - lightPosition[2] ,0 };
		glLightfv(GL_LIGHT0, GL_POSITION, lightPosition_parall);
	}
	else if (light_mode == 1) {
		glLightfv(GL_LIGHT0, GL_POSITION, lightPosition);
		// 平行光不会随着距离d增加而衰减，但点光源和聚光灯会发生衰减。  
		glLightf(GL_LIGHT0, GL_CONSTANT_ATTENUATION, 2.0);     // c 系数
		glLightf(GL_LIGHT0, GL_LINEAR_ATTENUATION, 1.0);        // l 系数
		glLightf(GL_LIGHT0, GL_QUADRATIC_ATTENUATION, 0.5);    // q 系数
	}	

	//printLightParameters();
}
void Viewer::display()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	float	mvMatrix[16];
	float	fps;

	glGetFloatv(GL_MODELVIEW_MATRIX, mvMatrix);
	if (rec) {
		rec = false;
		for (int i = 0; i < 16; i++)
			fprintf_s(cfg, "%f ", mvMatrix[i]);
		fclose(cfg);
		cfg = 0;
	}/* Handle rotations and translations separately. */
	else if (res) {
		res = false;
		for (int i = 0; i < 16; i++)
			fscanf_s(cfg, "%f ", &mvMatrix[i]);
		fclose(cfg);
		cfg = 0;
	}
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glRotatef(rotate_y, Y.x, Y.y, Y.z);
	glRotatef(rotate_x, X.x, X.y, X.z);
	glMultMatrixf(mvMatrix);

	/* Save the rotational component of the modelview matrix. */
	glPushMatrix();
	glLoadIdentity();

	//in world coordinate
	gluLookAt(cameraPosition.x, cameraPosition.y, cameraPosition.z,
		sceneCenter.x + translateX, sceneCenter.y + translateY, sceneCenter.z + translateZ,
		cameraUp.x, cameraUp.y, cameraUp.z);

	glColor3f(1, 1, 0);
	glPointSize(10);
	glBegin(GL_POINTS);
	glVertex3f(lightPosition[0], lightPosition[1], lightPosition[2]);
	glEnd();
	glPointSize(1);
	//glBegin(GL_LINES);
	//glVertex3f(lightPosition[0], lightPosition[1], lightPosition[2]);
	//glVertex3f(sceneCenter.x + translateX, sceneCenter.y + translateY, sceneCenter.z + translateZ);
	//glEnd();
	
	glEnable(GL_LIGHTING); //在后面的渲染中使用光照
	//set lighting
	setLight();

	Vector3f trans;
	trans = zoom_Z / 20.0f * Z + transX / 150.0f * X - transY / 150.0f * Y;
	glTranslatef(trans.x, trans.y, trans.z);
	glTranslatef(+sceneCenter.x, +sceneCenter.y, +sceneCenter.z);

	glMultMatrixf(mvMatrix);
	glTranslatef(-sceneCenter.x, -sceneCenter.y, -sceneCenter.z);

	rotate_x = rotate_y = 0;
#ifndef DLL_INPUT
	if (doColors && is_change_mode_sign&&(CalcLineOrderThread::g_saliency||opacity_point.size()>0)){
		//set each point's opacity

		for (int i = 0; i < totalSize; i++)
		{
			if (CalcLineOrderThread::g_saliency)
			{
				colors[4 * i] = 255. / 255.;//CalcLineOrderThread::g_saliency[i];
				colors[4 * i + 1] = 215./255.;// 1 - CalcLineOrderThread::g_saliency[i];
				colors[4 * i + 2] = 0;// 1 - CalcLineOrderThread::g_saliency[i];

				colors[4 * i + 3] = /*1;//*/ CalcLineOrderThread::g_saliency[i];
			}

			else
			{
				colors[4 * i] = /*255. / 255.;//*/opacity_point[i];
				colors[4 * i + 1] = /*215./255.;//*/ 1 - opacity_point[i];
				colors[4 * i + 2] = /*0;//*/ 1 - opacity_point[i];

				colors[4 * i + 3] = 1;// opacity_point[i];
			}
		}

		pglGenBuffers(1, &vboColID);
		pglBindBuffer(GL_ARRAY_BUFFER, vboColID);
		pglBufferData(GL_ARRAY_BUFFER, 4 * sizeof(colors[0]) * totalSize, colors, GL_STATIC_DRAW);
	}
#else
	pglGenBuffers(1, &vboColID);
	pglBindBuffer(GL_ARRAY_BUFFER, vboColID);
	pglBufferData(GL_ARRAY_BUFFER, 4 * sizeof(colors[0]) * totalSize, colors, GL_STATIC_DRAW);
#endif

	//start draw the ball
	if (!doColors){
		if (is_ball_change_sign && radius > 0)
		{
			// start:calculate the depth z
			GLint viewport[4];
			GLdouble mvmatrix2[16], projmatrix[16];
			GLfloat winx, winy, winz;
			GLdouble posx = 0, posy = 0, posz = 0, posz_min = 0, posz_max = 0;
			GLdouble posx1 = 0, posy1 = 0, posz1 = 0;

			glGetIntegerv(GL_VIEWPORT, viewport);   /* 获取三个矩阵 */
			glGetDoublev(GL_MODELVIEW_MATRIX, mvmatrix2);
			glGetDoublev(GL_PROJECTION_MATRIX, projmatrix);

			winx = old_x;
			winy = widgetHeight - old_y;

			GLdouble max_depth = -10000;
			GLdouble min_depth = 999999999;
			double min_dis = 999999999;
			double max_dis = -1000000;
			//vector<double> Z_list;
			GLdouble posx_tmp = 0, posy_tmp = 0, posz_tmp = 0;
			for (int i = 0; i < totalSize; i++){
				gluProject(vertices[i * 3 + 0], vertices[i * 3 + 1], vertices[i * 3 + 2], mvmatrix2, projmatrix, viewport, &posx_tmp, &posy_tmp, &posz_tmp);
				//Z_list.push_back(posz_tmp);
				if (posz_tmp > max_depth){
					max_depth = posz_tmp;
				}
				if (posz_tmp < min_depth)
				{
					min_depth = posz_tmp;
				}

				double dis_to_winxy = (winx - posx_tmp)*(winx - posx_tmp) + (winy - posy_tmp)*(winy - posy_tmp);
				if (dis_to_winxy < min_dis){
					min_dis = dis_to_winxy;
					posz_min = posz_tmp;
				}
				if (dis_to_winxy > max_dis){
					max_dis = dis_to_winxy;
					posz_max = posz_tmp;
				}
			}

			//posz = (posz_max + posz_min) / 2;
			posz = posz_min;
			gluUnProject(winx, winy, posz, mvmatrix2, projmatrix, viewport, &posx, &posy, &posz);/* 获取 z为离视点最近点 的三维坐标 */
			circle_x = posx;
			circle_y = posy;
			circle_z = posz;
			circle_r = radius / 5;
			drawSphere(circle_x, circle_y, circle_z, circle_r);
			//deleted_lines_num = 0;
			prepareLinesToDraw(circle_x, circle_y, circle_z, circle_r);
			/*
			emit sendTotalLineNum(cur_line_num, deleted_lines_num);
			emit sendLineToDeleteNum(line_to_delete_num);
			emit sendSelectedLines(selected_lines_index_list);*/
		}
		else if (selected_one_line_index_list.size() > 0 || !is_global_sign){
			prepareLinesToDraw(circle_x, circle_y, circle_z, circle_r);
		}
		else if (finger_num == 5 && is_global_sign || is_global_sign && deleted_lines_num > 0){
			prepareLinesToDraw(0, 0, 0, 0);
			finger_num = 0;
		}
	}
	//end draw the ball

	clock_t start, finish;
	start = clock();
	displayScene();
	finish = clock();
	if (material_mode == 1) {
		glDisable(GL_COLOR_MATERIAL);
	}
	
	glDisable(GL_LIGHTING); //disable 光照


	// switch to 2D projection
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	gluOrtho2D(0, widgetWidth, 0, widgetHeight);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	string str;
	if (!doColors){
		int lineOrder_global_size = lineOrder_global.size();
		str = "Render on lines, lines already calculated num:" + to_string(lineOrder_global_size);
	}
	else
	{
		str = "Render on points";
		//render on points requires initializations avoid null ptrs
		if (0){
			str = "Can't Render on points";
			doColors = !doColors;
			update();
		}
	}
	drawString(10, 10, str);

	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	//2D end

	glFinish();
	/* Restore the rotational component of the modelview matrix. */
	glPopMatrix();
	//ILUtilities::repaint_pending--;
}

void Viewer::reshape(int width, int height)
{
	glViewport(0, 0, width, height);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	float aspect = (float)width / (float)height;
	gluPerspective(cameraPerspective[0], aspect*cameraPerspective[1],
		cameraPerspective[2], cameraPerspective[3]);
	widgetWidth = width;
	widgetHeight = height;
	this->repaint();
}
vector<int> Viewer::getSelectedLinesIndex(float r0, Vector3 pt)
{
	vector<int> ret;
	if (line_division){
		KD_Tree::KD_Vector3 kd_vec(pt.x, pt.y, pt.z, -1);
		float r = r0*r0;
		int n_lines = streamlines.size();
		for (int i = 0; i < n_lines; i++)
		{
			for each (int seg in line_division[i])
			{
				if (KD_Tree::search(kd_vec, r, linearKd + linearKD_id[seg]))
				{
					ret.push_back(i);
					break;
				}
			}
		}
	}
	return ret;
}
void Viewer::getParameters(vector<int> *line_division, int *linearKd, int *linearKD_id){
	this->line_division = line_division;
	this->linearKd = linearKd;
	this->linearKD_id = linearKD_id;
	update();
}
vector<int> Viewer::getLocalLineOrder(set<int> *local, int del) {//use binary search + qsort instead for better performance 
	vector<int> ret;
	vector<int>::iterator it = lineOrder_global.begin();
	while (it != lineOrder_global.end()) {
		if (local->find(*it) != local->end())
		{
			ret.push_back(*it);
			if (del-- <= 0)
				break;
		}
		++it;
	}
	return ret;

}
bool isInLineOrder(vector<int> line_order, int line_index, int num){
	if (line_order.size() == 0){
		return false;
	}
	for (int i = 0; i < num; i++){
		if (line_order[i] == line_index){
			return true;
		}
	}
	return false;
}
int getDeletedLinesNum(bool* arr[], int num){
	int count_deleted_lines_num = 0;
	for (int i = 0; i < num; i++){
		if ((*arr)[i]){
			count_deleted_lines_num++;
		}
	}
	return count_deleted_lines_num;
}
void Viewer::prepareLinesToDraw(double x, double y, double z, double r0)
{
	Vector3 pt = Vector3(x, y, z);
	clock_t start, finish;
	start = clock();
	selected_lines_index_list = getSelectedLinesIndex(r0, pt);
	finish = clock();
//	qDebug() << "get selected lines index time:" << (finish - start);
	deleted_lines_index_list.clear();
	for (int i = 0; i < selected_lines_index_list.size(); i++){
		if (deleted_lines_arr[selected_lines_index_list[i]]){
			deleted_lines_index_list.push_back(selected_lines_index_list[i]);
		}
	}

	if (selected_lines_index_list.size() > 0 || finger_num == 5 && is_global_sign || is_global_sign && deleted_lines_num > 0)//local delete lines,back to origin,global delete lines
	{
		if (selected_lines_index_list.size() > 0)
		{
			cur_line_num = selected_lines_index_list.size();
			is_global_sign = false;
		}

		int N_LINES = streamlines.size();
		if (is_global_sign){//global
			lineCount = N_LINES - deleted_lines_num;
			lineCount_deleted_lines = deleted_lines_num;
		}
		else{//local
			if (is_ball_change_sign) {
				lineCount = selected_lines_index_list.size();
				lineCount_deleted_lines = N_LINES - lineCount;
				//qDebug()<< is_ball_change_sign << ",,,,,,,,,,,,,,lineCount = " << lineCount << ",lineCount_deleted_lines = " << lineCount_deleted_lines;
			}
			else
			{
				int deleted_lines_num_tmp = getDeletedLinesNum(&deleted_lines_arr, streamlines.size());
				lineCount = N_LINES - deleted_lines_num_tmp;
				lineCount_deleted_lines = N_LINES - lineCount;
				//qDebug() << is_ball_change_sign << ",,,,,,,,,,,,,,lineCount = " << lineCount << ",lineCount_deleted_lines = " << lineCount_deleted_lines;
			}
		}

		if (!is_global_sign && selected_one_line_index_list.size() > 0){//global don't need to select one line
			delete[] oneline_first;
			delete[] oneline_vertCount;
			oneline_lineCount = selected_one_line_index_list.size();
			oneline_first = new int[oneline_lineCount];
			oneline_vertCount = new int[oneline_lineCount];
			for (int i = 0; i < selected_one_line_index_list.size(); i++){
				if (deleted_lines_arr[selected_one_line_index_list[i]]){
					lineCount_deleted_lines -= 1;
				}
				else
				{
					lineCount -= 1;
				}
			}
			
		}

		delete[] first;
		delete[] vertCount;

		delete[] first_deleted_lines;
		delete[] vertCount_deleted_lines;//heap_curropted

		first = new int[lineCount];
		vertCount = new int[lineCount];
		fill(vertCount, vertCount + lineCount, 0);
		fill(first, first + lineCount, 0);

		first_deleted_lines = new int[lineCount_deleted_lines];
		vertCount_deleted_lines = new int[lineCount_deleted_lines];
		fill(vertCount_deleted_lines, vertCount_deleted_lines + lineCount_deleted_lines, 0);
		fill(first_deleted_lines, first_deleted_lines + lineCount_deleted_lines, 0);
		int I = 0;
		int Icore = 0;
		int one_line_list_count = 0;
		totalSize = 0;
		start = clock();
		if (is_ball_change_sign) {
			for (int i = 0; i < streamlines.size(); i++)
			{
				if (!isin(selected_lines_index_list, i)) {
					vertCount_deleted_lines[Icore] = streamlines[i].path.size();
					first_deleted_lines[Icore] = totalSize;
					Icore++;
				}
				else
				{
					vertCount[I] = streamlines[i].path.size();
					first[I] = totalSize;
					I++;
				}
				totalSize += streamlines[i].path.size();
			}
		}
		else{
			for (int i = 0; i < streamlines.size(); i++)
			{
				if (!is_global_sign && selected_lines_index_list.size() != 0 && selected_one_line_index_list.size() > 0 && isin(selected_one_line_index_list, i))
				{
					oneline_vertCount[one_line_list_count] = streamlines[i].path.size();
					oneline_first[one_line_list_count] = totalSize;
					one_line_list_count++;
				}
				else if (deleted_lines_arr[i]) {//local line delete
					vertCount_deleted_lines[Icore] = streamlines[i].path.size();
					first_deleted_lines[Icore] = totalSize;
					Icore++;
				}
				else
				{
					vertCount[I] = streamlines[i].path.size();
					first[I] = totalSize;
					I++;
				}
				totalSize += streamlines[i].path.size();
			}
		}
	}
	finish = clock();
	//cout << "get vertCount...:" << (finish - start) << endl;
}
void Viewer::displayScene()
{
	if (extVertexBufferObject)
		pglBindBuffer(GL_ARRAY_BUFFER, 0);

	glEnableClientState(GL_VERTEX_ARRAY);
	if (extVertexBufferObject)
	{
		pglBindBuffer(GL_ARRAY_BUFFER, vboVertID);
		glVertexPointer(3, GL_FLOAT, 0, 0);
		pglBindBuffer(GL_ARRAY_BUFFER, 0);
	}
	else
		glVertexPointer(3, GL_FLOAT, 0, vertices);

	if (doColors && dataHasColors)
	{
		glEnableClientState(GL_COLOR_ARRAY);
		if (extVertexBufferObject)
		{
			pglBindBuffer(GL_ARRAY_BUFFER, vboColID);
			glColorPointer(4, GL_FLOAT, 4 * sizeof(GLfloat), 0);
			pglBindBuffer(GL_ARRAY_BUFFER, 0);
			//glEnableClientState(GL_COLOR_ARRAY);
		}
		else
			glColorPointer(4, GL_FLOAT, 4 * sizeof(GLfloat), colors);
	}
	else
		glDisableClientState(GL_COLOR_ARRAY);

	glEnable(GL_LINE_SMOOTH);
	glHint(GL_LINE_SMOOTH, GL_NICEST);

	//下面画线
	glLineWidth(line_width);
	//static bool isInitialized = false;
	clock_t start, finish;
	start = clock();

	//qDebug() << "alpha_value = " << alpha_value;
	if (deleted_lines_num != previous_deleted_lines_num || is_ball_change_sign && selected_lines_index_list.size() > 0 || selected_one_line_index_list.size() > 0 || selected_one_line_index == -1){
		previous_deleted_lines_num = deleted_lines_num;

		ILRender::deleteIdentifier(ilID_solid_lines);
		ILRender::deleteIdentifier(ilID_deleted_lines);
		if (selected_one_line_index_list.size() > 0)
			ILRender::deleteIdentifier(oneline_tmp);
		ilID_deleted_lines = ILRender::prepareMultiDrawArrays(first_deleted_lines, vertCount_deleted_lines, lineCount_deleted_lines);//透明的线
		ilID_solid_lines = ILRender::prepareMultiDrawArrays(first, vertCount, lineCount);//实线

		if (selected_one_line_index_list.size() > 0)
			oneline_tmp = ILRender::prepareMultiDrawArrays(oneline_first, oneline_vertCount, oneline_lineCount);
	}	

	finish = clock();
//	qDebug() << "prepare line time:" << (finish - start);// / CLOCKS_PER_SEC << " (s) ";
	curIL->enableZSort(true);
	glDepthMask(GL_FALSE);
	/*if (streamlines.size() > 300){
		uniformColor[4] = 0;
	}*/

	uniformColor[0] = color_r;
	uniformColor[1] = color_g;
	uniformColor[2] = color_b;
	
	start = clock();
	if (is_global_sign)//init
	{
		cur_line_num = streamlines.size();
		//第一次上色 虚线
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		//glColor4fv(uniformColor);
		//curIL->multiDrawArrays(ilID_deleted_lines);

		//第二次上色 实线
		glColor3fv(uniformColor);
		curIL->multiDrawArrays(ilID_solid_lines);

	}
	else//select lines
	{
		//第一次上色 虚线
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glColor4fv(uniformColor);
		curIL->multiDrawArrays(ilID_deleted_lines);

		//第二次上色 实线
		glColor3fv(uniformColor);
		curIL->multiDrawArrays(ilID_solid_lines);

		//给选定线设置颜色
		if (selected_one_line_index_list.size() > 0){
			if (selected_one_line_index_list.size() == 1)
				glLineWidth(line_width + 0.5);
			else
			{
				glLineWidth(line_width);
			}
			float uniformColor_tmp[4] = { 1.0f, 0.0f, 0.0f, 1.0f };
			glColor4fv(uniformColor_tmp);
			curIL->multiDrawArrays(oneline_tmp);
			glLineWidth(line_width);
		}
	}
	finish = clock();
	//qDebug() << "draw line time:" << (finish - start);// / CLOCKS_PER_SEC << " (s) ";
	glDepthMask(GL_TRUE);
}

void Viewer::receiveLineOrder(vector<int> order, int sign){
	if (sign == 1){
		lineOrder_global = order;
		//qDebug() << "lineOrder_global size is " << lineOrder_global.size();
	}
	else
	{
		lineOrder_part = order;
		line_to_delete_num = 0;
		deleted_lines_num = 0;
		//qDebug() << "lineOrder_part size is " << lineOrder_part.size();
		/*QString str_t = "";
		for (int i = 0; i < lineOrder_part.size(); i++)
		{
			str_t.append(QString::number(lineOrder_part[i]) + ",");
		}
		qDebug() << str_t;*/
	}
	if (finishedcal_sign)
		update();
	else
	{
		//if(line_to_delete_num < order.size())
			getLineToDelete(line_to_delete_num);
		//else 
		//	getLineToDelete(order.size());
	}
	//getLineToDelete(line_to_delete_num);
}

void Viewer::getLineToDelete(int num){
	//qDebug() << "viewer num is " << num;
	line_to_delete_num = num;
	getLineToDeleteLinesNum();
	//update();
}

void Viewer::getSelectedOneLine(vector<int> line_id_list){
	selected_one_line_index_list = line_id_list;
	selected_one_line_index = line_id_list[0];
	if (line_id_list[0] == -1){
		selected_one_line_index_list.clear();
	}
	//qDebug() << "get one line:" << selected_one_line_index;
	this->update();
}

void Viewer::getSelectedLinesIndexList(vector<int> list){
	//qDebug() << "list size():" << list.size();
	//qDebug() << calLineOrderThread->isFinished();
	if (list.size() > 0 && list[0] != -1 && !is_global_sign)//local one line
	{
		vector<int> *Ndelete = NULL;
		emit sendSelectedLinesIndex(list, selected_lines_index_list);
		selected_one_line_index_list = list;
		//if (list.size() != 1){
		//	selected_one_line_index = -1;
		//}
	}
	update();
}

void Viewer::clearLineOrderPartList(){
	//qDebug() << "clear lineorder part";
	//lineOrder_part.clear();
	deleted_lines_num = 0;
	line_to_delete_num = 0;
	emit sendTotalLineNum(cur_line_num, deleted_lines_num);
	emit sendLineToDeleteNum(line_to_delete_num);
	update();
}

void Viewer::getDevicePointers(void *device_pointers){
//	qDebug() << "val variation";
	this->device_pointers =(DevicePointers*) device_pointers;
	is_change_mode_sign = true;

	alphaChanged(1.5);
}


void Viewer::modeChangeAction(){
	doColors = !doColors;
	update();
}
void Viewer::selectAreaAction(){
	if (!is_ball_change_sign)
	{
		radius = 0;
		//selected_one_line_index = -1;
		selected_one_line_index_list.clear();
		line_to_delete_num = deleted_lines_index_list.size();
		deleted_lines_num = 0;
		emit reset();
	}
	is_ball_change_sign = true;
}

void Viewer::keyPressEvent(QKeyEvent *key){
	switch (key->key())
	{
		qDebug() << "keypress        ";
	case Qt::Key_B:
		if (cylinderBlinnSupported)
		{
			qDebug() << "Key_B";
			curIL = &cylinderBlinnIL;
			lightingModel = ILLightingModel::IL_CYLINDER_BLINN;
			update();
		}
		break;
	case Qt::Key_M:
		if (maximumPhongSupported)
		{
			qDebug() << "Key_M";
			curIL = &maximumPhongIL;
			lightingModel = ILLightingModel::IL_MAXIMUM_PHONG;
			update();
		}
		break;
	case Qt::Key_P:
		if (cylinderPhongSupported)
		{
			qDebug() << "Key_P";
			curIL = &cylinderPhongIL;
			lightingModel = ILLightingModel::IL_CYLINDER_PHONG;
			update();
		}
		break;
	default:
		break;
	}
}
class CompareDist
{
public:
	bool operator()(pair<int, float> n1, pair<int, float> n2) {
		return n1.second<n2.second;
	}
};


void QSort(int i, int j) {

}//data accuracy impact rate. (logn/n)*(31-logn)/31

//Sort->MakeKD-Tree


mutex mtx;
int Nlines;
priority_queue<pair<int, float>, vector<pair<int, float>>, CompareDist> pqueue;
int *KD_largest;
DWORD WINAPI UpdateCallback(LPVOID param) {
	float *opt = (float *)(param);
	KD_largest = new int[Viewer::This->N_p * 18];
	priority_queue<KD_Tree::KD_Vector3, vector<KD_Tree::KD_Vector3>, KD_Tree::cmp<0>> pq;
	mtx.lock();
	//KD_Tree::build();
	int Nlines = Viewer::This->streamlines.size();
	int n_p = Viewer::This->N_p;
	CalcLineOrderThread *clot = Viewer::This->calLineOrderThread;
	for (int i = 0; i < n_p; i++)
	{
		pqueue.push(make_pair(i, opt[i]));
	}
	for (int i = 0; i < n_p*.14; i++)
	{
		pair<int, float> tmp_pair = pqueue.top();
		pqueue.pop();
		KD_Tree::KD_Vector3 pt = clot->ptMappings(tmp_pair.first);//new instance, sharing may be an option for space saving;
		pt.id = tmp_pair.first;
		pq.push(pt);
	}
	int idx = 0;

	KD_Tree::build<0>(pq, 0, KD_largest, idx);
	float sigma = .6;
	float *opt_orig = new float[n_p];
	std::copy(opt, opt + n_p, opt_orig);
	for (int i = 0; i < n_p; i++)
	{
		float dist = -1;
		auto ret_tuple = KD_Tree::search(clot->ptMappings(i), KD_largest);
		dist = ret_tuple.second;
		if (opt[i] > 1 || opt[i] < 0)
		{
			exit(3);
		//	break;
		}
		opt[i] = max((1 * opt_orig[i] + 1.8 * opt_orig[ret_tuple.first >> 2] * exp(-dist / (2.f*sigma*sigma))) / 2.f, .2*exp(-dist / (2.f*sigma*sigma)));
	}
	delete[] opt_orig;
	Viewer::This->update();
	mtx.unlock();
	return 0;
}

void Viewer::alphaChanged(int value) {
	
	if (is_change_mode_sign&&!mtx.try_lock()) {
		mtx.unlock();
		alpha_value = (double)value/ 10;
		printf("\n%d %f\n\n", value, alpha_value);

		if (alpha_value == 0) {
			alpha_value = 0.00001;
		}

		static float jun_v = -1;
		static float *output = 0;
		static float max;
		static bool runonce = true;
		float *d_variation = device_pointers->variation;
		float *d_distance = device_pointers->val;
		float *d_avg;

		if (runonce)
		{
			float *d_max;
			cudaMalloc(&d_max, 4);
			opacity_point.resize(N_p);
			if (!output)
				cudaMalloc(&output, sizeof(float) * N_p);

			cudaMalloc(&d_avg, sizeof(float));
			cavg(d_variation, N_p, d_avg, d_max);
			cudaMemcpy(&jun_v, d_avg, 4, cudaMemcpyDeviceToHost);
			jun_v /= (64 * N_p);
			cudaMemcpy(&max, d_max, sizeof(float), cudaMemcpyDeviceToHost);
			printf("\n\n Max: %f %f\n", max, pow(E, -max*max / 2.));
			max = pow(E, -max*max / 2.);
			cudaFree(d_avg);
			cudaFree(d_max);
		}

		if (!output)
			cudaMalloc(&output, sizeof(float) * N_p);
		printf("\n\n\n%f\n\n", alpha_value);
		

		if (runonce) {
			cuda(d_variation, d_distance, output, N_p, jun_v, alpha_value, max);
			Nlines = streamlines.size();
			//WaitForSingleObject(CreateThread(0, 0, UpdateCallback, opacity_point.data(), 0, 0), INFINITE);
			
			
			runonce = false;

		}
		cudaMemcpy(opacity_point.data(), output, N_p * sizeof(float), cudaMemcpyDeviceToHost);
		float _max = INT_MIN;
		for (int i = 0; i < N_p; i++) {
			if (opacity_point[i] > _max) {
				_max = opacity_point[i];
			}
		}
		for (int i = 0; i < N_p; i++) {
			opacity_point[i] /= _max;
		}
		for (int i = 0; i < N_p; i++) {
			opacity_point[i] *= alpha_value;
			//opacity_point[i] = 0;// opacity_point[i] > 1 ? 1 : opacity_point[i];
		}
		runonce = false;
		update();
	}
}
void Viewer::setModeAction(int mode) {
	if (mode == 1)
		doColors = true;
	else
		doColors = false;
}

void Viewer::moveLightAction() {
	is_light_move_action = !is_light_move_action;
}

void Viewer::getLightParameters(int id, float value) {
		
	switch (id)
	{
	case 0:
		ka = value;
		break;
	case 1:
		kd = value;
		break;
	case 2:
		ks = value;
		break;
	case 3:
		gloss = value;
		break;
	case 4:
		color_r = value;
		break;
	case 5:
		color_g = value;
		break;
	case 6:
		color_b = value;
		break;
	case 8:
		/*light_move_z = value;
		lightPosition[2] = light_z;*/
		lightPosition[2] = value;
		break;
	case 9:
	case 10:
		light_mode = value;
		break;
	default:
		break;
	}
	update();
}

void Viewer::getColor(QColor c) {
	color_r = (float)c.red() / 255;
	color_g = (float)c.green() / 255;
	color_b = (float)c.blue() / 255;
	//qDebug() << "color:" << color_r << "," << color_g << "," << color_b;
	update();
}

void Viewer::printLightParameters() {
#ifdef QT_NO_DEBUG_OUTPUT
	return;
#endif // QT
	qDebug() << "Light Info:============================";
	/*qDebug() << "ambient:" << ambient_r << "," << ambient_g << "," << ambient_b;
	qDebug() << "diffuse:" << diffuse_r << "," << diffuse_g << "," << diffuse_b;
	qDebug() << "specular:" << specular_r << "," << specular_g << "," << specular_b;
	qDebug() << "shininess:" << shininess;*/
	if (light_mode == 0)
		qDebug() << "Parall Light";
	else if (light_mode == 1)
		qDebug() << "Point Light";
	else if (light_mode == 2) {
		qDebug() << "Spot Light";
	}
	if (material_mode == 0) {
		qDebug() << "Material";
	}
	else if (material_mode == 1) {
		qDebug() << "Color Material";
	}
	qDebug() << "color:" << color_r << "," << color_g << "," << color_b;
	qDebug() << "Light Position:" << lightPosition[0] << "," << lightPosition[1] << "," << lightPosition[2];
	qDebug() << "Light Direction:" << lightDirection[0] << "," << lightDirection[1] << "," << lightDirection[2];
	qDebug() << "Light move z:" << light_move_z;
	qDebug() << "end==========================================";
}
