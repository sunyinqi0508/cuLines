#ifndef LIGHTPARAMETERS_H
#define LIGHTPARAMETERS_H

#include <qwidget.h>
#include <QVBoxlayout>
#include <QHBoxlayout>
#include <qpushbutton.h>
#include <QVariant>
#include <QCloseEvent> 
#include <qdebug.h>
#include <qlabel.h>
#include <qslider.h>
#include <qscrollarea.h>
#include <QDoubleSpinBox>

//#include "ColorLuminancePicker.h"
#include "ColorPicker.h"
using namespace std;

class LightParameters : public QWidget
{
    Q_OBJECT
public:
	LightParameters(QWidget* parent=0);
    QVBoxLayout* mainLayout;
	int H = 0;
	int S = 0;
	int V = 0;
	QColor *color;
	QSlider *slider_pos_z;
	QLabel *label_ambient;
	QLabel *label_diffuse;
	QLabel *label_specular;
	QLabel *label_light;
	QLabel *label_color_r;
	QLabel *label_color_g;
	QLabel *label_color_b;
	QLabel *label_color_picker;

signals:
	void sendLightParameters(int id, float value);
	void sendColor(QColor c);
	void sendFlowPara(int id, double value);

	public slots :
		void getChangeValue(int value);
		void getButtonName();
		void getHS(int h, int s);
		void getV(int v);
		void getPosZSlider(int max_z);
		void getTop();
		void spinboxValueChanged(double value);
		void applyThePara();
		void sliderValuechanged(int value);
		void setFlowParas(vector<double> paras_list);
		void setAllParas();

protected:
	vector<QDoubleSpinBox*> spinbox_list;/*contains all spinboxs' pointer*/
	vector<QSlider*> slider_list;/*contains all sliderbars' pointer*/
};

#endif // LIGHTPARAMETERS_H
