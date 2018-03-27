#include "LightParameters.h"

LightParameters::LightParameters(QWidget* parent):QWidget(parent)
{
    resize(600,600);
	setWindowTitle("Parameters");
	QIcon systemIcon;
	systemIcon.addPixmap(QPixmap("icons/lightparameters.png"), QIcon::Active, QIcon::On);
	this->setWindowIcon(systemIcon);
	//this->setWindowFlags(windowFlags() | Qt::WindowStaysOnTopHint);

	QScrollArea *m_ScrollArea = new QScrollArea(this);
	m_ScrollArea->setSizePolicy(QSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding));
	m_ScrollArea->setWidgetResizable(true);

	QWidget* widget = new QWidget(this);
	widget->setMinimumSize(72, 32);
	widget->setMaximumSize(80, 32);

    QVBoxLayout *vboxLayout = new QVBoxLayout();
	vboxLayout->setSizeConstraint(QVBoxLayout::SetMinAndMaxSize);
	//ambient
	QHBoxLayout *hb_ambient = new QHBoxLayout();
	label_ambient = new QLabel("Ambient:", this);
	QSlider *slider_ambient = new QSlider(this);
	slider_ambient->setProperty("id", 0);
	slider_ambient->setRange(0, 100);
	slider_ambient->setValue(0);
	slider_ambient->setSingleStep(1);
	slider_ambient->setOrientation(Qt::Horizontal);
	hb_ambient->addWidget(label_ambient);
	hb_ambient->addWidget(slider_ambient);

	//diffuse
	QHBoxLayout *hb_diffuse = new QHBoxLayout();
	label_diffuse = new QLabel("Diffuse:", this);
	QSlider *slider_diffuse = new QSlider(this);
	slider_diffuse->setProperty("id", 1);
	slider_diffuse->setRange(0, 100);
	slider_diffuse->setValue(0);
	slider_diffuse->setSingleStep(1);
	slider_diffuse->setOrientation(Qt::Horizontal);
	hb_diffuse->addWidget(label_diffuse);
	hb_diffuse->addWidget(slider_diffuse);

	//specular
	QHBoxLayout *hb_specular = new QHBoxLayout();
	label_specular = new QLabel("Specular:", this);
	QSlider *slider_specular = new QSlider(this);
	slider_specular->setProperty("id", 2);
	slider_specular->setRange(0, 1000);
	slider_specular->setValue(0);
	slider_specular->setSingleStep(1);
	slider_specular->setOrientation(Qt::Horizontal);
	hb_specular->addWidget(label_specular);
	hb_specular->addWidget(slider_specular);

	//shininess
	QHBoxLayout *hb_shininess = new QHBoxLayout();
	QLabel *label_shininess = new QLabel("Shininess:", this);
	QSlider *slider_shininess = new QSlider(this);
	slider_shininess->setProperty("id", 3);
	slider_shininess->setRange(0, 128);
	slider_shininess->setValue(10);
	slider_shininess->setSingleStep(1);
	slider_shininess->setOrientation(Qt::Horizontal);
	hb_shininess->addWidget(label_shininess);
	hb_shininess->addWidget(slider_shininess);

	//color
	QHBoxLayout *hb_color_r = new QHBoxLayout();
	label_color_r = new QLabel("Color R:", this);
	QSlider *slider_color_r = new QSlider(this);
	slider_color_r->setProperty("id", 4);
	slider_color_r->setRange(0, 255);
	slider_color_r->setValue(0);
	slider_color_r->setSingleStep(1);
	slider_color_r->setOrientation(Qt::Horizontal);
	hb_color_r->addWidget(label_color_r);
	hb_color_r->addWidget(slider_color_r);

	QHBoxLayout *hb_color_g = new QHBoxLayout();
	label_color_g = new QLabel("Color G:", this);
	QSlider *slider_color_g = new QSlider(this);
	slider_color_g->setProperty("id", 5);
	slider_color_g->setRange(0, 255);
	slider_color_g->setValue(0);
	slider_color_g->setSingleStep(1);
	slider_color_g->setOrientation(Qt::Horizontal);
	hb_color_g->addWidget(label_color_g);
	hb_color_g->addWidget(slider_color_g);

	QHBoxLayout *hb_color_b = new QHBoxLayout();
	label_color_b = new QLabel("Color B:", this);
	QSlider *slider_color_b = new QSlider(this);
	slider_color_b->setProperty("id", 6);
	slider_color_b->setRange(0, 255);
	slider_color_b->setValue(0);
	slider_color_b->setSingleStep(1);
	slider_color_b->setOrientation(Qt::Horizontal);
	hb_color_b->addWidget(label_color_b);
	hb_color_b->addWidget(slider_color_b);

	//light select
	QHBoxLayout *hb_btn_light = new QHBoxLayout();
	label_light = new QLabel("Parall Light", this);
	QPushButton *btn_parall_light = new QPushButton("Parall Light", this);
	btn_parall_light->setProperty("id", 9);
	QPushButton *btn_point_light = new QPushButton("Point Light", this);
	btn_point_light->setProperty("id", 10);
	/*QPushButton *btn_spot_light = new QPushButton("Spot Light", this);
	btn_spot_light->setProperty("id", 2);*/
	hb_btn_light->addWidget(label_light);
	hb_btn_light->addWidget(btn_parall_light);
	hb_btn_light->addWidget(btn_point_light);
	//hb_btn_light->addWidget(btn_spot_light);

	QString  pushButtonStyle(
		"QPushButton{border: 0px solid white;background-color:rgb(211, 211, 211)}"
		"QPushButton:pressed{background-color:rgb(211, 211, 211);border-style: inset; }"
		);
	QHBoxLayout *hb_picker_label = new QHBoxLayout();
	label_color_picker = new QLabel("Color Picker:(0,0,0)", this);
	QPushButton *btn_hide = new QPushButton("", this);
	btn_hide->setStyleSheet(pushButtonStyle);
	hb_picker_label->addWidget(label_color_picker);
	hb_picker_label->addWidget(btn_hide);

	QHBoxLayout *hb_palette = new QHBoxLayout();
	ColorPicker *colorPicker = new ColorPicker(this);
	colorPicker->setProperty("id", 7);
	connect(colorPicker, SIGNAL(newCol(int,int)), this, SLOT(getHS(int, int)));
	//ColorLuminancePicker *colorLuminancePicker = new ColorLuminancePicker(this);
	//connect(colorLuminancePicker, SIGNAL(sendV(int)), this, SLOT(getV(int)));
	//hb_palette->addWidget(colorPicker);
	//hb_palette->addWidget(colorLuminancePicker);

	QHBoxLayout *hb_pos_z = new QHBoxLayout();
	QLabel *label_pos_z = new QLabel("Position z:", this);
	slider_pos_z = new QSlider(this);
	slider_pos_z->setProperty("id", 8);
	slider_pos_z->setRange(0, 100);
	slider_pos_z->setValue(50);
	slider_pos_z->setSingleStep(1);
	slider_pos_z->setOrientation(Qt::Horizontal);
	hb_pos_z->addWidget(label_pos_z);
	hb_pos_z->addWidget(slider_pos_z);

	vboxLayout->addLayout(hb_ambient);
	vboxLayout->addLayout(hb_diffuse);
	vboxLayout->addLayout(hb_specular);
	vboxLayout->addLayout(hb_shininess);
	vboxLayout->addLayout(hb_color_r);
	vboxLayout->addLayout(hb_color_g);
	vboxLayout->addLayout(hb_color_b);
	vboxLayout->addLayout(hb_btn_light);
	vboxLayout->addLayout(hb_picker_label);
	vboxLayout->addLayout(hb_palette);
	vboxLayout->addLayout(hb_pos_z);

	connect(slider_ambient, SIGNAL(valueChanged(int)), this, SLOT(getChangeValue(int)));
	connect(slider_diffuse, SIGNAL(valueChanged(int)), this, SLOT(getChangeValue(int)));
	connect(slider_specular, SIGNAL(valueChanged(int)), this, SLOT(getChangeValue(int)));
	connect(slider_shininess, SIGNAL(valueChanged(int)), this, SLOT(getChangeValue(int)));
	connect(slider_color_r, SIGNAL(valueChanged(int)), this, SLOT(getChangeValue(int)));
	connect(slider_color_g, SIGNAL(valueChanged(int)), this, SLOT(getChangeValue(int)));
	connect(slider_color_b, SIGNAL(valueChanged(int)), this, SLOT(getChangeValue(int)));
	connect(btn_parall_light, SIGNAL(clicked()), this, SLOT(getButtonName()));
	connect(btn_point_light, SIGNAL(clicked()), this, SLOT(getButtonName()));
	connect(slider_pos_z, SIGNAL(valueChanged(int)), this, SLOT(getChangeValue(int)));

	//flow parameters
	QString  previous_pushButtonStyle(
		"QPushButton{border: 1px solid black;border-radius:8px;background-color:rgb(192, 192, 192)}"
		"QPushButton:pressed{background-color:rgb(41, 36, 33);border-style: inset; }"
	);
	QString flow_paras_array[3] = {"para1","para2","para3"};
	for (int i = 0; i < 3; i++)
	{
		QLabel *label = new QLabel(flow_paras_array[i]);
		QVBoxLayout* vb_layout = new QVBoxLayout();
		QHBoxLayout* hb_layout = new QHBoxLayout();
		QDoubleSpinBox* spinBox = new QDoubleSpinBox(this);
		spinbox_list.push_back(spinBox);
		spinBox->setProperty("flow_paras_id", i);
		spinBox->setDecimals(1);
		spinBox->setRange(0.1, 100);
		spinBox->setSingleStep(0.1);
		spinBox->setPrefix("Value:");
		spinBox->setMinimumHeight(60);
		connect(spinBox, SIGNAL(valueChanged(double)), this, SLOT(spinboxValueChanged(double)));

		QPushButton *btn_apply = new QPushButton();
		btn_apply->setProperty("flow_paras_id", i);
		btn_apply->setIcon(QIcon("icons/apply_button.png"));
		btn_apply->setStyleSheet(previous_pushButtonStyle);
		connect(btn_apply, SIGNAL(clicked()), this, SLOT(applyThePara()));
		hb_layout->addWidget(label);
		hb_layout->addWidget(spinBox);
		hb_layout->addWidget(btn_apply);
		QSlider *m_slider = new QSlider(this);
		m_slider->setRange(1, 1000);
		m_slider->setSingleStep(1);
		m_slider->setValue(1);
		m_slider->setOrientation(Qt::Horizontal);
		vb_layout->addLayout(hb_layout);
		vb_layout->addWidget(m_slider);
		m_slider->setProperty("flow_paras_id", i);
		slider_list.push_back(m_slider);
		connect(m_slider, SIGNAL(valueChanged(int)), this, SLOT(sliderValuechanged(int)));
		QWidget *sub_widget = new QWidget(this);
		sub_widget->setLayout(vb_layout);
		vboxLayout->addWidget(sub_widget);
		QSizePolicy sp;
		sp = label->sizePolicy();
		sp.setHorizontalStretch(4);
		label->setSizePolicy(sp);
		sp = spinBox->sizePolicy();
		sp.setHorizontalStretch(2);
		spinBox->setSizePolicy(sp);
		sp = btn_apply->sizePolicy();
		sp.setHorizontalStretch(1);
		btn_apply->setSizePolicy(sp);
	}
	QPushButton *btn_apply_all = new QPushButton();
	btn_apply_all->setIcon(QIcon("icons/confirm_btn.png"));
	btn_apply_all->setStyleSheet(previous_pushButtonStyle);
	connect(btn_apply_all, SIGNAL(clicked()), this, SLOT(setAllParas()));
	vboxLayout->addWidget(btn_apply_all);

	widget->setLayout(vboxLayout);
	m_ScrollArea->setWidget(widget);

	mainLayout = new QVBoxLayout(this);
	mainLayout->addWidget(m_ScrollArea);
	this->setLayout(mainLayout);

	color = new QColor();
}
void LightParameters::getChangeValue(int value) {
	QVariant propertyV = sender()->property("id");
	float para;
	if (propertyV.isValid()) {
		int slider_id = propertyV.toInt();
		if (slider_id == 3 || slider_id == 8) {//shininess pos_z
			para = value;
			emit sendLightParameters(slider_id, value);
		}
		else if (slider_id == 4 || slider_id == 5 || slider_id == 6) {//color rgb
			para = (float)value / 255;
			emit sendLightParameters(slider_id, para);
		}
		else//light
		{
			para = (float)value / 100.0;
			emit sendLightParameters(slider_id, para);
		}
		switch (slider_id)
		{
		case 0:
			label_ambient->setText("Ambient:" + QString::number(para));
			break;
		case 1:
			label_diffuse->setText("Diffuse:" + QString::number(para));
			break;
		case 2:
			label_specular->setText("Specular:" + QString::number(para));
			break;
		case 4:
			label_color_r->setText("Color R:" + QString::number(value));
			break;
		case 5:
			label_color_g->setText("Color G:" + QString::number(value));
			break;
		case 6:
			label_color_b->setText("Color B:" + QString::number(value));
			break;
		default:
			break;
		}
	}
}

void LightParameters::getButtonName() {
	QVariant propertyV = sender()->property("id");
	if (propertyV.isValid()) {
		int btn_id = propertyV.toInt();
		switch (btn_id)
		{
		case 9:
			emit sendLightParameters(btn_id, 0);
			label_light->setText("Parall Light");
			break;
		case 10:
			emit sendLightParameters(btn_id, 1);
			label_light->setText("Point Light");
			break;
		default:
			break;
		}
	}
}

void LightParameters::getHS(int h, int s) {
	H = h;
	S = s;
	color->setHsv(H, S, V);
	color->setHsv(H, S, V);
	//qDebug() << "hsv:" << H << "," << S << "," << V;
	QColor c = color->toRgb();
	//qDebug() << c;
	emit sendColor(c);
	label_color_picker->setText("Color Picker:(" + QString::number((float)c.red() / 255) + "," + QString::number((float)c.green() / 255) + "," + QString::number((float)c.blue() / 255) + ")");
}

void LightParameters::getV(int v) {
	V = v;
	color->setHsv(H, S, V);
	QColor c = color->toRgb();
	emit sendColor(c);	
	label_color_picker->setText("Color Picker:(" + QString::number((float)c.red() / 255) + "," + QString::number((float)c.green() / 255) + "," + QString::number((float)c.blue() / 255) + ")");
}

void LightParameters::getPosZSlider(int max_z){
	slider_pos_z->setRange(0, max_z * 2);
	slider_pos_z->setValue(max_z);
}

void LightParameters::getTop(){
	this->raise();
}

void LightParameters::spinboxValueChanged(double value) {
	QVariant propertyV = sender()->property("flow_paras_id");
	if (propertyV.isValid()) {
		int spinbox_id = propertyV.toInt();
		slider_list[spinbox_id]->setValue(value * 10);
	}
}

void LightParameters::applyThePara() {
	QVariant propertyV = sender()->property("flow_paras_id");
	if (propertyV.isValid()) {
		int spinbox_id = propertyV.toInt();
		//send changed value to viewer.cpp
		emit sendFlowPara(spinbox_id, spinbox_list[spinbox_id]->value());
	}
}

void LightParameters::sliderValuechanged(int value) {
	QVariant propertyV = sender()->property("flow_paras_id");
	if (propertyV.isValid()) {
		int spinbox_id = propertyV.toInt();
		double spinbox_value = (double)value / 10;
		spinbox_list[spinbox_id]->setValue(spinbox_value);
	}
}

void LightParameters::setFlowParas(vector<double> paras_list) {
	for (int i = 0; i < paras_list.size(); i++)
	{
		spinbox_list[i]->setValue(paras_list[i]);
		slider_list[i]->setValue(paras_list[i]);
	}
}

void LightParameters::setAllParas() {
	for (int i = 0; i < spinbox_list.size(); i++)
	{
		emit sendFlowPara(i, spinbox_list[i]->value());
	}
}