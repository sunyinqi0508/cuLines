#include "mainwindow.h"

using namespace std;

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{
	setAttribute(Qt::WA_AcceptTouchEvents, true);
	
    //set program icon
    QIcon systemIcon;
    systemIcon.addPixmap(QPixmap("icons/systemIcon.png"),QIcon::Active,QIcon::On);
    this->setWindowIcon(systemIcon);

	//set dialog title
	setWindowTitle(QString("Flow Visualization"));

	viewer = new Viewer();
	connect(viewer, SIGNAL(sendTotalLineNum(int, int)), this, SLOT(receiveTotalLineNum(int, int)));
	connect(viewer, SIGNAL(sendLineToDeleteNum(int)), this, SLOT(receiveLineToDeleteNum(int)));
	connect(viewer, SIGNAL(sendUnfinishSign(bool)), this, SLOT(receiveUnfinishSign(bool)));

	//toolbar
	mode_change_action = new QAction(tr("&Change Mode"), this);
	mode_change_action->setStatusTip(tr("Change Mode."));
	mode_change_action->setIcon(QIcon("icons/mode_change.png"));
	connect(mode_change_action, SIGNAL(triggered()), viewer, SLOT(modeChangeAction()));
	QToolBar *toolBar = addToolBar(tr("&Change Mode"));
	toolBar->addAction(mode_change_action);
	toolBar->addSeparator();
	toolBar->setFloatable(false);
	toolBar->setMovable(false);

	select_area_action = new QAction(tr("&Select Area"), this);
	select_area_action->setStatusTip(tr("Select Area."));
	select_area_action->setIcon(QIcon("icons/select_area.png"));
	connect(select_area_action, SIGNAL(triggered()), viewer, SLOT(selectAreaAction()));
	toolBar = addToolBar(tr("&Select Area"));
	toolBar->setFloatable(false);
	toolBar->setMovable(false);
	toolBar->addAction(select_area_action);
	toolBar->addSeparator();

	move_light_action = new QAction(tr("&Move Light Position"), this);
	move_light_action->setStatusTip(tr("Move Light Position."));
	move_light_action->setIcon(QIcon("icons/move_light.png"));
	connect(move_light_action, SIGNAL(triggered()), viewer, SLOT(moveLightAction()));
	toolBar = addToolBar(tr("&Move Light Position"));
	toolBar->addAction(move_light_action);
	toolBar->setFloatable(false);
	toolBar->setMovable(false);
	toolBar->addSeparator();

	reset_action = new QAction(tr("&Reset"), this);
	reset_action->setStatusTip(tr("Reset."));
	reset_action->setIcon(QIcon("icons/reset.png"));
	connect(reset_action, SIGNAL(triggered()), viewer, SLOT(resetAction()));
	toolBar = addToolBar(tr("&Reset"));
	toolBar->addAction(reset_action);
	toolBar->setFloatable(false);
	toolBar->setMovable(false);
	toolBar->addSeparator();

	save_action = new QAction(tr("&Save Settings"), this);
	save_action->setStatusTip(tr("Save Settings."));
	save_action->setIcon(QIcon("icons/save.png"));
	connect(save_action, SIGNAL(triggered()), viewer, SLOT(saveSettings()));
	toolBar = addToolBar(tr("&Save Settings"));
	toolBar->addAction(save_action);
	toolBar->setFloatable(false);
	toolBar->setMovable(false);
	toolBar->addSeparator();

	load_action = new QAction(tr("&Load Settings"), this);
	load_action->setStatusTip(tr("Load Settings."));
	load_action->setIcon(QIcon("icons/load.png"));
	connect(load_action, SIGNAL(triggered()), viewer, SLOT(loadSettings()));
	toolBar = addToolBar(tr("&Load Settings"));
	toolBar->setFloatable(false);
	toolBar->setMovable(false);
	toolBar->addAction(load_action);
	toolBar->addSeparator();

	parameter_action = new QAction(tr("&Show Para Label"), this);
	parameter_action->setStatusTip(tr("Show Para Label."));
	parameter_action->setIcon(QIcon("icons/show_para.png"));
	connect(parameter_action, SIGNAL(triggered()), this, SLOT(showParaLabel()));
	toolBar = addToolBar(tr("&Show Para Label"));
	toolBar->setFloatable(false);
	toolBar->setMovable(false);
	toolBar->addAction(parameter_action);
	toolBar->addSeparator();

	QWidget *spacer = new QWidget();
	spacer->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
	spacer->setFixedSize(500, 50);
	toolBar->addWidget(spacer);
	alpha_label = new QLabel(this);
	alpha_slider = new QSlider(this);
	connect(alpha_slider, SIGNAL(valueChanged(int)), viewer, SLOT(alphaChanged(int)));
	connect(alpha_slider, SIGNAL(valueChanged(int)), this, SLOT(alphaChanged(int)));
	alpha_slider->setOrientation(Qt::Horizontal);
	alpha_slider->setRange(0, 100);
	alpha_slider->setValue(15);
	QHBoxLayout *hboxlayout = new QHBoxLayout();
	hboxlayout->addWidget(alpha_label);
	hboxlayout->addWidget(alpha_slider);
	spacer->setLayout(hboxlayout);
	hboxlayout->setContentsMargins(50, 0, 50, 0);

	QString style_sheet_slider("QSlider::groove:horizontal{"
		"border: 1px solid #4A708B;"
		"background: #C0C0C0;"
		"height: 25px;"
		"left: 50px; right: 50px;"
		"border-radius: 1px;"
		"padding-left:-1px;"
		"padding-right:-1px;"
		"}"

		"QSlider::sub-page:horizontal{"
		"background: qlineargradient(x1 : 0, y1 : 0, x2 : 0, y2 : 1,"
		"stop : 0 #B1B1B1, stop:1 #c4c4c4);"
		"background: qlineargradient(x1 : 0, y1 : 0.2, x2 : 1, y2 : 1,"
		"stop : 0 #5DCCFF, stop: 1 #1874CD);"
		"border: 2px solid #4A708B;"
		"height: 50px;"
		"border-radius: 12px;"
		"}"
		"QSlider::add-page:horizontal{"
		"background: #575757;"
		"border: 2px solid #777;"
		"height: 50px;"
		"border-radius: 2px;"
		"}"
		"	QSlider::handle:horizontal"
		"	{"
		"background: qradialgradient(spread : pad, cx : 0.5, cy : 0.5, radius : 0.5, fx : 0.5, fy : 0.5,"
		"		stop : 0.6 #45ADED, stop:0.778409 rgba(255, 255, 255, 255));"
		"width: 55px;"
		"margin-top: -3px;"
		"	margin-bottom: -3px;"
		"border-radius: 5px;"
		"}"
		"QSlider::handle:horizontal : hover{"
		"background: qradialgradient(spread : pad, cx : 0.5, cy : 0.5, radius : 0.5, fx : 0.5, fy : 0.5, stop : 0.6 #2A8BDA,"
		"	stop:0.778409 rgba(255, 255, 255, 255));"
		"width: 55px;"
		"	margin-top: -3px;"
		"	margin-bottom: -3px;"
		"	border-radius: 5px;"
		"}"
		"QSlider::sub-page:horizontal : disabled{"
		"background: #00009C;"
		"border-color: #999;"
		"}"
		"QSlider::add-page : horizontal : disabled{"
		"background: #eee;"
		"border-color: #999;"
		"}"
		"	QSlider::handle : horizontal : disabled{"
		"background: #eee;"
		"border: 1px solid #aaa;"
		"	border-radius: 4px;"
		"}"
	);
	//progressbar slider
	progressBar = new QProgressBar;
	m_slider = new QSlider(this);
	viewer->slider = m_slider;

	m_slider->setOrientation(Qt::Horizontal);
	m_slider->setMinimumHeight(100);
	m_slider->setStyleSheet(
		style_sheet_slider
		);

	connect(m_slider, SIGNAL(valueChanged(int)), viewer, SLOT(getLineToDelete(int)));
	QThread *th = m_slider->thread();
	th->setPriority(QThread::Priority::HighestPriority);
	connect(m_slider, SIGNAL(valueChanged(int)), this, SLOT(updateTheBackground(int)));
	m_ScrollArea = new QScrollArea(this);
	m_ScrollArea->setSizePolicy(QSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding));
	m_ScrollArea->setWidgetResizable(true);
	m_VBoxLayout = new QVBoxLayout();
	m_VBoxLayout->setSizeConstraint(QVBoxLayout::SetMinAndMaxSize);
	m_VBoxLayout->setAlignment(Qt::AlignCenter);
	QLabel *label = new QLabel("Selected lines:");
	m_VBoxLayout->addWidget(label);
	QWidget* sub_widget = new QWidget(this); 
	sub_widget->setLayout(m_VBoxLayout);
	sub_widget->setStyleSheet("background-color:white;");
	m_ScrollArea->setWidget(sub_widget);

	QString  pushButtonStyle(
		"QPushButton{ background-image:url(icons/button.png);font-family:'Microsoft YaHei';font-size:24px;border: 3px solid black;border-radius:8px;min-height:100;background-color:rgb(192, 192, 192)}"
		"QPushButton:pressed{background-color:rgb(41, 36, 33);border-style: inset; }"
		);

	QPushButton *confirm_btn = new QPushButton("Confirm Select", this);
	confirm_btn->setStyleSheet(pushButtonStyle);
	confirm_btn->setIcon(QIcon("icons/confirm_btn.png"));
	connect(confirm_btn, SIGNAL(clicked()), this, SLOT(confirmSelectBtnAction()));

	multi_select_btn = new QPushButton("Multi-Select", this);
	multi_select_btn->setStyleSheet(pushButtonStyle);
	multi_select_btn->setIcon(QIcon("icons/multi_select.png"));
	connect(multi_select_btn, SIGNAL(clicked()), this, SLOT(selectModeAction()));

    mainLayout = new QGridLayout();
	mainLayout->addWidget(progressBar, 0, 0, 1, 10);
	mainLayout->addWidget(multi_select_btn, 0, 10, 1, 1);
	mainLayout->addWidget(viewer, 1, 0, 10, 10);
	mainLayout->addWidget(m_ScrollArea, 1, 10, 10, 1);
	mainLayout->addWidget(m_slider, 11, 0, 1, 10);
	mainLayout->addWidget(confirm_btn, 11, 10, 1, 1);

	connect(viewer, SIGNAL(sendSelectedLines(vector<int>)), this, SLOT(getSelectedLines(vector<int>)));
	connect(this, SIGNAL(sendSelectedLinesIndexList(vector<int>)), viewer, SLOT(getSelectedLinesIndexList(vector<int>)));
	connect(this, SIGNAL(sendSelectedOneLine(vector<int>)), viewer, SLOT(getSelectedOneLine(vector<int>)));
	connect(viewer, SIGNAL(reset()), this, SLOT(cleanScrollArea()));
	connect(this, SIGNAL(clearLineOrderPart()), viewer, SLOT(clearLineOrderPartList()));
	connect(viewer, SIGNAL(sendDeletedLinesIndexList(vector<bool>)), this, SLOT(getDeletedLinesIndexList(vector<bool>)));

	for (int i = 0; i < 11; i++){
		mainLayout->setColumnStretch(i, 1);
	}

	lightParameters = new LightParameters();
	connect(lightParameters, SIGNAL(sendLightParameters(int, float)), viewer, SLOT(getLightParameters(int, float)));
	connect(lightParameters, SIGNAL(sendColor(QColor)), viewer, SLOT(getColor(QColor)));
	connect(lightParameters, SIGNAL(sendFlowPara(int,double)), viewer, SLOT(getFlowPara(int, double)));
	connect(viewer, SIGNAL(sendPosZSlider(int)), lightParameters, SLOT(getPosZSlider(int)));
	connect(viewer, SIGNAL(sendTop()), lightParameters, SLOT(getTop()));
	//lightParameters->show();

    QWidget *widget = new QWidget();
    widget->setLayout(mainLayout);
    widget->setStyleSheet("background-color:white;");
    setCentralWidget(widget);

    QDesktopWidget *desktop = QApplication::desktop();
	int screenNum = desktop->screenCount();
	for (int i = 0; i<screenNum; i++)
	{
		QRect screen = desktop->screenGeometry();
		qDebug("screen %d, width %d, height %d", i, screen.width(), screen.height());
		int width_offset = 100;
		int height_offset = (16 * screen.height() - 9 * screen.width() + 18 * width_offset) / 32;
		lightParameters->setGeometry(width_offset, height_offset + 200, 600, 600);
		setGeometry(width_offset, height_offset, screen.width() - 2 * width_offset, screen.height() - 2 * height_offset);//set window size and position
		break;
	}

	QPropertyAnimation *animation = new QPropertyAnimation(this, "windowOpacity");
	animation->setDuration(1000);
	animation->setStartValue(0);
	animation->setEndValue(1);
	animation->start();

}

MainWindow::~MainWindow()
{
	lightParameters->close();
	lightParameters->deleteLater();
}

void MainWindow::receiveTotalLineNum(int num1, int num2){
	progressBar->setRange(0, num1);
	QString text = QString("selected line: %1 ,need to delete: %v, already deleted: %2").arg(num1).arg(num2);// / %1 
	progressBar->setFormat(text);
	progressBar->setAlignment(Qt::AlignCenter);
	
}

void MainWindow::receiveLineToDeleteNum(int num){
	progressBar->setValue(num);
}

void MainWindow::receiveUnfinishSign(bool sign){
	if (sign){
		progressBar->setStyleSheet("QProgressBar::chunk {background-color: #00FF00;} QProgressBar{text-align: center;}");
	}
	else
	{
		progressBar->setStyleSheet("QProgressBar::chunk {background-color: red;} QProgressBar{text-align: center;}");
	}
	
}

void MainWindow::confirmSelectBtnAction(){
	vector<int> selected_lines_index_list;
	for (int i = 0; i < checkbox_list.size(); i++){
		if (checkbox_list[i]->isChecked()){
			checkbox_list[i]->setStyleSheet("background-color:red;");
			QString str = checkbox_list[i]->text();
			QStringList str_list = str.split(" ");
			int line_id = str_list[1].toInt();
			selected_lines_index_list.push_back(line_id);
		}
	}
	if (selected_lines_index_list.size() > 0){
		emit sendSelectedLinesIndexList(selected_lines_index_list);
	}
}

void MainWindow::getSelectedLines(vector<int> line_order){
	cleanScrollArea();
	QString str("Line %1");
	QCheckBox* checkBox;
	for (int i = 0; i < line_order.size(); i++)
	{
		checkBox = new QCheckBox(str.arg(line_order[i]), m_ScrollArea);
		checkBox->setProperty("id", i);
		if (deleted_lines_index_list[line_order[i]]){
			checkBox->setStyleSheet("background-color:rgb(128,138,135);");
		}
		connect(checkBox, SIGNAL(stateChanged(int)), this, SLOT(onStateChanged(int)));
		checkbox_list.push_back(checkBox);
		m_VBoxLayout->addWidget(checkBox);
	}
}
void MainWindow::onStateChanged(int state)
{
	QVariant propertyV = sender()->property("id");
	if (propertyV.isValid()){
		int checkBox_id = propertyV.toInt();
		QString str = checkbox_list[checkBox_id]->text();
		QStringList str_list = str.split(" ");
		int line_id = str_list[1].toInt();
		vector<int> line_id_list;
		if (pre_checkbox_id >= 0 && pre_checkbox_id < checkbox_list.size()){
			checkbox_list[pre_checkbox_id]->setStyleSheet("background-color:white;");
			if (!multi_select_sign){
				checkbox_list[pre_checkbox_id]->setChecked(false);
			}
		}
			
		if (state == Qt::Checked) // "选中"
		{
			line_id_list.push_back(line_id);
			emit sendSelectedOneLine(line_id_list);
			checkbox_list[checkBox_id]->setStyleSheet("background-color:rgb(139,126,102);");
			pre_checkbox_id = checkBox_id;
		}
		else // 未选中 - Qt::Unchecked
		{
			if (pre_checkbox_id == checkBox_id){
				line_id = -1;
				line_id_list.push_back(line_id);
				emit sendSelectedOneLine(line_id_list);
			}
			checkbox_list[checkBox_id]->setStyleSheet("background-color:white;");
			if (pre_checkbox_id >= 0 && pre_checkbox_id < checkbox_list.size() && pre_checkbox_id != checkBox_id)
				checkbox_list[pre_checkbox_id]->setStyleSheet("background-color:rgb(139,126,102);");
		}
	}
	updateTheBackground(0);
}

void MainWindow::cleanScrollArea(){
	for (int i = 0; i < checkbox_list.size(); i++)
	{
		if (checkbox_list[i] != NULL){
			delete checkbox_list[i];
			checkbox_list[i] = NULL;
		}
	}
	checkbox_list.clear();
	pre_checkbox_id = -1;
}

void MainWindow::selectModeAction(){
	multi_select_sign = !multi_select_sign;
	if (multi_select_sign){
		multi_select_btn->setText("Single-Select");
		multi_select_btn->setIcon(QIcon("icons/single_select.png"));
	}
	else
	{
		for (int i = 0; i < checkbox_list.size(); i++){
			if (checkbox_list[i]->isChecked()){
				checkbox_list[i]->setChecked(false);
			}
		}
		multi_select_btn->setText("Multi-Select");
		multi_select_btn->setIcon(QIcon("icons/multi_select.png"));
	}
}

void MainWindow::getDeletedLinesIndexList(vector<bool> deleted_lines_index_list){
	this->deleted_lines_index_list = deleted_lines_index_list;
}

void MainWindow::updateTheBackground(int num){
	viewer->setModeAction(2);
	for (int i = 0; i < checkbox_list.size(); i++){
		if (checkbox_list[i]->isChecked()){
			checkbox_list[i]->setStyleSheet("background-color:rgb(139,126,102);");
		}
		else
		{
			QString str = checkbox_list[i]->text();
			QStringList str_list = str.split(" ");
			int line_id = str_list[1].toInt();
			if (deleted_lines_index_list[line_id]){
				checkbox_list[i]->setStyleSheet("background-color:rgb(128,138,135);");
			}
			else
			{
				checkbox_list[i]->setStyleSheet("background-color:white;");
			}
		}		
	}
}

void MainWindow::alphaChanged(int value) {
	viewer->setModeAction(1);//render on points when alpha changed
	double alpha_value = (double)value / 10;
	alpha_label->setText("Current alpha is "+ QString::number(alpha_value));
}

void MainWindow::showParaLabel() {
	is_show_para_label = !is_show_para_label;
	if (is_show_para_label) {
		lightParameters->show();
	}
	else
	{
		lightParameters->hide();
	}
}