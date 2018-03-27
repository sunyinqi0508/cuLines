#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QGridLayout>
#include <QGraphicsScene>
#include <QGraphicsView>
#include <QIcon>
#include <QPixmap>
#include <QWidget>
#include <QDesktopWidget>
#include <qapplication.h>
#include <qprogressbar.h>
#include <qslider.h>
#include <QVBoxLayout>
#include <qscrollarea.h>
#include <qlabel.h>
#include <qcheckbox.h>
#include <qpushbutton.h>
#include <qthread.h>
#include <qaction.h>
#include <qtoolbar.h>
#include <QPropertyAnimation>
#include <QKeyEvent>

#include "viewer.h"

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = 0);
    ~MainWindow();

    QGridLayout *mainLayout;

	QAction *mode_change_action;
	QAction *select_area_action;
	QAction *move_light_action;
	QAction *reset_action;
	QAction *save_action;
	QAction *load_action;
	QAction *parameter_action;

	Viewer *viewer;
	LightParameters *lightParameters;
	QProgressBar *progressBar;
	QSlider *m_slider;
	QSlider *alpha_slider;
	QLabel *alpha_label;
	QPushButton *multi_select_btn;
	vector<QCheckBox*> checkbox_list;
	int pre_checkbox_id = -1;
	bool multi_select_sign = false;
	vector<bool> deleted_lines_index_list;

signals:
	//void sendLineToDelete(int num);
	void sendSelectedOneLine(vector<int> line_id_list);
	void sendSelectedLinesIndexList(vector<int> list);
	void clearLineOrderPart();
	void sendKey(QKeyEvent* event);

private slots:
	void receiveTotalLineNum(int num1, int num2);
	void receiveLineToDeleteNum(int num);
	void receiveUnfinishSign(bool sign);
	//void sliderValuechanged(int value);
	void confirmSelectBtnAction();
	void getSelectedLines(vector<int> line_order);
	void onStateChanged(int state);
	void cleanScrollArea();
	void selectModeAction();
	void getDeletedLinesIndexList(vector<bool> deleted_lines_index_list);
	void updateTheBackground(int num);
	void alphaChanged(int value);
	void showParaLabel();

private:
	QScrollArea* m_ScrollArea;
	QVBoxLayout* m_VBoxLayout;
	bool is_show_para_label = false;
};

#endif // MAINWINDOW_H
