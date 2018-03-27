
#include "ColorPicker.h"
#include <QColor>
#include <QMouseEvent>
#include <QPainter>
#include <QPaintEvent>




/**************************************************************
 *             construct
 **************************************************************/

ColorPicker::ColorPicker(QWidget* parent)
    : QFrame(parent)
    , hue_(0)
    , sat_(0)
    , pix_(0)
{
    setCol(0, 0);

    setAttribute(Qt::WA_NoSystemBackground);
    setSizePolicy(QSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred));
    setMinimumSize(128, 50);
}

ColorPicker::~ColorPicker() {
    delete pix_;
}



/**************************************************************
 *            public slots
 **************************************************************/



void ColorPicker::setCol(int h, int s) {
    int nhue = qMin(qMax(0,h), 359);
    int nsat = qMin(qMax(0,s), 255);
    if ((nhue == hue_) && (nsat == sat_))
        return;
    QRect r(colPt(), QSize(20,20));
    hue_ = nhue;
    sat_ = nsat;
    r = r.united(QRect(colPt(), QSize(20,20)));
    r.translate(contentsRect().x()-9, contentsRect().y()-9);
    repaint(r);
}

void ColorPicker::setCol(const QColor& c) {
    setCol(c.hue(), c.saturation());
}




/**************************************************************
 *            protected
 **************************************************************/


void ColorPicker::mouseMoveEvent(QMouseEvent* event) {
    QPoint p = event->pos() - contentsRect().topLeft();
    setCol(p);
    emit newCol(hue_, sat_);
}

void ColorPicker::mousePressEvent(QMouseEvent* event) {
    QPoint p = event->pos() - contentsRect().topLeft();
    setCol(p);
   // emit toggleInteractionMode(true);
    emit newCol(hue_, sat_);
}

void ColorPicker::mouseReleaseEvent(QMouseEvent* event) {
    event->accept();
    //emit toggleInteractionMode(false);
}

void ColorPicker::paintEvent(QPaintEvent* /*event*/) {
    QPainter p(this);
    drawFrame(&p);
    QRect r = contentsRect();
    if ((r.width() <= 1) || (r.height() <= 1))
        return;

    int w = r.width();
    int h = r.height();
    if (!pix_ || (pix_->width() != w) || (pix_->height() != h)) {
        QImage img(w, h, QImage::Format_RGB32);
        img.fill(Qt::blue);
        for (int y = 0 ; y < h ; ++y) {
            for (int x = 0 ; x < w ; ++x) {
                QPoint p(x, y);
                QColor c;
                c.setHsv(huePt(p), satPt(p), 200);
                img.setPixel(x, y, c.rgb());
            }
        }
        delete pix_;
        pix_ = new QPixmap(QPixmap::fromImage(img));
    }
    p.drawPixmap(r.topLeft(), *pix_);

    // black cross
    QPoint pt = colPt() + r.topLeft();
    p.setPen(Qt::black);
    p.fillRect(pt.x()-9, pt.y(), 20, 2, Qt::black);
    p.fillRect(pt.x(), pt.y()-9, 2, 20, Qt::black);
}








/**************************************************************
 *            private
 **************************************************************/


QPoint ColorPicker::colPt() {
    return QPoint((360-hue_)*(contentsRect().width()-1)/360, (255-sat_)*(contentsRect().height()-1)/255);
}

int ColorPicker::huePt(const QPoint &pt) {
    if (contentsRect().width()-1 != 0)
        return 360 - pt.x()*360/(contentsRect().width()-1);
    else
        return 0;
}

int ColorPicker::satPt(const QPoint &pt) {
    if (contentsRect().height()-1 != 0)
        return 255 - pt.y()*255/(contentsRect().height()-1) ;
    else
        return 0;
}

void ColorPicker::setCol(const QPoint &pt) {
    setCol(huePt(pt), satPt(pt));
}



