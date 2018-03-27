
#include "ColorLuminancePicker.h"

#include <QColor>
#include <QMouseEvent>
#include <QPainter>
#include <QPaintEvent>
#include <qdrawutil.h>

/**************************************************************
 *             construct
 **************************************************************/

ColorLuminancePicker::ColorLuminancePicker(QWidget* parent)
    : QWidget(parent)
    , hue_(0)
    , sat_(0)
    , val_(0)
    , pix_(0)
{
    setSizePolicy(QSizePolicy(QSizePolicy::Fixed, QSizePolicy::Preferred));
    setMinimumSize(10, 50);
}

ColorLuminancePicker::~ColorLuminancePicker() {
    delete pix_;
}



/**************************************************************
 *             public slot
 **************************************************************/
void ColorLuminancePicker::setCol(int h, int s) {
    setCol(h, s, val_);
    //emit newHsv(h, s, val_);
}

void ColorLuminancePicker::setCol(const QColor& c) {
    setCol(c.hue(), c.saturation(), c.value());
}



/**************************************************************
 *             event
 **************************************************************/

void ColorLuminancePicker::mouseMoveEvent(QMouseEvent* event) {
    setVal(y2val(event->y()));
}

void ColorLuminancePicker::mousePressEvent(QMouseEvent* event) {
  //  emit toggleInteractionMode(true);
    setVal(y2val(event->y()));
}

void ColorLuminancePicker::mouseReleaseEvent(QMouseEvent* event) {
    event->accept();
//    emit toggleInteractionMode(false);
}


void ColorLuminancePicker::paintEvent(QPaintEvent* /*event*/) {
    int w = width() - 5;

    // start point is at left up corner

    QRect r(0, foff, w, height() - 2*foff);
    int wi = r.width() - 2;
    int hi = r.height() - 2;
    if (wi <= 1 || hi <= 1)
        return;
    if ((pix_ == 0) || (pix_->height() != hi) || (pix_->width() != wi)) {
        if (!pix_)
            delete pix_;
        QImage img(wi, hi, QImage::Format_RGB32);

        for (int y = 0 ; y < hi ; ++y) {
            QColor c;
            c.setHsv(hue_, sat_, y2val(y+coff));
            QRgb r = c.rgb();
            for (int x = 0 ; x < wi ; ++x)
                img.setPixel(x, y, r);
        }
        pix_ = new QPixmap(QPixmap::fromImage(img));
    }
    // color bar
    QPainter p(this);
    p.drawPixmap(1, coff, *pix_);
    const QPalette &g = palette();
    qDrawShadePanel(&p, r, g, true);
    p.setPen(g.foreground().color());
    p.setBrush(g.foreground());

    // arrow
    QPolygon a;
    int y = val2y(val_);
    a.setPoints(3, w, y, w+5, y+5, w+5, y-5);

    // erase the right bar color
    p.eraseRect(w, 0, 5, height());
    p.drawPolygon(a);
}


/**************************************************************
 *                    private
 **************************************************************/

// set the hsv color
void ColorLuminancePicker::setCol(int h, int s , int v) {
    val_ = v;
    hue_ = h;
    sat_ = s;
    delete pix_;
    pix_ = 0;
    update();
}



void ColorLuminancePicker::setVal(int v) {
    if (val_ == v)
        return;

    val_ = qMax(0, qMin(v,255));

    delete pix_;
    pix_ = 0;

    update();

   // emit newHsv(hue_, sat_, val_);
//	emit sendV(val_);
}


int ColorLuminancePicker::y2val(int y) {
    int d = height() - 2*coff - 1;
    if (d != 0)
        return 255 - (y - coff)*255/d;
    else
        return 0;
}

int ColorLuminancePicker::val2y(int v) {
    int d = height() - 2*coff - 1;
    return coff + (255-v)*d/255;
}



