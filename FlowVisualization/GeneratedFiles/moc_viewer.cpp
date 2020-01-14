/****************************************************************************
** Meta object code from reading C++ file 'viewer.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.11.0)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../viewer.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'viewer.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.11.0. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_Viewer_t {
    QByteArrayData data[55];
    char stringdata0[659];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_Viewer_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_Viewer_t qt_meta_stringdata_Viewer = {
    {
QT_MOC_LITERAL(0, 0, 6), // "Viewer"
QT_MOC_LITERAL(1, 7, 16), // "sendTotalLineNum"
QT_MOC_LITERAL(2, 24, 0), // ""
QT_MOC_LITERAL(3, 25, 4), // "num1"
QT_MOC_LITERAL(4, 30, 4), // "num2"
QT_MOC_LITERAL(5, 35, 19), // "sendLineToDeleteNum"
QT_MOC_LITERAL(6, 55, 3), // "num"
QT_MOC_LITERAL(7, 59, 16), // "sendUnfinishSign"
QT_MOC_LITERAL(8, 76, 4), // "sign"
QT_MOC_LITERAL(9, 81, 22), // "sendSelectedLinesIndex"
QT_MOC_LITERAL(10, 104, 11), // "vector<int>"
QT_MOC_LITERAL(11, 116, 4), // "vec1"
QT_MOC_LITERAL(12, 121, 4), // "vec2"
QT_MOC_LITERAL(13, 126, 17), // "sendSelectedLines"
QT_MOC_LITERAL(14, 144, 10), // "line_order"
QT_MOC_LITERAL(15, 155, 5), // "reset"
QT_MOC_LITERAL(16, 161, 25), // "sendDeletedLinesIndexList"
QT_MOC_LITERAL(17, 187, 12), // "vector<bool>"
QT_MOC_LITERAL(18, 200, 24), // "deleted_lines_index_list"
QT_MOC_LITERAL(19, 225, 14), // "sendPosZSlider"
QT_MOC_LITERAL(20, 240, 5), // "max_z"
QT_MOC_LITERAL(21, 246, 7), // "sendTop"
QT_MOC_LITERAL(22, 254, 12), // "scaleTimeout"
QT_MOC_LITERAL(23, 267, 9), // "startMove"
QT_MOC_LITERAL(24, 277, 16), // "receiveLineOrder"
QT_MOC_LITERAL(25, 294, 5), // "order"
QT_MOC_LITERAL(26, 300, 15), // "getLineToDelete"
QT_MOC_LITERAL(27, 316, 18), // "getSelectedOneLine"
QT_MOC_LITERAL(28, 335, 12), // "line_id_list"
QT_MOC_LITERAL(29, 348, 25), // "getSelectedLinesIndexList"
QT_MOC_LITERAL(30, 374, 4), // "list"
QT_MOC_LITERAL(31, 379, 13), // "getParameters"
QT_MOC_LITERAL(32, 393, 12), // "vector<int>*"
QT_MOC_LITERAL(33, 406, 13), // "line_division"
QT_MOC_LITERAL(34, 420, 4), // "int*"
QT_MOC_LITERAL(35, 425, 8), // "linearKd"
QT_MOC_LITERAL(36, 434, 11), // "linearKD_id"
QT_MOC_LITERAL(37, 446, 22), // "clearLineOrderPartList"
QT_MOC_LITERAL(38, 469, 17), // "getDevicePointers"
QT_MOC_LITERAL(39, 487, 2), // "dp"
QT_MOC_LITERAL(40, 490, 12), // "alphaChanged"
QT_MOC_LITERAL(41, 503, 5), // "value"
QT_MOC_LITERAL(42, 509, 16), // "modeChangeAction"
QT_MOC_LITERAL(43, 526, 16), // "selectAreaAction"
QT_MOC_LITERAL(44, 543, 13), // "keyPressEvent"
QT_MOC_LITERAL(45, 557, 10), // "QKeyEvent*"
QT_MOC_LITERAL(46, 568, 3), // "key"
QT_MOC_LITERAL(47, 572, 15), // "moveLightAction"
QT_MOC_LITERAL(48, 588, 18), // "getLightParameters"
QT_MOC_LITERAL(49, 607, 2), // "id"
QT_MOC_LITERAL(50, 610, 8), // "getColor"
QT_MOC_LITERAL(51, 619, 1), // "c"
QT_MOC_LITERAL(52, 621, 11), // "resetAction"
QT_MOC_LITERAL(53, 633, 12), // "saveSettings"
QT_MOC_LITERAL(54, 646, 12) // "loadSettings"

    },
    "Viewer\0sendTotalLineNum\0\0num1\0num2\0"
    "sendLineToDeleteNum\0num\0sendUnfinishSign\0"
    "sign\0sendSelectedLinesIndex\0vector<int>\0"
    "vec1\0vec2\0sendSelectedLines\0line_order\0"
    "reset\0sendDeletedLinesIndexList\0"
    "vector<bool>\0deleted_lines_index_list\0"
    "sendPosZSlider\0max_z\0sendTop\0scaleTimeout\0"
    "startMove\0receiveLineOrder\0order\0"
    "getLineToDelete\0getSelectedOneLine\0"
    "line_id_list\0getSelectedLinesIndexList\0"
    "list\0getParameters\0vector<int>*\0"
    "line_division\0int*\0linearKd\0linearKD_id\0"
    "clearLineOrderPartList\0getDevicePointers\0"
    "dp\0alphaChanged\0value\0modeChangeAction\0"
    "selectAreaAction\0keyPressEvent\0"
    "QKeyEvent*\0key\0moveLightAction\0"
    "getLightParameters\0id\0getColor\0c\0"
    "resetAction\0saveSettings\0loadSettings"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_Viewer[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
      28,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       9,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    2,  154,    2, 0x06 /* Public */,
       5,    1,  159,    2, 0x06 /* Public */,
       7,    1,  162,    2, 0x06 /* Public */,
       9,    2,  165,    2, 0x06 /* Public */,
      13,    1,  170,    2, 0x06 /* Public */,
      15,    0,  173,    2, 0x06 /* Public */,
      16,    1,  174,    2, 0x06 /* Public */,
      19,    1,  177,    2, 0x06 /* Public */,
      21,    0,  180,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
      22,    0,  181,    2, 0x08 /* Private */,
      23,    0,  182,    2, 0x08 /* Private */,
      24,    2,  183,    2, 0x08 /* Private */,
      26,    1,  188,    2, 0x08 /* Private */,
      27,    1,  191,    2, 0x08 /* Private */,
      29,    1,  194,    2, 0x08 /* Private */,
      31,    3,  197,    2, 0x08 /* Private */,
      37,    0,  204,    2, 0x08 /* Private */,
      38,    1,  205,    2, 0x08 /* Private */,
      40,    1,  208,    2, 0x08 /* Private */,
      42,    0,  211,    2, 0x08 /* Private */,
      43,    0,  212,    2, 0x08 /* Private */,
      44,    1,  213,    2, 0x08 /* Private */,
      47,    0,  216,    2, 0x08 /* Private */,
      48,    2,  217,    2, 0x08 /* Private */,
      50,    1,  222,    2, 0x08 /* Private */,
      52,    0,  225,    2, 0x08 /* Private */,
      53,    0,  226,    2, 0x08 /* Private */,
      54,    0,  227,    2, 0x08 /* Private */,

 // signals: parameters
    QMetaType::Void, QMetaType::Int, QMetaType::Int,    3,    4,
    QMetaType::Void, QMetaType::Int,    6,
    QMetaType::Void, QMetaType::Bool,    8,
    QMetaType::Void, 0x80000000 | 10, 0x80000000 | 10,   11,   12,
    QMetaType::Void, 0x80000000 | 10,   14,
    QMetaType::Void,
    QMetaType::Void, 0x80000000 | 17,   18,
    QMetaType::Void, QMetaType::Int,   20,
    QMetaType::Void,

 // slots: parameters
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, 0x80000000 | 10, QMetaType::Int,   25,    8,
    QMetaType::Void, QMetaType::Int,    6,
    QMetaType::Void, 0x80000000 | 10,   28,
    QMetaType::Void, 0x80000000 | 10,   30,
    QMetaType::Void, 0x80000000 | 32, 0x80000000 | 34, 0x80000000 | 34,   33,   35,   36,
    QMetaType::Void,
    QMetaType::Void, QMetaType::VoidStar,   39,
    QMetaType::Void, QMetaType::Int,   41,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, 0x80000000 | 45,   46,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Int, QMetaType::Float,   49,   41,
    QMetaType::Void, QMetaType::QColor,   51,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,

       0        // eod
};

void Viewer::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Viewer *_t = static_cast<Viewer *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->sendTotalLineNum((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        case 1: _t->sendLineToDeleteNum((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 2: _t->sendUnfinishSign((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 3: _t->sendSelectedLinesIndex((*reinterpret_cast< vector<int>(*)>(_a[1])),(*reinterpret_cast< vector<int>(*)>(_a[2]))); break;
        case 4: _t->sendSelectedLines((*reinterpret_cast< vector<int>(*)>(_a[1]))); break;
        case 5: _t->reset(); break;
        case 6: _t->sendDeletedLinesIndexList((*reinterpret_cast< vector<bool>(*)>(_a[1]))); break;
        case 7: _t->sendPosZSlider((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 8: _t->sendTop(); break;
        case 9: _t->scaleTimeout(); break;
        case 10: _t->startMove(); break;
        case 11: _t->receiveLineOrder((*reinterpret_cast< vector<int>(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        case 12: _t->getLineToDelete((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 13: _t->getSelectedOneLine((*reinterpret_cast< vector<int>(*)>(_a[1]))); break;
        case 14: _t->getSelectedLinesIndexList((*reinterpret_cast< vector<int>(*)>(_a[1]))); break;
        case 15: _t->getParameters((*reinterpret_cast< vector<int>*(*)>(_a[1])),(*reinterpret_cast< int*(*)>(_a[2])),(*reinterpret_cast< int*(*)>(_a[3]))); break;
        case 16: _t->clearLineOrderPartList(); break;
        case 17: _t->getDevicePointers((*reinterpret_cast< void*(*)>(_a[1]))); break;
        case 18: _t->alphaChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 19: _t->modeChangeAction(); break;
        case 20: _t->selectAreaAction(); break;
        case 21: _t->keyPressEvent((*reinterpret_cast< QKeyEvent*(*)>(_a[1]))); break;
        case 22: _t->moveLightAction(); break;
        case 23: _t->getLightParameters((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< float(*)>(_a[2]))); break;
        case 24: _t->getColor((*reinterpret_cast< QColor(*)>(_a[1]))); break;
        case 25: _t->resetAction(); break;
        case 26: _t->saveSettings(); break;
        case 27: _t->loadSettings(); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            using _t = void (Viewer::*)(int , int );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Viewer::sendTotalLineNum)) {
                *result = 0;
                return;
            }
        }
        {
            using _t = void (Viewer::*)(int );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Viewer::sendLineToDeleteNum)) {
                *result = 1;
                return;
            }
        }
        {
            using _t = void (Viewer::*)(bool );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Viewer::sendUnfinishSign)) {
                *result = 2;
                return;
            }
        }
        {
            using _t = void (Viewer::*)(vector<int> , vector<int> );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Viewer::sendSelectedLinesIndex)) {
                *result = 3;
                return;
            }
        }
        {
            using _t = void (Viewer::*)(vector<int> );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Viewer::sendSelectedLines)) {
                *result = 4;
                return;
            }
        }
        {
            using _t = void (Viewer::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Viewer::reset)) {
                *result = 5;
                return;
            }
        }
        {
            using _t = void (Viewer::*)(vector<bool> );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Viewer::sendDeletedLinesIndexList)) {
                *result = 6;
                return;
            }
        }
        {
            using _t = void (Viewer::*)(int );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Viewer::sendPosZSlider)) {
                *result = 7;
                return;
            }
        }
        {
            using _t = void (Viewer::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Viewer::sendTop)) {
                *result = 8;
                return;
            }
        }
    }
}

QT_INIT_METAOBJECT const QMetaObject Viewer::staticMetaObject = {
    { &QOpenGLWidget::staticMetaObject, qt_meta_stringdata_Viewer.data,
      qt_meta_data_Viewer,  qt_static_metacall, nullptr, nullptr}
};


const QMetaObject *Viewer::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *Viewer::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_Viewer.stringdata0))
        return static_cast<void*>(this);
    return QOpenGLWidget::qt_metacast(_clname);
}

int Viewer::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QOpenGLWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 28)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 28;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 28)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 28;
    }
    return _id;
}

// SIGNAL 0
void Viewer::sendTotalLineNum(int _t1, int _t2)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void Viewer::sendLineToDeleteNum(int _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 1, _a);
}

// SIGNAL 2
void Viewer::sendUnfinishSign(bool _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 2, _a);
}

// SIGNAL 3
void Viewer::sendSelectedLinesIndex(vector<int> _t1, vector<int> _t2)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)) };
    QMetaObject::activate(this, &staticMetaObject, 3, _a);
}

// SIGNAL 4
void Viewer::sendSelectedLines(vector<int> _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 4, _a);
}

// SIGNAL 5
void Viewer::reset()
{
    QMetaObject::activate(this, &staticMetaObject, 5, nullptr);
}

// SIGNAL 6
void Viewer::sendDeletedLinesIndexList(vector<bool> _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 6, _a);
}

// SIGNAL 7
void Viewer::sendPosZSlider(int _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 7, _a);
}

// SIGNAL 8
void Viewer::sendTop()
{
    QMetaObject::activate(this, &staticMetaObject, 8, nullptr);
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
