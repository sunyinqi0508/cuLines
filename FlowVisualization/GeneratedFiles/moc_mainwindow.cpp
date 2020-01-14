/****************************************************************************
** Meta object code from reading C++ file 'mainwindow.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.11.0)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../mainwindow.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'mainwindow.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.11.0. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_MainWindow_t {
    QByteArrayData data[32];
    char stringdata0[436];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_MainWindow_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_MainWindow_t qt_meta_stringdata_MainWindow = {
    {
QT_MOC_LITERAL(0, 0, 10), // "MainWindow"
QT_MOC_LITERAL(1, 11, 19), // "sendSelectedOneLine"
QT_MOC_LITERAL(2, 31, 0), // ""
QT_MOC_LITERAL(3, 32, 11), // "vector<int>"
QT_MOC_LITERAL(4, 44, 12), // "line_id_list"
QT_MOC_LITERAL(5, 57, 26), // "sendSelectedLinesIndexList"
QT_MOC_LITERAL(6, 84, 4), // "list"
QT_MOC_LITERAL(7, 89, 18), // "clearLineOrderPart"
QT_MOC_LITERAL(8, 108, 7), // "sendKey"
QT_MOC_LITERAL(9, 116, 10), // "QKeyEvent*"
QT_MOC_LITERAL(10, 127, 5), // "event"
QT_MOC_LITERAL(11, 133, 19), // "receiveTotalLineNum"
QT_MOC_LITERAL(12, 153, 4), // "num1"
QT_MOC_LITERAL(13, 158, 4), // "num2"
QT_MOC_LITERAL(14, 163, 22), // "receiveLineToDeleteNum"
QT_MOC_LITERAL(15, 186, 3), // "num"
QT_MOC_LITERAL(16, 190, 19), // "receiveUnfinishSign"
QT_MOC_LITERAL(17, 210, 4), // "sign"
QT_MOC_LITERAL(18, 215, 22), // "confirmSelectBtnAction"
QT_MOC_LITERAL(19, 238, 16), // "getSelectedLines"
QT_MOC_LITERAL(20, 255, 10), // "line_order"
QT_MOC_LITERAL(21, 266, 14), // "onStateChanged"
QT_MOC_LITERAL(22, 281, 5), // "state"
QT_MOC_LITERAL(23, 287, 15), // "cleanScrollArea"
QT_MOC_LITERAL(24, 303, 16), // "selectModeAction"
QT_MOC_LITERAL(25, 320, 24), // "getDeletedLinesIndexList"
QT_MOC_LITERAL(26, 345, 12), // "vector<bool>"
QT_MOC_LITERAL(27, 358, 24), // "deleted_lines_index_list"
QT_MOC_LITERAL(28, 383, 19), // "updateTheBackground"
QT_MOC_LITERAL(29, 403, 12), // "alphaChanged"
QT_MOC_LITERAL(30, 416, 5), // "value"
QT_MOC_LITERAL(31, 422, 13) // "showParaLabel"

    },
    "MainWindow\0sendSelectedOneLine\0\0"
    "vector<int>\0line_id_list\0"
    "sendSelectedLinesIndexList\0list\0"
    "clearLineOrderPart\0sendKey\0QKeyEvent*\0"
    "event\0receiveTotalLineNum\0num1\0num2\0"
    "receiveLineToDeleteNum\0num\0"
    "receiveUnfinishSign\0sign\0"
    "confirmSelectBtnAction\0getSelectedLines\0"
    "line_order\0onStateChanged\0state\0"
    "cleanScrollArea\0selectModeAction\0"
    "getDeletedLinesIndexList\0vector<bool>\0"
    "deleted_lines_index_list\0updateTheBackground\0"
    "alphaChanged\0value\0showParaLabel"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_MainWindow[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
      16,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       4,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    1,   94,    2, 0x06 /* Public */,
       5,    1,   97,    2, 0x06 /* Public */,
       7,    0,  100,    2, 0x06 /* Public */,
       8,    1,  101,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
      11,    2,  104,    2, 0x08 /* Private */,
      14,    1,  109,    2, 0x08 /* Private */,
      16,    1,  112,    2, 0x08 /* Private */,
      18,    0,  115,    2, 0x08 /* Private */,
      19,    1,  116,    2, 0x08 /* Private */,
      21,    1,  119,    2, 0x08 /* Private */,
      23,    0,  122,    2, 0x08 /* Private */,
      24,    0,  123,    2, 0x08 /* Private */,
      25,    1,  124,    2, 0x08 /* Private */,
      28,    1,  127,    2, 0x08 /* Private */,
      29,    1,  130,    2, 0x08 /* Private */,
      31,    0,  133,    2, 0x08 /* Private */,

 // signals: parameters
    QMetaType::Void, 0x80000000 | 3,    4,
    QMetaType::Void, 0x80000000 | 3,    6,
    QMetaType::Void,
    QMetaType::Void, 0x80000000 | 9,   10,

 // slots: parameters
    QMetaType::Void, QMetaType::Int, QMetaType::Int,   12,   13,
    QMetaType::Void, QMetaType::Int,   15,
    QMetaType::Void, QMetaType::Bool,   17,
    QMetaType::Void,
    QMetaType::Void, 0x80000000 | 3,   20,
    QMetaType::Void, QMetaType::Int,   22,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, 0x80000000 | 26,   27,
    QMetaType::Void, QMetaType::Int,   15,
    QMetaType::Void, QMetaType::Int,   30,
    QMetaType::Void,

       0        // eod
};

void MainWindow::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        MainWindow *_t = static_cast<MainWindow *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->sendSelectedOneLine((*reinterpret_cast< vector<int>(*)>(_a[1]))); break;
        case 1: _t->sendSelectedLinesIndexList((*reinterpret_cast< vector<int>(*)>(_a[1]))); break;
        case 2: _t->clearLineOrderPart(); break;
        case 3: _t->sendKey((*reinterpret_cast< QKeyEvent*(*)>(_a[1]))); break;
        case 4: _t->receiveTotalLineNum((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        case 5: _t->receiveLineToDeleteNum((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 6: _t->receiveUnfinishSign((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 7: _t->confirmSelectBtnAction(); break;
        case 8: _t->getSelectedLines((*reinterpret_cast< vector<int>(*)>(_a[1]))); break;
        case 9: _t->onStateChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 10: _t->cleanScrollArea(); break;
        case 11: _t->selectModeAction(); break;
        case 12: _t->getDeletedLinesIndexList((*reinterpret_cast< vector<bool>(*)>(_a[1]))); break;
        case 13: _t->updateTheBackground((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 14: _t->alphaChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 15: _t->showParaLabel(); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            using _t = void (MainWindow::*)(vector<int> );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&MainWindow::sendSelectedOneLine)) {
                *result = 0;
                return;
            }
        }
        {
            using _t = void (MainWindow::*)(vector<int> );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&MainWindow::sendSelectedLinesIndexList)) {
                *result = 1;
                return;
            }
        }
        {
            using _t = void (MainWindow::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&MainWindow::clearLineOrderPart)) {
                *result = 2;
                return;
            }
        }
        {
            using _t = void (MainWindow::*)(QKeyEvent * );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&MainWindow::sendKey)) {
                *result = 3;
                return;
            }
        }
    }
}

QT_INIT_METAOBJECT const QMetaObject MainWindow::staticMetaObject = {
    { &QMainWindow::staticMetaObject, qt_meta_stringdata_MainWindow.data,
      qt_meta_data_MainWindow,  qt_static_metacall, nullptr, nullptr}
};


const QMetaObject *MainWindow::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *MainWindow::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_MainWindow.stringdata0))
        return static_cast<void*>(this);
    return QMainWindow::qt_metacast(_clname);
}

int MainWindow::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QMainWindow::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 16)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 16;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 16)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 16;
    }
    return _id;
}

// SIGNAL 0
void MainWindow::sendSelectedOneLine(vector<int> _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void MainWindow::sendSelectedLinesIndexList(vector<int> _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 1, _a);
}

// SIGNAL 2
void MainWindow::clearLineOrderPart()
{
    QMetaObject::activate(this, &staticMetaObject, 2, nullptr);
}

// SIGNAL 3
void MainWindow::sendKey(QKeyEvent * _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 3, _a);
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
