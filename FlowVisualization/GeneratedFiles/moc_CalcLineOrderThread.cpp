/****************************************************************************
** Meta object code from reading C++ file 'CalcLineOrderThread.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.11.0)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../CalcLineOrderThread.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'CalcLineOrderThread.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.11.0. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_CalcLineOrderThread_t {
    QByteArrayData data[17];
    char stringdata0[187];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_CalcLineOrderThread_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_CalcLineOrderThread_t qt_meta_stringdata_CalcLineOrderThread = {
    {
QT_MOC_LITERAL(0, 0, 19), // "CalcLineOrderThread"
QT_MOC_LITERAL(1, 20, 13), // "sendLineOrder"
QT_MOC_LITERAL(2, 34, 0), // ""
QT_MOC_LITERAL(3, 35, 11), // "vector<int>"
QT_MOC_LITERAL(4, 47, 5), // "order"
QT_MOC_LITERAL(5, 53, 4), // "sign"
QT_MOC_LITERAL(6, 58, 14), // "sendParameters"
QT_MOC_LITERAL(7, 73, 12), // "vector<int>*"
QT_MOC_LITERAL(8, 86, 13), // "line_division"
QT_MOC_LITERAL(9, 100, 4), // "int*"
QT_MOC_LITERAL(10, 105, 8), // "linearKd"
QT_MOC_LITERAL(11, 114, 11), // "linearKD_id"
QT_MOC_LITERAL(12, 126, 18), // "sendDevicePointers"
QT_MOC_LITERAL(13, 145, 13), // "devicepointer"
QT_MOC_LITERAL(14, 159, 17), // "calcCurrLineOrder"
QT_MOC_LITERAL(15, 177, 4), // "vec1"
QT_MOC_LITERAL(16, 182, 4) // "vec2"

    },
    "CalcLineOrderThread\0sendLineOrder\0\0"
    "vector<int>\0order\0sign\0sendParameters\0"
    "vector<int>*\0line_division\0int*\0"
    "linearKd\0linearKD_id\0sendDevicePointers\0"
    "devicepointer\0calcCurrLineOrder\0vec1\0"
    "vec2"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_CalcLineOrderThread[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
       4,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       3,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    2,   34,    2, 0x06 /* Public */,
       6,    3,   39,    2, 0x06 /* Public */,
      12,    1,   46,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
      14,    2,   49,    2, 0x08 /* Private */,

 // signals: parameters
    QMetaType::Void, 0x80000000 | 3, QMetaType::Int,    4,    5,
    QMetaType::Void, 0x80000000 | 7, 0x80000000 | 9, 0x80000000 | 9,    8,   10,   11,
    QMetaType::Void, QMetaType::VoidStar,   13,

 // slots: parameters
    QMetaType::Void, 0x80000000 | 3, 0x80000000 | 3,   15,   16,

       0        // eod
};

void CalcLineOrderThread::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        CalcLineOrderThread *_t = static_cast<CalcLineOrderThread *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->sendLineOrder((*reinterpret_cast< vector<int>(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        case 1: _t->sendParameters((*reinterpret_cast< vector<int>*(*)>(_a[1])),(*reinterpret_cast< int*(*)>(_a[2])),(*reinterpret_cast< int*(*)>(_a[3]))); break;
        case 2: _t->sendDevicePointers((*reinterpret_cast< void*(*)>(_a[1]))); break;
        case 3: _t->calcCurrLineOrder((*reinterpret_cast< vector<int>(*)>(_a[1])),(*reinterpret_cast< vector<int>(*)>(_a[2]))); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            using _t = void (CalcLineOrderThread::*)(vector<int> , int );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&CalcLineOrderThread::sendLineOrder)) {
                *result = 0;
                return;
            }
        }
        {
            using _t = void (CalcLineOrderThread::*)(vector<int> * , int * , int * );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&CalcLineOrderThread::sendParameters)) {
                *result = 1;
                return;
            }
        }
        {
            using _t = void (CalcLineOrderThread::*)(void * );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&CalcLineOrderThread::sendDevicePointers)) {
                *result = 2;
                return;
            }
        }
    }
}

QT_INIT_METAOBJECT const QMetaObject CalcLineOrderThread::staticMetaObject = {
    { &QThread::staticMetaObject, qt_meta_stringdata_CalcLineOrderThread.data,
      qt_meta_data_CalcLineOrderThread,  qt_static_metacall, nullptr, nullptr}
};


const QMetaObject *CalcLineOrderThread::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *CalcLineOrderThread::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_CalcLineOrderThread.stringdata0))
        return static_cast<void*>(this);
    return QThread::qt_metacast(_clname);
}

int CalcLineOrderThread::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QThread::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 4)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 4;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 4)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 4;
    }
    return _id;
}

// SIGNAL 0
void CalcLineOrderThread::sendLineOrder(vector<int> _t1, int _t2)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void CalcLineOrderThread::sendParameters(vector<int> * _t1, int * _t2, int * _t3)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)), const_cast<void*>(reinterpret_cast<const void*>(&_t3)) };
    QMetaObject::activate(this, &staticMetaObject, 1, _a);
}

// SIGNAL 2
void CalcLineOrderThread::sendDevicePointers(void * _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 2, _a);
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
