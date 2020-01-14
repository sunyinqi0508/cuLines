/****************************************************************************
** Meta object code from reading C++ file 'LightParameters.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.11.0)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../LightParameters.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'LightParameters.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.11.0. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_LightParameters_t {
    QByteArrayData data[25];
    char stringdata0[247];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_LightParameters_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_LightParameters_t qt_meta_stringdata_LightParameters = {
    {
QT_MOC_LITERAL(0, 0, 15), // "LightParameters"
QT_MOC_LITERAL(1, 16, 19), // "sendLightParameters"
QT_MOC_LITERAL(2, 36, 0), // ""
QT_MOC_LITERAL(3, 37, 2), // "id"
QT_MOC_LITERAL(4, 40, 5), // "value"
QT_MOC_LITERAL(5, 46, 9), // "sendColor"
QT_MOC_LITERAL(6, 56, 1), // "c"
QT_MOC_LITERAL(7, 58, 12), // "sendFlowPara"
QT_MOC_LITERAL(8, 71, 14), // "getChangeValue"
QT_MOC_LITERAL(9, 86, 13), // "getButtonName"
QT_MOC_LITERAL(10, 100, 5), // "getHS"
QT_MOC_LITERAL(11, 106, 1), // "h"
QT_MOC_LITERAL(12, 108, 1), // "s"
QT_MOC_LITERAL(13, 110, 4), // "getV"
QT_MOC_LITERAL(14, 115, 1), // "v"
QT_MOC_LITERAL(15, 117, 13), // "getPosZSlider"
QT_MOC_LITERAL(16, 131, 5), // "max_z"
QT_MOC_LITERAL(17, 137, 6), // "getTop"
QT_MOC_LITERAL(18, 144, 19), // "spinboxValueChanged"
QT_MOC_LITERAL(19, 164, 12), // "applyThePara"
QT_MOC_LITERAL(20, 177, 18), // "sliderValuechanged"
QT_MOC_LITERAL(21, 196, 12), // "setFlowParas"
QT_MOC_LITERAL(22, 209, 14), // "vector<double>"
QT_MOC_LITERAL(23, 224, 10), // "paras_list"
QT_MOC_LITERAL(24, 235, 11) // "setAllParas"

    },
    "LightParameters\0sendLightParameters\0"
    "\0id\0value\0sendColor\0c\0sendFlowPara\0"
    "getChangeValue\0getButtonName\0getHS\0h\0"
    "s\0getV\0v\0getPosZSlider\0max_z\0getTop\0"
    "spinboxValueChanged\0applyThePara\0"
    "sliderValuechanged\0setFlowParas\0"
    "vector<double>\0paras_list\0setAllParas"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_LightParameters[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
      14,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       3,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    2,   84,    2, 0x06 /* Public */,
       5,    1,   89,    2, 0x06 /* Public */,
       7,    2,   92,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
       8,    1,   97,    2, 0x0a /* Public */,
       9,    0,  100,    2, 0x0a /* Public */,
      10,    2,  101,    2, 0x0a /* Public */,
      13,    1,  106,    2, 0x0a /* Public */,
      15,    1,  109,    2, 0x0a /* Public */,
      17,    0,  112,    2, 0x0a /* Public */,
      18,    1,  113,    2, 0x0a /* Public */,
      19,    0,  116,    2, 0x0a /* Public */,
      20,    1,  117,    2, 0x0a /* Public */,
      21,    1,  120,    2, 0x0a /* Public */,
      24,    0,  123,    2, 0x0a /* Public */,

 // signals: parameters
    QMetaType::Void, QMetaType::Int, QMetaType::Float,    3,    4,
    QMetaType::Void, QMetaType::QColor,    6,
    QMetaType::Void, QMetaType::Int, QMetaType::Double,    3,    4,

 // slots: parameters
    QMetaType::Void, QMetaType::Int,    4,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Int, QMetaType::Int,   11,   12,
    QMetaType::Void, QMetaType::Int,   14,
    QMetaType::Void, QMetaType::Int,   16,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Double,    4,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Int,    4,
    QMetaType::Void, 0x80000000 | 22,   23,
    QMetaType::Void,

       0        // eod
};

void LightParameters::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        LightParameters *_t = static_cast<LightParameters *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->sendLightParameters((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< float(*)>(_a[2]))); break;
        case 1: _t->sendColor((*reinterpret_cast< QColor(*)>(_a[1]))); break;
        case 2: _t->sendFlowPara((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< double(*)>(_a[2]))); break;
        case 3: _t->getChangeValue((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 4: _t->getButtonName(); break;
        case 5: _t->getHS((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        case 6: _t->getV((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 7: _t->getPosZSlider((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 8: _t->getTop(); break;
        case 9: _t->spinboxValueChanged((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 10: _t->applyThePara(); break;
        case 11: _t->sliderValuechanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 12: _t->setFlowParas((*reinterpret_cast< vector<double>(*)>(_a[1]))); break;
        case 13: _t->setAllParas(); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            using _t = void (LightParameters::*)(int , float );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&LightParameters::sendLightParameters)) {
                *result = 0;
                return;
            }
        }
        {
            using _t = void (LightParameters::*)(QColor );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&LightParameters::sendColor)) {
                *result = 1;
                return;
            }
        }
        {
            using _t = void (LightParameters::*)(int , double );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&LightParameters::sendFlowPara)) {
                *result = 2;
                return;
            }
        }
    }
}

QT_INIT_METAOBJECT const QMetaObject LightParameters::staticMetaObject = {
    { &QWidget::staticMetaObject, qt_meta_stringdata_LightParameters.data,
      qt_meta_data_LightParameters,  qt_static_metacall, nullptr, nullptr}
};


const QMetaObject *LightParameters::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *LightParameters::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_LightParameters.stringdata0))
        return static_cast<void*>(this);
    return QWidget::qt_metacast(_clname);
}

int LightParameters::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 14)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 14;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 14)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 14;
    }
    return _id;
}

// SIGNAL 0
void LightParameters::sendLightParameters(int _t1, float _t2)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void LightParameters::sendColor(QColor _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 1, _a);
}

// SIGNAL 2
void LightParameters::sendFlowPara(int _t1, double _t2)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)) };
    QMetaObject::activate(this, &staticMetaObject, 2, _a);
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
