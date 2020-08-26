/* info: this file defines the interface between Pyhton and the C-Lib */

#ifndef CHESSSLIBMODULE_H
#define CHESSSLIBMODULE_H

/* ====================================================
      I N I T I A L I Z E    P Y T H O N   T O O L S
   ==================================================== */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

/* TODO: use a non-deprecated version instead of removing the deprecation warning!!! */
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

/* ====================================================
            L O A D   C H E S S   L I B S
   ==================================================== */

#include "chesspiece.h"
#include "chessposition.h"
#include "chessboard.h"
#include "chesspieceatpos.h"
#include "chessdraw.h"
#include "chessdrawgen.h"

PyObject* chesslib_create_chessposition(PyObject* self, PyObject* args);
PyObject* chesslib_create_chesspiece(PyObject* self, PyObject* args);
PyObject* chesslib_create_chesspieceatpos(PyObject* self, PyObject* args);
PyObject* chesslib_create_chessboard(PyObject* self, PyObject* args);
PyObject* chesslib_create_chessboard_startformation(PyObject* self, PyObject* args);
PyObject* chesslib_create_chessdraw(PyObject* self, PyObject* args);
PyObject* chesslib_get_all_draws(PyObject* self, PyObject* args);

static PyMethodDef chesslib_methods[] = {
    {"ChessBoard", chesslib_create_chessboard, METH_VARARGS, "Python interface for creating a new chess board using a C library function"},
    {"ChessBoard_StartFormation", chesslib_create_chessboard_startformation, METH_VARARGS, "Python interface for creating a new chess board in start formation using a C library function"},
    {"ChessDraw", chesslib_create_chessdraw, METH_VARARGS, "Python interface for creating a new chess draw using a C library function"},
    {"ChessPiece", chesslib_create_chesspiece, METH_VARARGS, "Python interface for creating a new chess piece using a C library function"},
    {"ChessPosition", chesslib_create_chessposition, METH_VARARGS, "Python interface for creating a new chess position using a C library function"},
    {"ChessPieceAtPos", chesslib_create_chesspieceatpos, METH_VARARGS, "Python interface for creating a new chess piece including its' position using a C library function"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef chesslib_module = {
    PyModuleDef_HEAD_INIT,
    "chesslib",
    "Python interface for efficient chess draw-gen C library functions",
    -1,
    chesslib_methods
};

PyMODINIT_FUNC PyInit_chesslib(void);

/* TODO: add missing makros, constants, method stubs, etc. */

#endif