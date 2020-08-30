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

static PyObject* chesslib_create_chesscolor_white(PyObject* self, PyObject* args);
static PyObject* chesslib_create_chesscolor_black(PyObject* self, PyObject* args);
static PyObject* chesslib_create_chessposition(PyObject* self, PyObject* args);
static PyObject* chesslib_create_chesspiece(PyObject* self, PyObject* args);
static PyObject* chesslib_create_chesspieceatpos(PyObject* self, PyObject* args);
static PyObject* chesslib_create_chessboard(PyObject* self, PyObject* args);
static PyObject* chesslib_create_chessboard_startformation(PyObject* self, PyObject* args);
static PyObject* chesslib_create_chessdraw(PyObject* self, PyObject* args);
static PyObject* chesslib_create_chessdraw_null(PyObject* self, PyObject* args);
static PyObject* chesslib_get_all_draws(PyObject* self, PyObject* args);

PyMODINIT_FUNC PyInit_chesslib(void);

/* TODO: add missing makros, constants, method stubs, etc. */

#endif
