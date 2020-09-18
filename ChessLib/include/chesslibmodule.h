/*
 * MIT License
 * 
 * Copyright(c) 2020 Marco Tröster
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

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

static PyObject* chesslib_create_chessposition(PyObject* self, PyObject* args);
static PyObject* chesslib_create_chesspiece(PyObject* self, PyObject* args);
static PyObject* chesslib_create_chesspieceatpos(PyObject* self, PyObject* args);
static PyObject* chesslib_create_chessboard(PyObject* self, PyObject* args);
static PyObject* chesslib_create_chessboard_startformation(PyObject* self, PyObject* args);
static PyObject* chesslib_create_chessdraw(PyObject* self, PyObject* args);
static PyObject* chesslib_get_all_draws(PyObject* self, PyObject* args);
static PyObject* chesslib_board_to_hash(PyObject* self, PyObject* args);
static PyObject* chesslib_apply_draw(PyObject* self, PyObject* args);
static PyObject* chesslib_get_game_state(PyObject* self, PyObject* args);


PyMODINIT_FUNC PyInit_chesslib(void);

/* TODO: add missing makros, constants, method stubs, etc. */

#endif
