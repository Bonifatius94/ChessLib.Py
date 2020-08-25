/* info: this file defines the interface between Pyhton and the C-Lib */

#ifndef CHESSSLIBMODULE_H
#define CHESSSLIBMODULE_H

/* ====================================================
      I N I T I A L I Z E    P Y T H O N   T O O L S
   ==================================================== */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

/* ====================================================
            L O A D   C H E S S   L I B S
   ==================================================== */

#include "chesspiece.h"
#include "chessposition.h"
#include "chessboard.h"
#include "chesspieceatpos.h"
#include "chessdraw.h"
#include "chessdrawgen.h"

PyObject* PyInit_chesslib();

/* TODO: add missing makros, constants, method stubs, etc. */

#endif