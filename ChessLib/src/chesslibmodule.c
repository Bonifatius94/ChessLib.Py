
#include "chesslibmodule.h"

static PyObject*

/* TODO: implement python-lib interface here ... */

chesslib_create(PyObject* self, PyObject* args)
{
    const char* command;
    int sts;

    if (!PyArg_ParseTuple(args, "s", &command))
        return NULL;

    sts = system(command);

    return PyLong_FromLong(sts);
}