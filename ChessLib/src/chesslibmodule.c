#include "chesslibmodule.h"

/* =================================================
      H E L P E R    F U N C T I O N    S T U B S
   ================================================= */

static PyObject* serialize_chessboard(ChessBoard board);
static ChessBoard deserialize_chessboard(PyObject* board);

/* =================================================
                 I N I T I A L I Z E
              P Y T H O N    M O D U L E
   ================================================= */

/* info: module extension code taken from following tutorial: https://realpython.com/build-python-c-extension-module/ */

#define PY_METHODS_TERM_SYMBOL {NULL, NULL, 0, NULL}

/* Define all functions exposed to python. */
static PyMethodDef chesslib_methods[] = {
    {"GenerateDraws", chesslib_get_all_draws, METH_VARARGS, "Python interface for generating all possible draws for the given position using a C library function"},
    {"ChessBoard", chesslib_create_chessboard, METH_VARARGS, "Python interface for creating a new chess board using a C library function"},
    {"ChessBoard_StartFormation", chesslib_create_chessboard_startformation, METH_NOARGS, "Python interface for creating a new chess board in start formation using a C library function"},
    {"ChessColor_White", chesslib_create_chesscolor_white, METH_NOARGS, "Python interface for creating a new chess draw using a C library function"},
    {"ChessColor_Black", chesslib_create_chesscolor_black, METH_NOARGS, "Python interface for creating a new chess draw using a C library function"},
    {"ChessDraw", chesslib_create_chessdraw, METH_VARARGS, "Python interface for creating a new chess draw using a C library function"},
    {"ChessDraw_Null", chesslib_create_chessdraw_null, METH_NOARGS, "Python interface for creating a null-value chess draw using a C library function"},
    {"ChessPiece", chesslib_create_chesspiece, METH_VARARGS, "Python interface for creating a new chess piece using a C library function"},
    {"ChessPosition", chesslib_create_chessposition, METH_VARARGS, "Python interface for creating a new chess position using a C library function"},
    {"ChessPieceAtPos", chesslib_create_chesspieceatpos, METH_VARARGS, "Python interface for creating a new chess piece including its' position using a C library function"},
    PY_METHODS_TERM_SYMBOL
};

/* Define the chesslib python module. */
static struct PyModuleDef chesslib_module = {
    PyModuleDef_HEAD_INIT,
    "chesslib",
    "Python interface for efficient chess draw-gen C library functions",
    -1,
    chesslib_methods
};

/* Retrieve an instance of the python module. */
PyMODINIT_FUNC PyInit_chesslib(void)
{
    return PyModule_Create(&chesslib_module);
}

/* =================================================
       C R E A T E    D A T A    O B J E C T S

         -    C H E S S    C O L O R
         -    C H E S S    P O S I T I O N
         -    C H E S S    P I E C E
         -    C H E S S    P I E C E    AT   POSITION
         -    C H E S S    B O A R D
         -    C H E S S    D R A W
   ================================================= */

static PyObject* chesslib_create_chesscolor_white(PyObject* self)
{
    /* create uint32 python object and return it */
    return PyLong_FromUnsignedLong((uint8_t)White);
}

static PyObject* chesslib_create_chesscolor_black(PyObject* self)
{
    /* create uint32 python object and return it */
    return PyLong_FromUnsignedLong((uint8_t)Black);
}

/**************************************************************************
  Create an instance of the ChessPosition type given the position as string
  (e.g. 'A2' for row=1, column=0). The value of the ChessPosition is equal to
  an uint32 object within value range of 0-63 (both bounds included).
 **************************************************************************/
static PyObject* chesslib_create_chessposition(PyObject* self, PyObject* args)
{
    const char* pos_as_string;
    uint8_t row = 0, column = 0;

    /* read position string, quit if the parameter does not exist */
    if (!PyArg_ParseTuple(args, "s", &pos_as_string)) { return NULL; }

    /* make sure that the overloaded string is of the correct format */
    if (strlen(pos_as_string) != 2
        || (!isalpha(pos_as_string[0]) || toupper(pos_as_string[0]) - 'A' >= 8 || toupper(pos_as_string[0]) - 'A' < 0)
        || (!isdigit(pos_as_string[1]) || pos_as_string[1] - '1' >= 8)) { return NULL; }
    /* TODO: implement throwing an invalid argument exception instead */

    /* parse position from position string */
    row = pos_as_string[1] - '1';
    column = toupper(pos_as_string[0]) - 'A';

    /* create uint32 python object and return it */
    return PyLong_FromUnsignedLong(create_position(row, column));
}

/**************************************************************************
  Create an instance of the ChessPiece type given the color and piece type
  as a single character and the boolean was_moved as integer (0=FALSE, else TRUE).
  The value of the ChessPiece is equal to an uint32 object with the lowest 3 bits
  as piece type enum, the next higher bit as color and the highest bit as was_moved.
  For further details see the documentation of the ChessPiece type.
 **************************************************************************/
static PyObject* chesslib_create_chesspiece(PyObject* self, PyObject* args)
{
    const char *color_as_char, *type_as_char;
    ChessColor color;
    ChessPieceType type;
    int was_moved;

    /* read chess color and chess piece type string, quit if the parameter does not exist */
    /* read was moved boolean, quit if the parameter does not exist */
    if (!PyArg_ParseTuple(args, "ssi", &color_as_char, &type_as_char, &was_moved)) { return NULL; }

    /* parse the chess piece color */
    switch (toupper(*color_as_char))
    {
        case 'W': color = White; break;
        case 'B': color = Black; break;
        default: return NULL;    break;
    }

    /* parse the chess piece type */
    switch (toupper(*type_as_char))
    {
        case 'K': type = King;    break;
        case 'Q': type = Queen;   break;
        case 'R': type = Rook;    break;
        case 'B': type = Bishop;  break;
        case 'N': type = Knight;  break;
        case 'P': type = Peasant; break;
        default: return NULL;
    }

    /* create uint32 python object and return it */
    return PyLong_FromUnsignedLong(create_piece(type, color, was_moved));
}

/**************************************************************************
  Create an instance of the ChessPieceAtPos type given the ChessPiece
  and ChessPosition as integers. The value of the ChessPieceAtPos
  instance equals an uint32 object with lowest 5 bits as ChessPiece and
  the next higher 6 bits as ChessPosition.
  For further details see the documentation of the ChessPieceAtPos type.
 **************************************************************************/
static PyObject* chesslib_create_chesspieceatpos(PyObject* self, PyObject* args)
{
    ChessPiece piece;
    ChessPosition pos;

    /* read chess piece and chess position, quit if the parameters do not exist */
    if (!PyArg_ParseTuple(args, "ii", &piece, &pos)) { return NULL; }
    /* TODO: check if parsing requires 'O' option for parsing the python uint32 objects */

    /* make sure that the chess piece and chess position value are withing their numeric bounds */
    if (piece >= 32 || pos >= 64) { return NULL; }

    /* create uint32 python object and return it */
    return PyLong_FromUnsignedLong(create_pieceatpos(pos, piece));
}

/**************************************************************************
  Create an instance of the ChessBoard struct given a list of ChessPieceAtPos
  values defining where the pieces have to be put onto the chess board.
 **************************************************************************/
static PyObject* chesslib_create_chessboard(PyObject* self, PyObject* args)
{
    PyObject *pieces_list = NULL, *temp_obj, *iterator;
    ChessPieceAtPos pieces_at_pos[32];
    uint8_t offset = 0;
    ChessBoard board;

    /* parse args as object */
    if (!PyArg_ParseTuple(args, "O", &pieces_list)) { return NULL; }

    /* get an iterator of the list to parse */
    iterator = PyObject_GetIter(pieces_list);

    /* make sure that the iterator is valid */
    if (!iterator) { return NULL; }

    /* loop through the list using the iterator */
    while ((temp_obj = PyIter_Next(iterator)))
    {
        if (!PyLong_Check(temp_obj)) { return NULL; }
        pieces_at_pos[offset++] = (ChessPieceAtPos)PyLong_AsLong(temp_obj);
    }

    /* create the chess board */
    board = create_board_from_piecesatpos(pieces_at_pos, offset);
    return serialize_chessboard(board);
}

/**************************************************************************
  Create an instance of the ChessBoard struct with all chess pieces in
  start formation.
 **************************************************************************/
static PyObject* chesslib_create_chessboard_startformation(PyObject* self)
{
    /* create the chess board */
    ChessBoard board = START_FORMATION;
    return serialize_chessboard(board);
}

/**************************************************************************
  Create an instance of the ChessDraw type given following parameters:
    1) An instance of ChessBoard representing the position before the draw
    2) The old position of the moving chess piece
    3) The new position of the moving chess piece
    4) The type that a peasant promotes to in case of a peasant promotion (optional parameter)
  The ChessDraw value consists of an uint32 object that represent bitwise
  concatenation of several parameters defining the draw (lowest 25 bits).
  For further details see the documentation of the ChessDraw type.
 **************************************************************************/
static PyObject* chesslib_create_chessdraw(PyObject* self, PyObject* args)
{
    PyObject *bitboards, *old_pos_obj, *new_pos_obj, *prom_type_obj;
    ChessBoard board = BOARD_NULL;
    ChessPosition old_pos = 0, new_pos = 0;
    ChessPieceType peasant_promotion_type = Invalid;

    /* TODO: make sure that both overloads work correctly */
    if (!PyArg_ParseTuple(args, "Okkk", &bitboards, &old_pos_obj, &new_pos_obj, &prom_type_obj)
        || !PyArg_ParseTuple(args, "Okk", &bitboards, &old_pos_obj, &new_pos_obj)) { return NULL; }

    /* convert parsed python objects */
    if (!PyLong_Check(old_pos_obj) || !PyLong_Check(new_pos_obj) || !PyLong_Check(prom_type_obj)) { return NULL; }
    old_pos = (ChessPosition)PyLong_AsUnsignedLong(old_pos_obj);
    new_pos = (ChessPosition)PyLong_AsUnsignedLong(new_pos_obj);
    peasant_promotion_type = (ChessPieceType)PyLong_AsUnsignedLong(prom_type_obj);

    return PyLong_FromUnsignedLong(create_draw(board, old_pos, new_pos, peasant_promotion_type));
}

static PyObject* chesslib_create_chessdraw_null(PyObject* self)
{
    return PyLong_FromUnsignedLong(DRAW_NULL);
}


/* =================================================
            G E N E R A T E    D R A W S
   ================================================= */

/**************************************************************************
  Retrive a list of all possible chess draws for the current chess position
  given following arguments:
    1) An instance of ChessBoard representing the position before the draws
    2) The drawing side as ChessColor enum (White=0, Black=1)
    3) The last draw or DRAW_NULL on first draw (only relevant for en-passant rule, default: DRAW_NULL)
    4) A boolean indicating whether draw-into-check should be analyzed or not (default: FALSE)

 **************************************************************************/
static PyObject* chesslib_get_all_draws(PyObject* self, PyObject* args)
{
    PyObject *bitboards, *drawing_side_obj, *last_draw_obj;
    ChessDraw *out_draws, last_draw = DRAW_NULL;
    const size_t dims[1];
    ChessBoard board;
    ChessColor drawing_side;
    int analyze_draw_into_check;

    /* parse args as object */
    if (!PyArg_ParseTuple(args, "Okki", &bitboards, &drawing_side_obj, &last_draw_obj, &analyze_draw_into_check)) { return NULL; }
    /* TODO: add overloads without last_draw and/or analyze_draw_into_check */

    drawing_side = (ChessColor)PyLong_AsUnsignedLong(drawing_side_obj);
    last_draw = (ChessDraw)PyLong_AsUnsignedLong(last_draw_obj);
    board = deserialize_chessboard(bitboards);

    /* compute possible draws */
    get_all_draws(&out_draws, dims, board, drawing_side, last_draw, analyze_draw_into_check);

    /* serialize draws as numpy list */
    return PyArray_SimpleNewFromData(1, dims, NPY_UINT32, out_draws);
}

/* =================================================
           H E L P E R    F U N C T I O N S
   ================================================= */

static PyObject* serialize_chessboard(ChessBoard board)
{
    /* init a one-dimensional 64-bit integer numpy array with 13 elements */
    const npy_intp dims[1] = { 13 };
    return PyArray_SimpleNewFromData(1, dims, NPY_UINT64, board);
}

static ChessBoard deserialize_chessboard(PyObject* bitboards)
{
    PyObject* iterator, * temp_obj;
    size_t i;
    ChessBoard board = BOARD_NULL;

    /* get an iterator of the list to parse */
    iterator = PyObject_GetIter(bitboards);

    /* make sure that the iterator is valid */
    if (!iterator) { return BOARD_NULL; }

    /* loop through the list using the iterator */
    /*while ((temp_obj = PyIter_Next(iterator)))*/
    for (i = 0; i < 13; i++)
    {
        /* TODO: make sure that the PyLong_Check() makro works correctly! */
        if (!PyLong_Check(temp_obj)) { return BOARD_NULL; }
        board[i] = PyLong_AsUnsignedLongLong(temp_obj);
    }

    return board;
}
