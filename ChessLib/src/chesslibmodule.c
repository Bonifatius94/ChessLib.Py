#include "chesslibmodule.h"

/* =================================================
      H E L P E R    F U N C T I O N    S T U B S
   ================================================= */

static PyObject* serialize_chessboard(ChessBoard board);
static ChessBoard deserialize_chessboard(PyObject* board);
static void compress_pieces_array(const ChessPiece pieces[], uint8_t* out_bytes);

/*static PyObject* cos_func_np(PyObject* self, PyObject* args);*/

/* =================================================
                 I N I T I A L I Z E
              P Y T H O N    M O D U L E
   ================================================= */

/* enforce Python 3 or higher */
#if PY_MAJOR_VERSION >= 3

/* info: module extension code taken from following tutorial: https://realpython.com/build-python-c-extension-module/ */

#define PY_METHODS_SENTINEL {NULL, NULL, 0, NULL}

/* Define all functions exposed to python. */
static PyMethodDef chesslib_methods[] = {
    {"GenerateDraws", chesslib_get_all_draws, METH_VARARGS, "Generate all possible draws for the given position."},
    {"ChessBoard", chesslib_create_chessboard, METH_VARARGS, "Create a new chess board."},
    {"ChessDraw", chesslib_create_chessdraw, METH_VARARGS, "Create a new chess draw."},
    {"ChessPiece", chesslib_create_chesspiece, METH_VARARGS, "Create a new chess piece."},
    {"ChessPosition", chesslib_create_chessposition, METH_VARARGS, "Create a new chess position."},
    {"ChessPieceAtPos", chesslib_create_chesspieceatpos, METH_VARARGS, "Create a new chess piece including its' position."},
    {"Board_Hash", chesslib_board_to_hash, METH_VARARGS, "Compute the given chess board's hash as string."},
    {"ChessBoard_StartFormation", chesslib_create_chessboard_startformation, METH_NOARGS, "Create a new chess board in start formation."},
    /*{"cos_custom", cos_func_np, METH_VARARGS, "A custom implementation of math. cos(x) function."},*/
    PY_METHODS_SENTINEL,
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
    PyObject* module;

    /* initialize the python environment */
    Py_Initialize();

    /* create module from definition */
    module = PyModule_Create(&chesslib_module);
    if (!module) { return NULL; }

    /* add integer constants 'White' and 'Black' for enum ChessColor */
    PyModule_AddIntConstant(module, "ChessColor_White", (int8_t)White);
    PyModule_AddIntConstant(module, "ChessColor_Black", (int8_t)Black);

    /* add integer constant for enum ChessDraw NULL */
    PyModule_AddIntConstant(module, "ChessDraw_Null", (int32_t)DRAW_NULL);

    /* init numpy array tools */
    import_array();
    if (PyErr_Occurred()) { return NULL; }

    return module;
}

#endif

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
    /* TODO: replace with  */

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
    if (*(pos_as_string + 2) != '\0'
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
    if (!PyArg_ParseTuple(args, "kk", &piece, &pos)) { return NULL; }
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
    const Bitboard start_formation[] = {
        0x0000000000000010uLL, /* white king     */
        0x0000000000000008uLL, /* white queen(s) */
        0x0000000000000081uLL, /* white rooks    */
        0x0000000000000024uLL, /* white bishops  */
        0x0000000000000042uLL, /* white knights  */
        0x000000000000FF00uLL, /* white pawns    */
        0x1000000000000000uLL, /* black king     */
        0x0800000000000000uLL, /* black queen(s) */
        0x8100000000000000uLL, /* black rooks    */
        0x2400000000000000uLL, /* black bishops  */
        0x4200000000000000uLL, /* black knights  */
        0x00FF000000000000uLL, /* black pawns    */
        0xFFFF00000000FFFFuLL  /* was_moved mask */
    };

    return serialize_chessboard(start_formation);
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
    ChessBoard board = NULL;
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
    PyArrayObject* bitboards;
    PyObject *drawing_side_obj, *last_draw_obj;
    const size_t dims[1];

    ChessDraw *out_draws, last_draw = DRAW_NULL;
    ChessBoard board;
    ChessColor drawing_side;
    int analyze_draw_into_check;

    /* parse args as object */
    if (!PyArg_ParseTuple(args, "O!kki", &PyArray_Type, &bitboards, &drawing_side_obj, &last_draw_obj, &analyze_draw_into_check)) { return NULL; }
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
                B O A R D    H A S H
   ================================================= */

/**************************************************************************
  Retrieve a 40-byte representation of the given ChessBoard instance.
 **************************************************************************/
static PyObject* chesslib_board_to_hash(PyObject* self, PyObject* args)
{
    ChessBoard board;
    PyObject *bitboards;
    uint8_t *bytes, i;
    size_t dims[1] = { 40 };

    ChessColor color; ChessPieceType piece_type; int was_moved;
    Bitboard temp_bitboard; ChessPosition temp_pos;
    ChessPiece temp_piece, *temp_pieces;

    /* parse bitboards as ChessBoard struct */
    if (!PyArg_ParseTuple(args, "O", &bitboards)) { return NULL; }
    board = deserialize_chessboard(bitboards);

    /* allocate 40-bytes array */
    bytes = (uint8_t*)calloc(40, sizeof(uint8_t));
    if (bytes == NULL) { return NULL; }

    /* allocate 64-bytes pieces cache (index = position on the chess board) */
    temp_pieces = (ChessPiece*)calloc(64, sizeof(uint8_t));
    if (temp_pieces == NULL) { return NULL; free(bytes); }

    /* determine the pieces at each position and write them to the result bytes */
    for (i = 0; i < 12; i++)
    {
        color = (ChessColor)(i / 6);
        piece_type = (ChessPieceType)(i % 6);
        temp_bitboard = board[i];

        /* until all set bits were evaluated */
        while (temp_bitboard)
        {
            /* get the index of the highest bit set on the bitboard */
            temp_pos = get_board_position(temp_bitboard);

            /* determine was_moved for the given chess piece and finally create the ChessPiece struct */
            was_moved = (int)(was_piece_moved(board, temp_pos) >> temp_pos);
            temp_piece = create_piece(piece_type, color, was_moved);

            /* write the 5 bits of the ChessPiece struct to the pieces cache at the given position */
            temp_pieces[temp_pos] = temp_piece;

            /* remove bit from bitboard to make the while loop terminate eventually */
            temp_bitboard ^= 0x1uLL << temp_pos;
        }
    }

    /* compress the pieces cache to 40 bytes by removing the unused leading 3 bits of each ChessPiece value */
    compress_pieces_array(temp_pieces, bytes);
    free(temp_pieces);

    /* convert parsed bytes to Python bytearray struct */
    return PyArray_SimpleNewFromData(1, dims, NPY_UINT8, bytes);
}

/* =================================================
           H E L P E R    F U N C T I O N S
   ================================================= */

static PyObject* serialize_chessboard(const ChessBoard board)
{
    /* init a one-dimensional 64-bit integer numpy array with 13 elements */
    npy_intp dims[1] = { 13 };

    uint8_t i;
    uint64_t *data_copy = (uint64_t*)malloc(13 * sizeof(uint64_t));
    if (data_copy == NULL) { return NULL; }
    for (i = 0; i < 13; i++) { data_copy[i] = board[i]; }

    return PyArray_SimpleNewFromData(1, dims, NPY_UINT64, data_copy);
}

static ChessBoard deserialize_chessboard(PyObject* bitboards)
{
    PyObject *iterator, *temp_obj;
    size_t i;
    ChessBoard board;

    board = (ChessBoard)malloc(13 * sizeof(Bitboard));
    if(!board) { return NULL; }

    /* get an iterator of the list to parse (and make sure that the iterator is valid) */
    iterator = PyObject_GetIter(bitboards);
    if (!iterator) { return NULL; }

    /* loop through the list using the iterator */
    for (i = 0; i < 13; i++)
    {
        temp_obj = PyIter_Next(iterator);
        if (!PyLong_Check(temp_obj)) { return NULL; }
        board[i] = PyLong_AsUnsignedLongLong(temp_obj);
    }

    return board;
}

static void compress_pieces_array(const ChessPiece pieces[], uint8_t* out_bytes)
{
    ChessPosition pos;
    const uint8_t mask = 0xF8u;
    uint8_t offset, index, piece_bits;

    /* loop through all positions */
    for (pos = 0; pos < 64; pos++)
    {
        /* get chess piece from array by position */
        piece_bits = pieces[pos] << 3;

        /* determine the output byte's index and bit offset */
        index = ((int)pos * 5) / 8;
        offset = ((int)pos * 5) % 8;

        /* write leading bits to byte at the piece's position */
        out_bytes[index] |= (piece_bits & mask) >> offset;

        /* write overlapping bits to the next byte (only if needed) */
        if (offset > 3) { out_bytes[index + 1] |= (uint8_t)((piece_bits & mask) << (8 - offset)); }
    }
}
