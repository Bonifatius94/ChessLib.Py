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

#include "chesslibmodule.h"

/* =================================================
      H E L P E R    F U N C T I O N    S T U B S
   ================================================= */

static PyObject* serialize_as_bitboards(const Bitboard board[]);
static PyObject* serialize_as_pieces(const ChessPiece pieces[]);
static ChessBoard deserialize_as_bitboards(PyObject* bitboards_obj, int is_simple_board);
static SimpleChessBoard deserialize_as_pieces(PyObject* bitboards_obj, int is_simple_board);
static ChessDraw deserialize_chessdraw(const Bitboard board[], const ChessDraw draw);
static void compress_pieces_array(const ChessPiece pieces[], uint8_t* out_bytes);
static void uncompress_pieces_array(const uint8_t hash_bytes[], ChessPiece* out_pieces);
uint8_t get_bits_at(const uint8_t data_bytes[], size_t arr_size, int bit_index, int length);

/* =================================================
                 I N I T I A L I Z E
              P Y T H O N    M O D U L E
   ================================================= */

/* enforce Python 3 or higher */
#if PY_MAJOR_VERSION >= 3

#define PY_METHODS_SENTINEL {NULL, NULL, 0, NULL}

/* Define all functions exposed to python. */
static PyMethodDef chesslib_methods[] = {
    {"GenerateDraws", chesslib_get_all_draws, METH_VARARGS, "Generate all possible draws for the given position."},
    {"ApplyDraw", chesslib_apply_draw, METH_VARARGS, "Apply the given chess draw to the given chess board (result as new reference)."},
    {"ChessBoard", chesslib_create_chessboard, METH_VARARGS, "Create a new chess board."},
    {"ChessDraw", chesslib_create_chessdraw, METH_VARARGS, "Create a new chess draw."},
    {"ChessPiece", chesslib_create_chesspiece, METH_VARARGS, "Create a new chess piece."},
    {"ChessPosition", chesslib_create_chessposition, METH_VARARGS, "Create a new chess position."},
    {"ChessPieceAtPos", chesslib_create_chesspieceatpos, METH_VARARGS, "Create a new chess piece including its' position."},
    {"Board_ToHash", chesslib_board_to_hash, METH_VARARGS, "Compute the given chess board's hash as string."},
    {"Board_FromHash", chesslib_board_from_hash, METH_VARARGS, "Compute the given chess board's hash as string."},
    {"ChessBoard_StartFormation", chesslib_create_chessboard_startformation, METH_VARARGS, "Create a new chess board in start formation."},
    {"GameState", chesslib_get_game_state, METH_VARARGS, "Determine the game state for the given chess board and side."},
    {"VisualizeBoard", chesslib_visualize_board, METH_VARARGS, "Transform the chess board instance into a printable string."},
    {"VisualizeDraw", chesslib_visualize_draw, METH_VARARGS, "Transform the chess draw instance into a printable string."},
    /*{"ApplyDraw", chesslib_apply_draw, METH_VARARGS, "Apply the given chess draw to the given chess board (result as new reference)."},*/
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

    /* init numpy array tools */
    import_array();
    if (PyErr_Occurred()) { return NULL; }

    /* create module from definition */
    module = PyModule_Create(&chesslib_module);
    if (!module) { return NULL; }

    /* add integer constants 'White' and 'Black' for enum ChessColor */
    PyModule_AddIntConstant(module, "ChessColor_White", (int8_t)White);
    PyModule_AddIntConstant(module, "ChessColor_Black", (int8_t)Black);

    /* add integer constant for ChessDraw NULL value */
    PyModule_AddIntConstant(module, "ChessDraw_Null", (int32_t)DRAW_NULL);

    /* add integer constants for ChessGameState enum */
    PyModule_AddIntConstant(module, "GameState_None", (ChessGameState)None);
    PyModule_AddIntConstant(module, "GameState_Check", (ChessGameState)Check);
    PyModule_AddIntConstant(module, "GameState_Checkmate", (ChessGameState)Checkmate);
    PyModule_AddIntConstant(module, "GameState_Tie", (ChessGameState)Tie);
    
    /* add package version attribute */
    PyModule_AddStringConstant(module, "version", CHESSLIB_PACKAGE_VERSION);

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

    /* parse the chess piece's color and type */
    color = color_from_char(*color_as_char);
    type = piece_type_from_char(*type_as_char);

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
    ChessPiece piece = 0;
    ChessPosition pos = 0;

    /* read chess piece and chess position, quit if the parameters do not exist */
    if (!PyArg_ParseTuple(args, "bb", &piece, &pos)) { return NULL; }

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
    PyArrayObject* nd_pieces_at_pos;
    PyObject *pieces_list = NULL;
    ChessPieceAtPos* pieces_at_pos;
    uint8_t count = 0;
    ChessBoard board;
    int is_simple_board = 0;

    /* parse all args */
    if (!PyArg_ParseTuple(args, "O|i", &pieces_list, &is_simple_board)) { return NULL; }
    nd_pieces_at_pos = (PyArrayObject*)PyArray_FromObject(pieces_list, NPY_UINT16, 1, 32);
    count = (size_t)PyArray_Size((PyObject*)nd_pieces_at_pos);
    pieces_at_pos = (ChessPieceAtPos*)PyArray_DATA(nd_pieces_at_pos);

    /* create the chess board */
    board = create_board_from_piecesatpos(pieces_at_pos, count);
    return is_simple_board
        ? serialize_as_pieces(to_simple_board(board))
        : serialize_as_bitboards(board);
}

/**************************************************************************
  Create an instance of the ChessBoard struct with all chess pieces in
  start formation.
 **************************************************************************/
static PyObject* chesslib_create_chessboard_startformation(PyObject* self, PyObject* args)
{
    int is_simple_board = 0;

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
        0x0000FFFFFFFF0000uLL  /* was_moved mask */
    };

    /* parse all args */
    if (!PyArg_ParseTuple(args, "|i", &is_simple_board)) { return NULL; }

    return is_simple_board
        ? serialize_as_pieces(to_simple_board(start_formation))
        : serialize_as_bitboards(start_formation);
}

/**************************************************************************
  Create an instance of the ChessDraw type given following parameters:
    1) An instance of ChessBoard representing the position before the draw
    2) The old position of the moving chess piece
    3) The new position of the moving chess piece
    4) The type that a peasant promotes to in case of a peasant promotion (optional parameter)
    5) A boolean indicating whether the draw should be put as compact format (default: False)
  The ChessDraw value consists of an uint32 object that represent bitwise
  concatenation of several parameters defining the draw (lowest 25 bits).
  For further details see the documentation of the ChessDraw type.
 **************************************************************************/
static PyObject* chesslib_create_chessdraw(PyObject* self, PyObject* args)
{
    PyObject* chessboard;
    ChessBoard board = NULL;
    ChessPosition old_pos = 0, new_pos = 0;
    ChessPieceType prom_type = Invalid;
    int is_compact_format = 0;
    int is_simple_board = 0;
    ChessDraw draw;

    if (!PyArg_ParseTuple(args, "Okk|kii", &chessboard, &old_pos, &new_pos, 
        &prom_type, &is_compact_format, &is_simple_board)) { return NULL; }
    board = deserialize_as_bitboards(chessboard, is_simple_board);

    /* create the chess draw as unsigned 32-bit integer */
    draw = create_draw(board, old_pos, new_pos, prom_type);
    return PyLong_FromUnsignedLong(is_compact_format ? to_compact_draw(draw) : draw);
}

/* =================================================
            G E N E R A T E    D R A W S
   ================================================= */

/**************************************************************************
  Retrive a list of all possible chess draws for the current chess position
  given following arguments:
    1) An instance of ChessBoard representing the chess piece positions
    2) The drawing side as ChessColor enum (White=0, Black=1)
    3) The last draw or DRAW_NULL on first draw (only relevant for en-passant rule, default: DRAW_NULL)
    4) A boolean indicating whether draw-into-check should be analyzed or not (default: FALSE)
    5) A boolean indicating whether the draws should be returned as compact format (default: FALSE)
 **************************************************************************/
static PyObject* chesslib_get_all_draws(PyObject* self, PyObject* args)
{
    PyObject *chessboard, *serialized_draws;
    size_t dims[1];

    ChessDraw *out_draws, last_draw = DRAW_NULL;
    CompactChessDraw *comp_out_draws;
    ChessBoard board;
    ChessColor drawing_side;
    int analyze_draw_into_check = 0;
    int is_compact_format = 0;
    int is_simple_board = 0;
    int i = 0;

    /* parse input args */
    int is_valid = PyArg_ParseTuple(args, "Ok|iiii", &chessboard, &drawing_side, 
        &last_draw, &analyze_draw_into_check, &is_compact_format, &is_simple_board);

    /* make sure that any of the given arguments fit any of the patterns above, otherwise abort */
    if (!is_valid) { return NULL; }

    /* convert the numpy array into a chess bitboard instance */
    board = deserialize_as_bitboards(chessboard, is_simple_board);

    /* compute all possible draws for the given chess position */
    get_all_draws(&out_draws, dims, board, drawing_side, last_draw, analyze_draw_into_check);

    /* convert draws to compact format if required */
    if (is_compact_format)
    {
        comp_out_draws = (CompactChessDraw *)malloc((*dims) * sizeof(CompactChessDraw));
        for (i = 0; i < *dims; i++) { comp_out_draws[i] = to_compact_draw(out_draws[i]); }
        free(out_draws);
    }

    /* serialize all draws as a numpy array */
    serialized_draws = is_compact_format
        ? PyArray_SimpleNewFromData(1, (npy_intp*)dims, NPY_UINT16, comp_out_draws)
        : PyArray_SimpleNewFromData(1, (npy_intp*)dims, NPY_UINT32, out_draws);

    return serialized_draws;
}

/* =================================================
                 A P P L Y    D R A W
   ================================================= */

static PyObject* chesslib_apply_draw(PyObject* self, PyObject* args)
{
    PyObject *chessboard;
    ChessDraw draw_to_apply;
    ChessBoard old_board, new_board;
    int is_simple_board = 0;

    /* parse input args */
    if (!PyArg_ParseTuple(args, "Oi|i", &chessboard, &draw_to_apply, &is_simple_board)) { return NULL; }
    old_board = deserialize_as_bitboards(chessboard, is_simple_board);
    draw_to_apply = deserialize_chessdraw(old_board, draw_to_apply);

    /* apply the chess draw to a new ChessBoard instance */
    new_board = apply_draw(old_board, draw_to_apply);

    /* serialize the new Chessboard as numpy list */
    return serialize_as_bitboards(new_board);
}

/* =================================================
                B O A R D    H A S H
   ================================================= */

/**************************************************************************
  Retrieve a 40-byte representation of the given ChessBoard instance.
 **************************************************************************/
static PyObject* chesslib_board_to_hash(PyObject* self, PyObject* args)
{
    PyObject *chessboard;
    uint8_t *bytes;
    size_t dims[1] = { 40 };
    int is_simple_board = 0;
    SimpleChessBoard temp_pieces;

    /* parse bitboards as ChessBoard struct */
    if (!PyArg_ParseTuple(args, "O|i", &chessboard, &is_simple_board)) { return NULL; }
    temp_pieces = deserialize_as_pieces(chessboard, is_simple_board);

    /* allocate 40-bytes array */
    bytes = (uint8_t*)calloc(40, sizeof(uint8_t));
    if (bytes == NULL) { return NULL; }

    /* compress the pieces cache to 40 bytes by removing the unused leading 3 bits of each ChessPiece value */
    compress_pieces_array(temp_pieces, bytes);

    /* convert parsed bytes to Python bytearray struct */
    return PyArray_SimpleNewFromData(1, (npy_intp*)dims, NPY_UINT8, bytes);
}

/**************************************************************************
  Retrieve a ChessBoard instance of the given 40-byte hash representation.
 **************************************************************************/
static PyObject* chesslib_board_from_hash(PyObject* self, PyObject* args)
{
    /* TODO: export this function to chessboard.c */

    Bitboard *board;
    PyObject *hash_orig; PyArrayObject* hash;
    uint8_t *compressed_bytes;
    ChessPiece temp_pieces[64] = { 0 };

    /* parse bitboards as ChessBoard struct */
    if (!PyArg_ParseTuple(args, "O", &hash_orig)) { return NULL; }
    hash = (PyArrayObject*)PyArray_FromObject(hash_orig, NPY_UINT8, 1, 40);
    compressed_bytes = (uint8_t*)PyArray_DATA(hash);

    /* uncompress the pieces cache from 40 bytes by adding the unused leading 3 bits of each ChessPiece value */
    uncompress_pieces_array(compressed_bytes, temp_pieces);

    /* convert the single chess pieces (aka SimpleChessBoard) to the bitboard representation */
    board = from_simple_board(temp_pieces);

    /* convert parsed bytes to Python bytearray struct */
    return serialize_as_bitboards(board);
}

/* =================================================
                 G A M E    S T A T E
   ================================================= */

static PyObject* chesslib_get_game_state(PyObject* self, PyObject* args)
{
    PyObject* chessboard;
    ChessBoard board;
    ChessDraw last_draw = DRAW_NULL;
    ChessGameState state;
    int is_simple_board = 0;

    /* parse bitboards as ChessBoard struct */
    if (!PyArg_ParseTuple(args, "Oi|i", &chessboard, &last_draw, &is_simple_board)) { return NULL; }
    board = deserialize_as_bitboards(chessboard, is_simple_board);

    /* determine the game state */
    state = get_game_state(board, last_draw);
    return PyLong_FromUnsignedLong(state);
}

/* =================================================
                  V I S U A L I Z E
   ================================================= */

static PyObject* chesslib_visualize_board(PyObject* self, PyObject* args)
{
    PyObject* bitboards;
    ChessBoard board;
    char out[18 * 46], buf[6];
    char separator[] = "   -----------------------------------------\n";
    uint8_t row, column;
    ChessPosition pos;
    ChessPiece piece;
    int is_simple_board = 0;

    /* parse bitboards as ChessBoard struct */
    if (!PyArg_ParseTuple(args, "O|i", &bitboards, &is_simple_board)) { return NULL; }
    board = deserialize_as_bitboards(bitboards, is_simple_board);

    /* determine the chess board's textual representation */
    strcpy(out, separator);

    for (row = 7; row < 8; row--)
    {
        /* write row description */
        sprintf(buf, " %i |", (row + 1));
        strcat(out, buf);

        for (column = 0; column < 8; column++)
        {
            /* get the chess piece at the iteration's position */
            pos = create_position(row, column);
            piece = get_piece_at(board, pos);

            /* write the chess piece to the output */
            sprintf(buf, " %c%c |", 
                is_captured_at(board, pos) ? color_to_char(get_piece_color(piece)) : ' ',
                is_captured_at(board, pos) ? piece_type_to_char(get_piece_type(piece)) : ' ');
            strcat(out, buf);
        }

        /* write separator line */
        strcat(out, "\n");
        strcat(out, separator);

        if (row == -1) { return Py_BuildValue("s", out); }
    }

    /* write column descriptions */
    strcat(out, "     A    B    C    D    E    F    G    H");

    return Py_BuildValue("s", out);
}

static PyObject* chesslib_visualize_draw(PyObject* self, PyObject* args)
{
    ChessDraw draw;
    char out[100], buf[100], *old_pos, *new_pos;
    int is_left_side = 0;

    /* parse bitboards as ChessBoard struct */
    if (!PyArg_ParseTuple(args, "i", &draw)) { return NULL; }
    /* TODO: a chessboard is required for context information when only providing compact draws
             this function only works for non-compact draws*/

    old_pos = position_to_string(get_old_position(draw));
    new_pos = position_to_string(get_new_position(draw));

    /* determine the chess draw's base representation */
    sprintf(buf, "%s %s %s-%s",
        color_to_string(get_drawing_side(draw)),
        piece_type_to_string(get_drawing_piece_type(draw)),
        old_pos,
        new_pos);
    strcpy(out, buf);

    free(old_pos); free(new_pos);

    /* append additional information for special draws */
    switch (get_draw_type(draw))
    {
        case Rochade:
            is_left_side = (get_column(get_new_position(draw)) == 2 && get_drawing_side(draw) == White) 
                || (get_column(get_new_position(draw)) == 6 && get_drawing_side(draw) == Black);
            sprintf(buf, "%s-side rochade", (is_left_side ? "left" : "right"));
            strcat(out, buf);
            break;
        case EnPassant:
            strcat(out, " (en-passant)");
            break;
        case PeasantPromotion:
            sprintf(buf, " (%s)", piece_type_to_string(get_peasant_promotion_piece_type(draw)));
            strcat(out, buf);
            break;
        case Standard: break;
    }

    return Py_BuildValue("s", out, strlen(out));
}

/* =================================================
           H E L P E R    F U N C T I O N S
   ================================================= */

static PyObject* serialize_as_pieces(const ChessPiece pieces[])
{
    /* init a one-dimensional 8-bit integer numpy array with 64 elements */
    npy_intp dims[1] = { 64 };

    uint8_t i;
    SimpleChessBoard data_copy = (SimpleChessBoard)malloc(64 * sizeof(ChessPiece));
    if (data_copy == NULL) { return NULL; }
    for (i = 0; i < 64; i++) { data_copy[i] = pieces[i]; }

    return PyArray_SimpleNewFromData(1, (npy_intp*)dims, NPY_UINT8, data_copy);
}

static PyObject* serialize_as_bitboards(const Bitboard board[])
{
    /* init a one-dimensional 64-bit integer numpy array with 13 elements */
    npy_intp dims[1] = { 13 };

    uint8_t i;
    uint64_t *data_copy = (uint64_t*)malloc(13 * sizeof(uint64_t));
    if (data_copy == NULL) { return NULL; }
    for (i = 0; i < 13; i++) { data_copy[i] = board[i]; }

    return PyArray_SimpleNewFromData(1, (npy_intp*)dims, NPY_UINT64, data_copy);
}

static SimpleChessBoard deserialize_as_pieces(PyObject* bitboards_obj, int is_simple_board)
{
    SimpleChessBoard out_board = NULL;
    PyArrayObject *bitboards, *pieces;

    /* TODO: implement automatic detection of simple/efficient board formats
             use the functions PyArray_NDIM and PyArray_DTYPE
     */

    /* check if the given board can be interpreted as simple format */
    /*if (PyArray_NDIM(bitboards_obj) == 64 && PyArray_DTYPE(bitboards_obj) == NPY_UINT8)*/
    if (is_simple_board)
    {
        /* parse simple chess board as ndarray of 64 raw bytes */
        pieces = (PyArrayObject*)PyArray_FromObject(bitboards_obj, NPY_UINT8, 1, 64);

        /* convert the simple chess board into the bitboard representation for efficient operations */
        out_board = (SimpleChessBoard)PyArray_DATA(pieces);
    }
    /* check if the given board can be interpreted as bitboards format */
    /*else if (PyArray_NDIM(bitboards_obj) == 13 && PyArray_DTYPE(bitboards_obj) == NPY_UINT64)*/
    else
    {
        /* parse bitboards as 1-dimensional ndarray of type uint64 and size 13 */
        bitboards = (PyArrayObject*)PyArray_FromObject(bitboards_obj, NPY_UINT64, 1, 13);
        out_board = to_simple_board((Bitboard*)PyArray_DATA(bitboards));
    }

    return out_board;
}

static ChessBoard deserialize_as_bitboards(PyObject* bitboards_obj, int is_simple_board)
{
    ChessBoard out_board = NULL;
    PyArrayObject *bitboards, *pieces;

    /* TODO: implement automatic detection of simple/efficient board formats
             use the functions PyArray_NDIM and PyArray_DTYPE
     */

    /* check if the given board can be interpreted as simple format */
    /*if (PyArray_NDIM(bitboards_obj) == 64 && PyArray_DTYPE(bitboards_obj) == NPY_UINT8)*/
    if (is_simple_board)
    {
        /* parse simple chess board as ndarray of 64 raw bytes */
        pieces = (PyArrayObject*)PyArray_FromObject(bitboards_obj, NPY_UINT8, 1, 64);

        /* convert the simple chess board into the bitboard representation for efficient operations */
        out_board = from_simple_board((SimpleChessBoard)PyArray_DATA(pieces));
    }
    /* check if the given board can be interpreted as bitboards format */
    /*else if (PyArray_NDIM(bitboards_obj) == 13 && PyArray_DTYPE(bitboards_obj) == NPY_UINT64)*/
    else
    {
        /* parse bitboards as 1-dimensional ndarray of type uint64 and size 13 */
        bitboards = (PyArrayObject*)PyArray_FromObject(bitboards_obj, NPY_UINT64, 1, 13);
        out_board = (Bitboard*)PyArray_DATA(bitboards);
    }

    return out_board;
}

static ChessDraw deserialize_chessdraw(const Bitboard board[], const ChessDraw draw)
{
    /* if none of the leading 10 bits is set, the given draw has to be
       of the compact draw foramt -> append the missing properties. */
    return (draw < 0x800) ? from_compact_draw(board, (CompactChessDraw)draw) : draw;
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

static void uncompress_pieces_array(const uint8_t hash_bytes[], ChessPiece* out_pieces)
{
    ChessPosition pos;
    uint8_t piece_bits;

    /* loop through all positions */
    for (pos = 0; pos < 64; pos++)
    {
        piece_bits = get_bits_at(hash_bytes, 40, pos * 5, 5) >> 3;
        out_pieces[pos] = piece_bits;
    }
}

uint8_t get_bits_at(const uint8_t data_bytes[], size_t arr_size, int bit_index, int length)
{
    /* load data bytes into cache */
    uint8_t upper = data_bytes[bit_index / 8];
    uint8_t lower = (bit_index / 8 + 1 < arr_size) ? data_bytes[bit_index / 8 + 1] : (uint8_t)0x00;
    int bitOffset = bit_index % 8;

    /* cut the bits from the upper byte */
    uint8_t upperDataMask = (uint8_t)((1 << (8 - bitOffset)) - 1);
    int lastIndexOfByte = bitOffset + length - 1;
    if (lastIndexOfByte < 7) { upperDataMask = (uint8_t)((upperDataMask >> (7 - lastIndexOfByte)) << (7 - lastIndexOfByte)); }
    uint8_t upperData = (uint8_t)((upper & upperDataMask) << (bitOffset));

    /* cut bits from the lower byte (if needed, otherwise set all bits 0) */
    uint8_t lowerDataMask = (uint8_t)(0xFF << (16 - bitOffset - length));
    uint8_t lowerData = (uint8_t)((lower & lowerDataMask) >> (8 - bitOffset));

    /* put the data bytes together (with bitwise OR) */
    uint8_t data = (uint8_t)(upperData | lowerData);
    return data;
}
