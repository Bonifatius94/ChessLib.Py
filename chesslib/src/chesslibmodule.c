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
         C H E S S L I B   F U N C T I O N S
   ================================================= */

static PyObject* chesslib_create_chessposition(PyObject* self, PyObject* args);
static PyObject* chesslib_create_chesspiece(PyObject* self, PyObject* args);
static PyObject* chesslib_create_chesspieceatpos(PyObject* self, PyObject* args);
static PyObject* chesslib_create_chessboard(PyObject* self, PyObject* args);
static PyObject* chesslib_create_chessboard_startformation(PyObject* self, PyObject* args);
static PyObject* chesslib_create_chessdraw(PyObject* self, PyObject* args);
static PyObject* chesslib_get_all_draws(PyObject* self, PyObject* args);
static PyObject* chesslib_board_to_hash(PyObject* self, PyObject* args);
static PyObject* chesslib_board_from_hash(PyObject* self, PyObject* args);
static PyObject* chesslib_apply_draw(PyObject* self, PyObject* args);
static PyObject* chesslib_get_game_state(PyObject* self, PyObject* args);
static PyObject* chesslib_visualize_board(PyObject* self, PyObject* args);
static PyObject* chesslib_visualize_draw(PyObject* self, PyObject* args);

/* =================================================
      H E L P E R    F U N C T I O N    S T U B S
   ================================================= */

static PyObject* serialize_as_bitboards(const Bitboard board[]);
static PyObject* serialize_as_pieces(const ChessPiece pieces[]);
static Bitboard* deserialize_as_bitboards(PyObject* bitboards_obj, int is_simple_board);
static ChessPiece* deserialize_as_pieces(PyObject* bitboards_obj, int is_simple_board);
static ChessDraw deserialize_chessdraw(const Bitboard board[], const ChessDraw draw);

/* =================================================
             D O C S T R I N G S   F O R
              P Y T H O N    M O D U L E
   ================================================= */

/*if (!PyArg_ParseTuple(args, "s", &pos_as_string)) { return NULL; }*/
const char ChessPosition_Docstring[] =
"ChessPosition(pos_as_string: str) -> int\n\
\n\
Transform the literal chess position representation (e.g. 'A5' or 'H8') to a bitboard index.\n\
\n\
Args:\n\
    pos_as_string: The literal chess position representation (e.g. 'A5' or 'H8') as string\n\
\n\
Returns:\n\
    the bitboard index representing the given field on the chess board as integer/byte";

/* if (!PyArg_ParseTuple(args, "ssi", &color_as_char, &type_as_char, &was_moved)) { return NULL; } */
const char ChessPiece_Docstring[] =
"ChessPiece(color_as_char: str, type_as_char: str, was_moved: bool) -> int\n\
\n\
Create a chess piece given its color, type and whether it was already moved.\n\
\n\
Args:\n\
    color_as_char: The piece color (i.e. white='W' or black='B') as string, case-insensitive\n\
    type_as_char: The piece type (i.e. King='K', Queen='Q', Rook='R', Bishop='B', Knight='N', Pawn='P', Invalid/Null='I') as string, case-insensitive\n\
    was_moved: Indicates whether the piece was already moved\n\
\n\
Returns:\n\
    the chess piece as integer/byte";

/* if (!PyArg_ParseTuple(args, "bb", &piece, &pos)) { return NULL; } */
const char ChessPieceAtPos_Docstring[] =
"ChessPieceAtPos(piece: byte, pos: byte) -> int\n\
\n\
Create a chess piece given its color, type and whether it was already moved.\n\
\n\
Args:\n\
    piece: The chess piece encoded as a single byte (e.g. using ChessPiece() function)\n\
    pos: The chess position encoded as a single byte (e.g. using ChessPosition() function)\n\
\n\
Returns:\n\
    the piece@pos attaching positional information to the given chess pice as 16-bit integer";

/* if (!PyArg_ParseTuple(args, "O|i", &pieces_list, &is_simple_board)) { return NULL; } */
const char ChessBoard_Docstring[] =
"ChessBoard(pieces_list: byte, is_simple_board: byte=False) -> numpy.ndarray\n\
\n\
Create a chess board given a list of chess pieces (including position annotations, i.e. piece@pos type).\n\
\n\
Args:\n\
    pieces_list: The list containing piece@pos entries, defining the chess board to be created. \n\
    is_simple_board: Indicates whether the resulting chess board should be of the simple board format, defaults to False\n\
\n\
Returns:\n\
    a chess board having all listed pieces at the given positions, encoded as numpy array";

/* if (!PyArg_ParseTuple(args, "|i", &is_simple_board)) { return NULL; } */
const char ChessBoard_StartFormation_Docstring[] =
"ChessBoard_StartFormation(is_simple_board: byte=False) -> numpy.ndarray\n\
\n\
Create a chess board with all pieces in start formation.\n\
\n\
Args:\n\
    is_simple_board: Indicates whether the resulting chess board should be of the simple board format, defaults to False\n\
\n\
Returns:\n\
    a chess board having all pieces in start formation, encoded as numpy array";

/* if (!PyArg_ParseTuple(args, "Okk|kii", &chessboard, &old_pos, &new_pos, 
        &prom_type, &is_compact_format, &is_simple_board)) { return NULL; } */
const char ChessDraw_Docstring[] =
"ChessDraw(chessboard: numpy.ndarray,\n\
    old_pos: int,\n\
    new_pos: int,\n\
    prom_type: int=0,\n\
    is_compact_format: bool=False,\n\
    is_simple_board: bool=False) -> int\n\
\n\
Create a chess draw given a chess board and the old/new position of the piece to be moved.\n\
\n\
Args:\n\
    chessboard: The chess board representing the current game situation\n\
    old_pos: The old position of the chess piece to be moved\n\
    new_pos: The new position of the chess piece to be moved\n\
    prom_type: The prom. type, in case a pawn hits the last level, defaults to Invalid/Null (see chess piece types)\n\
    is_compact_format: Indicates whether the draws should be returned as compact draw format, defaults to False\n\
    is_simple_board: Indicates whether the given chess board is of the simple board format, defaults to False\n\
\n\
Returns:\n\
    a chess draw representing the given draw features, encoded as 32-bit integer (only lowest 16 bits used for the compact draw format)";

/*int is_valid = PyArg_ParseTuple(args, "Ok|iiii", &chessboard, &drawing_side, 
        &last_draw, &analyze_draw_into_check, &is_compact_format, &is_simple_board);*/
const char GenerateDraws_Docstring[] =
"GenerateDraws(chessboard: numpy.ndarray,\n\
    drawing_side: int,\n\
    last_draw: int=chesslib.ChessDraw_Null,\n\
    analyze_draw_into_check: bool=True,\n\
    is_compact_format: bool=False,\n\
    is_simple_board: bool=False) -> numpy.ndarray\n\
\n\
Generate all possible draws for the given chess board and drawing side\n\
\n\
Args:\n\
    chessboard: The chess board representing the current game situation\n\
    drawing_side: The chess player that is supposed to draw where white=0 and black=1\n\
    last_draw: The most recent draw made by the opponent - which is important to get the en-passant rule right, defaults to ChessDraw_Null\n\
    analyze_draw_into_check: Indicates whether draws-into-check should be properly filtered from the output, defaults to True\n\
    is_compact_format: Indicates whether the draws should be returned as compact draw format, defaults to False\n\
    is_simple_board: Indicates whether the given chess board is of the simple board format, defaults to False\n\
\n\
Returns:\n\
    a set of all possible draws for the given game situation where each draw is represented by a 32-bit integer, encoded as a numpy array";

/* if (!PyArg_ParseTuple(args, "Oi|i", &chessboard, &draw_to_apply, &is_simple_board)) { return NULL; } */
const char ApplyDraw_Docstring[] =
"ApplyDraw(chessboard: numpy.ndarray, draw_to_apply: int, is_simple_board: bool=False) -> numpy.ndarray\n\
\n\
Apply the draw to the given chess board and return the resulting chess board as a new numpy array instance (immutable).\n\
This function can also revert draws by calling it with the chess board that resulted from applying the draw in the first place.\n\
The revertibility feature is strongly related to the underlying implementation using only bitwise XOR operations, so flipping the bits\n\
twice will result in the original state before applying the draw for the first time.\n\
\n\
Args:\n\
    chessboard: The chess board representing the game situation before applying the draw\n\
    draw_to_apply: The draw to be applied.\n\
    is_simple_board: Indicates whether the given chess board is of the simple board format, defaults to False\n\
\n\
Returns:\n\
    a new chess board instance having the given draw applied, encoded as numpy array";

/* if (!PyArg_ParseTuple(args, "Oi|i", &chessboard, &last_draw, &is_simple_board)) { return NULL; } */
const char GameState_Docstring[] =
"GameState(chessboard: numpy.ndarray, last_draw: int, is_simple_board: bool=False) -> int\n\
\n\
Determine the current game state given the chess board and the last draw made.\n\
Possible outcomes are: None='N', Check='C', Checkmate='M', Tie='T' (as single ASCII char values).\n\
\n\
Args:\n\
    chessboard: The chess board representing the game situation to be evaluated\n\
    last_draw: The draw that was applied last\n\
    is_simple_board: Indicates whether the given chess board is of the simple board format, defaults to False\n\
\n\
Returns:\n\
    the game state related to the given game situation, encoded as integer/ASCII byte";

/* if (!PyArg_ParseTuple(args, "O|i", &chessboard, &is_simple_board)) { return NULL; } */
const char Board_ToHash_Docstring[] =
"Board_ToHash(chessboard: numpy.ndarray, is_simple_board: bool=False) -> int\n\
\n\
Convert the given chess board to a 40-byte hash.\n\
\n\
Args:\n\
    chessboard: The chess board representing the game situation to be exported\n\
    is_simple_board: Indicates whether the given chess board is of the simple board format, defaults to False\n\
\n\
Returns:\n\
    the game state related to the given game situation, encoded as integer/ASCII byte";

/* if (!PyArg_ParseTuple(args, "O|i", &hash_orig, &is_simple_board)) { return NULL; } */
const char Board_FromHash_Docstring[] =
"Board_FromHash(hash_orig: numpy.ndarray, is_simple_board: bool=False) -> int\n\
\n\
Convert the given 40-byte hash to a chess board.\n\
\n\
Args:\n\
    hash_orig: The 40-byte hash representation to be imported\n\
    is_simple_board: Indicates whether the resulting chess board should be of the simple board format, defaults to False\n\
\n\
Returns:\n\
    the game state related to the given game situation, encoded as integer/ASCII byte";

/* if (!PyArg_ParseTuple(args, "O|i", &bitboards, &is_simple_board)) { return NULL; } */
const char VisualizeBoard_Docstring[] =
"VisualizeBoard(chessboard: numpy.ndarray, is_simple_board: bool=False) -> str\n\
\n\
Visualize the given chess board as a human-readable textual representation.\n\
\n\
Args:\n\
    chessboard: The chess board representing the game situation to be visualized\n\
    is_simple_board: Indicates whether the given chess board is of the simple board format, defaults to False\n\
\n\
Returns:\n\
    the chess board's human-readable textual representation as ASCII string";

/* if (!PyArg_ParseTuple(args, "i", &draw)) { return NULL; } */
const char VisualizeDraw_Docstring[] =
"VisualizeDraw(draw: numpy.ndarray) -> str\n\
\n\
Visualize the given chess draw as a human-readable textual representation.\n\
\n\
Args:\n\
    draw: The chess draw to be visualized (needs to be of non-compact format)\n\
\n\
Returns:\n\
    the chess draw's human-readable textual representation as ASCII string";

/* =================================================
                 I N I T I A L I Z E
              P Y T H O N    M O D U L E
   ================================================= */

/* enforce Python 3 or higher */
#if PY_MAJOR_VERSION >= 3

#define PY_METHODS_SENTINEL {NULL, NULL, 0, NULL}

/* define all functions exposed to python */
static PyMethodDef chesslib_methods[] = {

    /* data types and structures */
    {"ChessPosition", chesslib_create_chessposition, METH_VARARGS, ChessPosition_Docstring},
    {"ChessPiece", chesslib_create_chesspiece, METH_VARARGS, ChessPiece_Docstring},
    {"ChessPieceAtPos", chesslib_create_chesspieceatpos, METH_VARARGS, ChessPieceAtPos_Docstring},
    {"ChessBoard", chesslib_create_chessboard, METH_VARARGS, ChessBoard_Docstring},
    {"ChessBoard_StartFormation", chesslib_create_chessboard_startformation, METH_VARARGS, ChessBoard_StartFormation_Docstring},
    {"ChessDraw", chesslib_create_chessdraw, METH_VARARGS, ChessDraw_Docstring},

    /* core chess logic for gameplay */
    {"GenerateDraws", chesslib_get_all_draws, METH_VARARGS, GenerateDraws_Docstring},
    {"ApplyDraw", chesslib_apply_draw, METH_VARARGS, ApplyDraw_Docstring},
    {"GameState", chesslib_get_game_state, METH_VARARGS, GameState_Docstring},

    /* extensions for data compression */
    {"Board_ToHash", chesslib_board_to_hash, METH_VARARGS, Board_ToHash_Docstring},
    {"Board_FromHash", chesslib_board_from_hash, METH_VARARGS, Board_FromHash_Docstring},

    /* extensions for data visualization of complex type encodings */
    {"VisualizeBoard", chesslib_visualize_board, METH_VARARGS, VisualizeBoard_Docstring},
    {"VisualizeDraw", chesslib_visualize_draw, METH_VARARGS, VisualizeDraw_Docstring},
    /* TODO: add functions for visualizing remaining data structures like chess piece, chess pos, piece@pos */

    PY_METHODS_SENTINEL
};

/* Define the chesslib python module. */
static struct PyModuleDef chesslib_module = {
    PyModuleDef_HEAD_INIT,
    "chesslib",
    "C-lib Python3 extension for efficient chess draw-gen",
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
        || (!isalpha(pos_as_string[0]) || toupper(pos_as_string[0]) - 'A' >= 8
                                       || toupper(pos_as_string[0]) - 'A' < 0)
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
    ChessColor color; ChessPieceType type; int was_moved;

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
  Create an instance of the ChessBoard struct with all chess pieces in
  start formation.
 **************************************************************************/
static PyObject* chesslib_create_chessboard_startformation(PyObject* self, PyObject* args)
{
    int is_simple_board = 0;
    ChessPiece simple_board[64];

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

    /* convert to simple format if needed */
    if (is_simple_board) { to_simple_board(start_formation, simple_board); }

    return is_simple_board
        ? serialize_as_pieces(simple_board)
        : serialize_as_bitboards(start_formation);
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
    Bitboard board[13]; ChessPiece simple_board[64];
    int is_simple_board = 0;

    /* parse all args */
    if (!PyArg_ParseTuple(args, "O|i", &pieces_list, &is_simple_board)) { return NULL; }
    nd_pieces_at_pos = (PyArrayObject*)PyArray_FromObject(pieces_list, NPY_UINT16, 1, 32);
    count = (size_t)PyArray_Size((PyObject*)nd_pieces_at_pos);
    pieces_at_pos = (ChessPieceAtPos*)PyArray_DATA(nd_pieces_at_pos);

    /* read in the pieces@pos as bitboards / simple board */
    create_board_from_piecesatpos(pieces_at_pos, count, board);
    if (is_simple_board) { to_simple_board(board, simple_board); }

    /* return the board as numpy array according to the requested format */
    return is_simple_board
        ? serialize_as_pieces(simple_board)
        : serialize_as_bitboards(board);
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
    Bitboard* board = NULL; ChessDraw draw;
    ChessPosition old_pos = 0, new_pos = 0;
    ChessPieceType prom_type = Invalid;
    int is_compact_format = 0; int is_simple_board = 0;

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
    Bitboard* board; ChessColor drawing_side;
    int analyze_draw_into_check = 0;
    int is_compact_format = 0; int is_simple_board = 0; int i = 0;

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
    Bitboard* board_before; ChessDraw draw_to_apply;
    Bitboard board_after[13]; ChessPiece simple_board_after[64];
    int is_simple_board = 0;

    /* parse input args */
    if (!PyArg_ParseTuple(args, "Oi|i", &chessboard, &draw_to_apply, &is_simple_board)) { return NULL; }
    board_before = deserialize_as_bitboards(chessboard, is_simple_board);
    draw_to_apply = deserialize_chessdraw(board_before, draw_to_apply);

    /* apply the chess draw to a new ChessBoard instance */
    copy_board(board_before, board_after);
    apply_draw(board_after, draw_to_apply);

    /* convert the new board to a simple board if requested */
    if (is_simple_board) { to_simple_board(board_after, simple_board_after); }

    return is_simple_board 
        ? serialize_as_pieces(simple_board_after)
        : serialize_as_bitboards(board_after);
}

/* =================================================
                 G A M E    S T A T E
   ================================================= */

static PyObject* chesslib_get_game_state(PyObject* self, PyObject* args)
{
    PyObject* chessboard;
    Bitboard* board;
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
    ChessPiece* simple_board;

    /* parse bitboards as ChessBoard struct */
    if (!PyArg_ParseTuple(args, "O|i", &chessboard, &is_simple_board)) { return NULL; }
    simple_board = deserialize_as_pieces(chessboard, is_simple_board);

    /* compress the pieces cache to 40 bytes by removing
       the unused leading 3 bits of each ChessPiece value */
    bytes = (uint8_t*)calloc(40, sizeof(uint8_t));
    if (bytes == NULL) { return NULL; }
    compress_pieces_array(simple_board, bytes);

    /* convert parsed bytes to Python bytearray struct */
    return PyArray_SimpleNewFromData(1, (npy_intp*)dims, NPY_UINT8, bytes);
}

/**************************************************************************
  Retrieve a ChessBoard instance of the given 40-byte hash representation.
 **************************************************************************/
static PyObject* chesslib_board_from_hash(PyObject* self, PyObject* args)
{
    PyObject *hash_orig; PyArrayObject* hash;
    uint8_t *compressed_bytes;
    ChessPiece simple_board[64] = { 0 }; Bitboard board[13];
    int is_simple_board = 0;

    /* parse bitboards as ChessBoard struct */
    if (!PyArg_ParseTuple(args, "O|i", &hash_orig, &is_simple_board)) { return NULL; }
    hash = (PyArrayObject*)PyArray_FromObject(hash_orig, NPY_UINT8, 1, 40);
    compressed_bytes = (uint8_t*)PyArray_DATA(hash);

    /* uncompress the pieces cache from 40 bytes by adding
       the unused leading 3 bits of each ChessPiece value */
    uncompress_pieces_array(compressed_bytes, simple_board);
    if (!is_simple_board) { from_simple_board(simple_board, board); }

    /* return as simple board or bitboards format */
    return is_simple_board
        ? serialize_as_pieces(simple_board)
        : serialize_as_bitboards(board);
}

/* =================================================
                  V I S U A L I Z E
   ================================================= */

static PyObject* chesslib_visualize_board(PyObject* self, PyObject* args)
{
    PyObject* bitboards;
    Bitboard* board;
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
    }

    /* write column descriptions */
    strcat(out, "     A    B    C    D    E    F    G    H");

    return Py_BuildValue("s", out);
}

static PyObject* chesslib_visualize_draw(PyObject* self, PyObject* args)
{
    ChessDraw draw;
    char out[100], buf[100], old_pos[3], new_pos[3];
    int is_left_side = 0;

    /* parse bitboards as ChessBoard struct */
    if (!PyArg_ParseTuple(args, "i", &draw)) { return NULL; }
    /* info: a chessboard is required for context information when only providing compact draws
             this function only works for non-compact draws*/

    position_to_string(get_old_position(draw), old_pos);
    position_to_string(get_new_position(draw), new_pos);

    /* determine the chess draw's base representation */
    sprintf(buf, "%s %s %s-%s",
        color_to_string(get_drawing_side(draw)),
        piece_type_to_string(get_drawing_piece_type(draw)),
        old_pos,
        new_pos);
    strcpy(out, buf);

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

/* TODO: add helper function for validating numpy arrays: shape, dtype, ...
         use something like that:
            if (PyArray_NDIM(bitboards_obj) == 64 && PyArray_DTYPE(bitboards_obj) == NPY_UINT8) */

static PyObject* serialize_as_pieces(const ChessPiece simple_board[])
{
    PyObject* nparray; npy_intp dims[1] = { 64 };

    /* create a heap copy of the chess board */
    ChessPiece* data_copy = create_empty_simple_chessboard();
    if (data_copy == NULL) { return NULL; }
    copy_simple_board(simple_board, data_copy);

    /* create a new numpy array from the board data */
    nparray = PyArray_SimpleNewFromData(1, (npy_intp*)dims, NPY_UINT8, data_copy);

    /* grant data ownership to the numpy array -> no memory leaks */
    PyArray_ENABLEFLAGS((PyArrayObject*)nparray, NPY_ARRAY_OWNDATA);

    return nparray;
}

static PyObject* serialize_as_bitboards(const Bitboard board[])
{
    PyObject* nparray; npy_intp dims[1] = { 13 };

    /* create a heap copy of the chess board */
    Bitboard *data_copy = create_empty_chessboard();
    if (data_copy == NULL) { return NULL; }
    copy_board(board, data_copy);

    /* create a new numpy array from the board data */
    nparray = PyArray_SimpleNewFromData(1, (npy_intp*)dims, NPY_UINT64, data_copy);

    /* grant data ownership to the numpy array -> no memory leaks */
    PyArray_ENABLEFLAGS((PyArrayObject*)nparray, NPY_ARRAY_OWNDATA);

    return nparray;
}

static ChessPiece* deserialize_as_pieces(PyObject* bitboards_obj, int is_simple_board)
{
    ChessPiece* out_board = NULL;
    PyArrayObject *bitboards, *pieces;

    /* check if the given board can be interpreted as simple format */
    if (is_simple_board)
    {
        /* parse simple chess board as ndarray of 64 raw bytes */
        pieces = (PyArrayObject*)PyArray_FromObject(bitboards_obj, NPY_UINT8, 1, 64);

        /* convert the simple chess board into the bitboard representation for efficient operations */
        out_board = (ChessPiece*)PyArray_DATA(pieces);
    }
    /* check if the given board can be interpreted as bitboards format */
    else
    {
        /* parse bitboards as 1-dimensional ndarray of type uint64 and size 13 */
        out_board = create_empty_simple_chessboard();
        bitboards = (PyArrayObject*)PyArray_FromObject(bitboards_obj, NPY_UINT64, 1, 13);
        to_simple_board((Bitboard*)PyArray_DATA(bitboards), out_board);
    }

    return out_board;
}

static Bitboard* deserialize_as_bitboards(PyObject* bitboards_obj, int is_simple_board)
{
    Bitboard* out_board = NULL;
    PyArrayObject *bitboards, *pieces;

    /* check if the given board can be interpreted as simple format */
    if (is_simple_board)
    {
        /* parse simple chess board as ndarray of 64 raw bytes */
        pieces = (PyArrayObject*)PyArray_FromObject(bitboards_obj, NPY_UINT8, 1, 64);

        /* convert the simple chess board into the bitboard representation for efficient operations */
        out_board = create_empty_chessboard();
        from_simple_board((ChessPiece*)PyArray_DATA(pieces), out_board);
    }
    /* check if the given board can be interpreted as bitboards format */
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
