
#include "chesslibmodule.h"
#include <numpy/arrayobject.h>

/* TODO: implement python-lib interface here ... */

/* =================================================
                 I N I T I A L I Z E    
              P Y T H O N    M O D U L E
   ================================================= */

PyObject* PyInit_chesslib()
{
    /* TODO: implement logic here ... */
    return NULL;
}

/* ================================================= 
       C R E A T E    D A T A    O B J E C T S

         -    C H E S S    P O S I T I O N
         -    C H E S S    P I E C E
         -    C H E S S    P I E C E    AT   POSITION
         -    C H E S S    B O A R D
         -    C H E S S    D R A W
   ================================================= */

PyObject* chesslib_create_chessposition(PyObject* self, PyObject* args)
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
    if (strlen(pos_as_string) == 2)
    {
        row = pos_as_string[1] - '1';
        column = toupper(pos_as_string[0]) - 'A';
    }

    return PyLong_FromUnsignedLong(create_position(row, column));
}

PyObject* chesslib_create_chesspiece(PyObject* self, PyObject* args)
{
    const char color_as_char, type_as_char;
    ChessColor color;
    ChessPieceType type;
    int was_moved;

    /* read chess color and chess piece type string, quit if the parameter does not exist */
    /* read was moved boolean, quit if the parameter does not exist */
    if (!PyArg_ParseTuple(args, "cci", &color_as_char, &type_as_char, &was_moved)) { return NULL; }

    /* parse the chess piece color */
    switch (toupper(color_as_char))
    {
        case 'W': color = White; break;
        case 'B': color = Black; break;
        default: return NULL;    break;
    }

    /* parse the chess piece type */
    switch (toupper(type_as_char))
    {
        case 'K': type = King;     break;
        case 'Q': type = Queen;   break;
        case 'R': type = Rook;    break;
        case 'B': type = Bishop;  break;
        case 'N': type = Knight;  break;
        case 'P': type = Peasant; break;
        default: return NULL;
    }

    return PyLong_FromUnsignedLong(create_piece(type, color, was_moved));
}

PyObject* chesslib_create_chesspieceatpos(PyObject* self, PyObject* args)
{
    ChessPiece piece;
    ChessPosition pos;

    /* read chess piece and chess position, quit if the parameters do not exist */
    if (!PyArg_ParseTuple(args, "ii", &piece, &pos)) { return NULL; }

    /* make sure that the chess piece and chess position value are withing their numeric bounds */
    if (piece >= 32 || pos >= 64) { return NULL; }

    return PyLong_FromUnsignedLong(create_pieceatpos(pos, piece));
}

PyObject* serialize_chessboard(ChessBoard board);

PyObject* chesslib_create_chessboard(PyObject* self, PyObject* args)
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
    while (temp_obj = PyIter_Next(iterator))
    {
        if (!PyLong_Check(temp_obj)) { return NULL; }
        pieces_at_pos[offset++] = (ChessPieceAtPos)PyLong_AsLong(temp_obj);
    }

    /* create the chess board */
    board = create_board_from_piecesatpos(pieces_at_pos, offset);
    return serialize_chessboard(board);
}

PyObject* chesslib_create_chessboard_startformation(PyObject* self, PyObject* args)
{
    /* create the chess board */
    ChessBoard board = START_FORMATION;
    return serialize_chessboard(board);
}

PyObject* serialize_chessboard(ChessBoard board)
{
    int dims[1];

    /* init a one-dimensional 64-bit integer array with 13 elements */
    dims[0] = 13;
    return PyArray_SimpleNewFromData(1, dims, NPY_UINT64, board.bitboards);
}

PyObject* chesslib_create_chessdraw(PyObject* self, PyObject* args)
{
    /* TODO: implement the creation of chessdraws */
    return NULL;
}

/* =================================================
             C A L L    D R A W - G E N
   ================================================= */

PyObject* chesslib_get_all_draws(PyObject* self, PyObject* args)
{
    ChessDraw* out_draws, last_draw;
    size_t dims[1];
    ChessBoard board;
    ChessColor drawing_side;
    int analyze_draw_into_check;

    /* TODO: parse args (board, drawing_side, last_draw, analyze_draw_into_check) */

    /* compute possible draws */
    get_all_draws(&out_draws, dims, board, drawing_side, last_draw, analyze_draw_into_check);

    /* serialize draws as numpy list */
    return PyArray_SimpleNewFromData(1, dims, NPY_UINT32, out_draws);
}
