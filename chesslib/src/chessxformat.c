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

#include "chessxformat.h"

/* start formation in FEN: 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1' */

/* Parse a positive integer value from the given decimal numeric string
   and return the length of the characters parsed (return -1 on format error). */
int parse_uint(const char* value_str, int* out_value, char term)
{
    int value = 0; size_t i = 0;

    /* make sure there are no heading zeros for non-zero values */
    if (value_str[i] == '0' && value_str[i+1] != term) { return -1; }

    /* parse the uint value from decimal notation */
    while (isdigit(value_str[i]) && value_str[i] != term && value_str[i] != '\0')
    { value = value * 10 + (value_str[i] - '0'); i++; }

    *out_value = value;
    return value_str[i] == term ? i : -1;
}

/* Look up the next appearance of the given character in the given zero-terminated string.
   Return -1 if the string does not contain the character searched. */
int str_index_of(char* search_str, char find)
{
    size_t i = 0; char temp;
    while ((temp = search_str[i++]) != find && temp != '\0') { }
    return temp == find ? i - 1 : -1;
}

int parse_first_fen_section(const char fen_str[], Bitboard* board)
{
    size_t i = 0, sep_count = 0; char temp; int is_terminal = 0;
    ChessPosition pos = 0; ChessColor color; ChessPieceType type; int was_moved;
    ChessPiece simple_board[64] = { 0 };

    /* parse the first FEN section (positions of pieces on the board) */
    do
    {
        /* access the next FEN character to be parsed */
        temp = fen_str[i++];

        switch (temp)
        {
            /* handle termination character */
            case '\0': if (!is_terminal) { return 0; } break;

            /* ensure that the row bounds are not violated */
            case '/': if (pos != ++sep_count * 8) { return 0; } break;

            /* handle empty fields declaration */
            case '1': case '2': case '3': case '4':
            case '5': case '6': case '7': case '8':
                pos += (ChessPosition)(temp - '0'); break;

            /* put another piece on the chess board */
            case 'K': case 'Q': case 'R': case 'B': case 'N': case 'P':
            case 'k': case 'q': case 'r': case 'b': case 'n': case 'p':
                color = (ChessColor)(isupper(temp) ? White : Black);
                type = piece_type_from_char(temp);
                was_moved = START_POSITIONS & (1uLL << pos) ? 0 : 1;
                simple_board[pos++] = create_piece(type, color, was_moved);
                break;

            /* handle invalid / unexpected characters */
            default: return 0;
        }

        /* check if the next state is supposed to be terminal */
        is_terminal = pos == 64 && sep_count == 7;

    /* loop until the end of the FEN string's first section */
    } while (temp != '\0');

    /* convert the simple board to a bitboard representation */
    from_simple_board(simple_board, board);

    return 1;
}

int parse_second_fen_section(const char fen_str[], ChessColor* side)
{
    ChessColor color = color_from_char(fen_str[0]);
    if (fen_str[1] != '\0') { return 0; }
    *side = color;
    return 1;
}

int parse_third_fen_section(const char fen_str[], uint8_t* rochades)
{
    size_t i = 0; uint8_t poss_rochades = 0xF;

    /* handle case with no castlings */
    if (fen_str[i] == '-' && fen_str[i+1] == '\0') { /* do nothing */ }

    /* handle case with rochades */
    else
    {
        /* enable castlings (ensure the correct order) */
        if (fen_str[i] == 'K') { poss_rochades ^= 0x2; i++; }
        if (fen_str[i] == 'Q') { poss_rochades ^= 0x1; i++; }
        if (fen_str[i] == 'k') { poss_rochades ^= 0x8; i++; }
        if (fen_str[i] == 'q') { poss_rochades ^= 0x4; i++; }

        /* ensure that any rochade is possible and the terminal symbol is hit */
        if (!poss_rochades || fen_str[i] != '\0') { return 0; }
    }

    /* apply the parsed rochades to the game context */
    *rochades = poss_rochades;

    return 1;
}

int chess_session_from_fen(const char fen_str[], ChessGameSession* session)
{
    size_t end = 0; size_t len, i = 0; int game_round = 0;
    ChessColor side; uint8_t rochades = 0, en_passants = 0, hdslpd = 0;
    char cache[54]; char* temp_str = cache; Bitboard board[13] = { 0 };

    /* create a carbon copy of the fen string (that can be safely modified) */
    strcpy(temp_str, fen_str);

    /* parse the first FEN section (positions of pieces on the board) */
    if ((end = str_index_of(temp_str, ' ')) == -1) { return 0; }
    temp_str[end] = '\0';
    if (!parse_first_fen_section(temp_str, board)) { return 0; }
    temp_str += end + 1;

    /* parse the second FEN section (drawing side) */
    if ((end = str_index_of(temp_str, ' ')) == -1) { return 0; }
    temp_str[end] = '\0';
    if (!parse_second_fen_section(temp_str, &side)) { return 0; }
    temp_str += end + 1;

    /* parse the third FEN section (possible castlings) */
    if ((end = str_index_of(temp_str, ' ')) == -1) { return 0; }
    temp_str[end] = '\0';
    if (!parse_third_fen_section(temp_str, &rochades)) { return 0; }
    temp_str += end + 1;

    /* parse the fourth FEN section (en-passant) */
    /* TODO: add parsing logic */

    /* parse the fifth FEN section (halfdraws since last pawn draw) */
    len = parse_uint(fen_str + i, (int*)&hdslpd, ' ');
    if (len > 0) { i += len + 1; } else { return 0; }

    /* parse the sixth FEN section (game round) */
    len = parse_uint(fen_str + i, &game_round, '\0');
    if (len <= 0) { return 0; }

    /* assign the parsed FEN string content to the game session object */
    copy_board(board, session->board);
    session->context = create_context(side,
        en_passants, rochades, hdslpd, game_round);

    return 1;
}

int chess_session_to_fen(char* fen_str, const ChessGameSession* session)
{
    size_t i = 0, empty_cnt = 0; ChessPosition pos; ChessPiece piece; char temp;
    ChessPiece simple_board[64] = { 0 };

    /* write the first section to the game session (pieces on board) */
    to_simple_board(session->board, simple_board);

    for (pos = 0; pos < 64; pos++)
    {
        piece = simple_board[pos];

        /* in case the field position is not empty */
        if (piece != CHESS_PIECE_NULL)
        {
            /* write the piece type (either uppercase for white or lowercase for black) */
            temp = piece_type_to_char(get_piece_type(piece));
            fen_str[i++] = get_piece_color(piece)
                ? tolower(temp) : toupper(temp);
        }
        else { empty_cnt++; }

        /* handle empty field spaces */
        if ((piece != CHESS_PIECE_NULL || (pos + 1) % 8 == 0) && empty_cnt > 0)
        { fen_str[i++] = '0' + empty_cnt; empty_cnt = 0; }

        /* handle end-of-row separator */
        if ((pos + 1) % 8 == 0 && pos < 63) { fen_str[i++] = '/'; }
    }

    /* TODO  */

    fen_str[i] = '\0';
    return 1;
}

int chess_session_from_pgn(const char pgn_str[], ChessGameSession* session)
{
    /* TODO: implement logic */
    return 0;
}

int chess_session_to_pgn(char* pgn_str, const ChessGameSession* session)
{
    /* TODO: implement logic */
    return 0;
}
