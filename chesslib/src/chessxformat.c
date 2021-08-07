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

int parse_uint(char* value_str, int* out_value, char term)
{
    int value = 0; size_t i = 0;

    while (isdigit(value_str[i]) && value_str[i] != term)
    { value = value * 10 + (value_str[i] - '0'); i++; }

    *out_value = value;
    return value_str[i] == term ? i : -1;
}

int chess_board_from_fen(const char fen_str[], ChessGameSession* session)
{
    size_t i = 0, sep_count = 0; char temp = '\0'; int is_terminal = 0, len = 0;
    ChessPosition pos = 0; ChessColor color; ChessPieceType type; int was_moved;

    /* parse the first FEN section (positions of pieces on the board) */
    do
    {
        /* access the next FEN character to be parsed */
        temp = fen_str[i++];

        switch (temp)
        {
            /* handle termination character */
            case ' ': if (!is_terminal) { return 0; } break;

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
                session->board[pos++] = create_piece(type, color, was_moved);
                break;

            /* handle invalid / unexpected characters */
            default: return 0;
        }

        /* check if the next state is supposed to be terminal */
        is_terminal = pos == 64 && sep_count == 7;

    /* loop until the end of the FEN string's first section */
    } while (temp != ' ');

    /* parse the second FEN section (possible castlings) */

    /* disable all rochades (enable them gradually if they are still possible) */
    session->board[12] |= FIELD_A1 | FIELD_H1 | FIELD_A8 | FIELD_H8;

    /* handle case with no castlings */
    if (fen_str[i] == '-' && fen_str[i+1] == ' ') { /* do nothing */ }
    else
    {
        /* enable castlings (ensure the correct order) */
        if (fen_str[i] == 'K') { session->board[12] &= ~FIELD_H1; i++; }
        if (fen_str[i] == 'Q') { session->board[12] &= ~FIELD_A1; i++; }
        if (fen_str[i] == 'k') { session->board[12] &= ~FIELD_H8; i++; }
        if (fen_str[i] == 'q') { session->board[12] &= ~FIELD_A8; i++; }

        /* ensure that any rochade is possible and the terminal symbol is hit */
        if (!((~session->board[12] & (FIELD_A1 | FIELD_H1 | FIELD_A8 | FIELD_H8))
            && fen_str[i] == ' ')) { return 0; }
    }

    /* info: the third FEN section (en-passant) can be skipped */
    while ((temp = fen_str[i++]) != '\0' && temp != ' ') { }

    /* parse the fourth FEN section (halfdraws since last pawn draw) */
    len = parse_uint(fen_str + i, &(session->halfdraws_since_last_pawn_draw), ' ');
    if (len > 0) { i += len + 1; } else { return 0; }

    /* parse the fifth FEN section (game round) */
    len = parse_uint(fen_str + i, &(session->game_round), '\0');
    if (len <= 0) { return 0; }

    return 1;
}

int chess_board_to_fen(char** fen_str, const ChessGameSession* session)
{
    /* TODO: implement logic */
    return 0;
}

int chess_draw_from_pgn(const char fen_str[], Bitboard* board)
{
    /* TODO: implement logic */
    return 0;
}

int chess_draw_to_pgn(char** fen_str, Bitboard* board)
{
    /* TODO: implement logic */
    return 0;
}
