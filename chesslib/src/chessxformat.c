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

int chess_board_from_fen(const char fen_str[], Bitboard* board)
{
    size_t i = 0; char temp = '\0'; size_t sep_count = 0;
    ChessPosition pos = 0; ChessColor color; ChessPieceType type; int was_moved;

    /* parse the first FEN section (positions of pieces on the board) */
    do
    {
        /* access the next FEN character to be parsed */
        temp = fen_str[i++];

        switch (temp)
        {
            /* handle ignored valid characters (do nothing) */
            case ' ': case '\0': break;

            /* ensure that the row bounds are not violated */
            case '/': if (pos != ++sep_count * 8) { return 0; } break;

            /* handle empty fields declaration (increment position) */
            case '1': case '2': case '3': case '4':
            case '5': case '6': case '7': case '8':
                pos += (ChessPosition)(temp - '0'); break;

            /* handle a piece to be put on the chess board */
            case 'K': case 'Q': case 'R': case 'B': case 'N': case 'P':
            case 'k': case 'q': case 'r': case 'b': case 'n': case 'p':
                color = (ChessColor)(isupper(temp) ? White : Black);
                type = piece_type_from_char(temp);
                was_moved = START_POSITIONS & (1uLL << pos) ? 0 : 1;
                board[pos++] = create_piece(type, color, was_moved);
                break;

            /* handle invalid / unexpected characters */
            default: return 0;
        }

    /* loop until the end of the FEN string's first section */
    } while (temp != '\0' && temp != ' ');

    /* make sure that parsing the first section was successful */
    if (temp != ' ' || pos != 64 || sep_count != 7) { return 0; }

    /* parse the second FEN section (possible castlings) */
    

    return 1;
}

int chess_board_to_fen(char** fen_str, const ChessGameSession session[])
{

}

int chess_draw_from_pgn(const char fen_str[], Bitboard* board)
{

}

int chess_draw_to_pgn(char** fen_str, Bitboard* board)
{

}
