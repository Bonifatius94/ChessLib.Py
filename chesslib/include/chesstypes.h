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

#ifndef CHESSTYPES_H
#define CHESSTYPES_H

#include <stdint.h>

/* include size_t type for gcc */
#ifdef __GNUC__
    #include <stddef.h>
#endif

typedef enum _CHESS_PIECE_TYPE {
    Invalid = 0,
    King = 1,
    Queen = 2,
    Rook = 3,
    Bishop = 4,
    Knight = 5,
    Peasant = 6
} ChessPieceType;

typedef enum _CHESS_COLOR {
    White = 0,
    Black = 1
} ChessColor;

typedef enum _CHESS_DRAW_TYPE {
    Standard = 0,
    Rochade = 1,
    EnPassant = 2,
    PeasantPromotion = 3
} ChessDrawType;

/* Representation of chess game states. */
typedef enum _CHESS_GAME_STATE {
    None = 'N',
    Check = 'C',
    Checkmate = 'M',
    Tie = 'T',
} ChessGameState;

/* | was moved | color | piece type |
   |         x |     x |        xxx | */
typedef uint8_t ChessPiece;

/* |  row | column |
   |  xxx |    xxx | */
typedef uint8_t ChessPosition;

/* | position | piece |
   |   xxxxxx | xxxxx | */
typedef uint16_t ChessPieceAtPos;

/* |  unused | is first move | side | draw type | piece type | taken piece type | promotion type | old position | new position |
   | xxxxxxx |             x |    x |        xx |        xxx |              xxx |            xxx |       xxxxxx |       xxxxxx | */
typedef uint32_t ChessDraw;

/* | prom. piece type | old position | new position | 
   |              xxx |       xxxxxx |       xxxxxx | */
typedef uint16_t CompactChessDraw;

/* A chess bitboard with each bit representing a field onto a chess board. Addressing is normalized, starting with the lowest bit as A1 and the highest bit as H8 (indexes A1=0, B1=1, ..., A2=8, ..., H8=63). */
typedef uint64_t Bitboard;

/* The chess board represented as 13 bitboards of unsigned 64-bit integers. 
   The chess field allocation is normalized by the index values corresponding to ChessPosition type starting with the lowest bit as A1 and ending with the highest bit as H8 (indexes A1=0, B1=1, ..., A2=8, ..., H8=63).
   The first 12 boards show positions of the chess pieces, the last board keeps track of was_moved states.
   The boards holding information on chess pieces are ordered by the occurance of the piece in the ChessPieceType enum (King=0, Queens=1, Rooks=2, Bishops=3, Knights=4, Peasants=5).
   All boards with indices 0-5 belong to the white side, the next 6 boards with indices 6-11 belong to the black side.
*/
typedef Bitboard * ChessBoard;

/*
 * The chess board represented as 64 bytes, each modelling the chess piece standing at the specific position of the board.
 * In case there is no piece at the given position, it is assigned to CHESS_PIECE_NULL.
 */
typedef ChessPiece * SimpleChessBoard;

#endif
