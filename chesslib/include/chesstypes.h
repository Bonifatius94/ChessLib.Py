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

/* Representation of chess piece types. */
typedef enum _CHESS_PIECE_TYPE {
    Invalid = 0,
    King = 1,
    Queen = 2,
    Rook = 3,
    Bishop = 4,
    Knight = 5,
    Peasant = 6
} ChessPieceType;

/* Representation of chess colors. */
typedef enum _CHESS_COLOR {
    White = 0,
    Black = 1
} ChessColor;

/* Representation of chess draw types. */
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
   | --------- | ----- | ---------- |
   |         x |     x |        xxx | */
typedef uint8_t ChessPiece;

/* |  row | column |
   | ---- | ------ |
   |  xxx |    xxx | */
typedef uint8_t ChessPosition;

/* | position | piece |
   | -------- | ----- |
   |   xxxxxx | xxxxx | */
typedef uint16_t ChessPieceAtPos;

/* |  unused | is first move | side | draw type | piece type | taken piece type | promotion type | old position | new position |
   |-------- | ------------- | ---- | --------- | ---------- | ---------------- | -------------- | ------------ | ------------ |
   | xxxxxxx |             x |    x |        xx |        xxx |              xxx |            xxx |       xxxxxx |       xxxxxx | */
typedef uint32_t ChessDraw;

/* | prom. piece type | old position | new position |
   | ---------------- | ------------ | ------------ |
   |              xxx |       xxxxxx |       xxxxxx | */
typedef uint16_t CompactChessDraw;

/* A chess bitboard with each bit representing a field onto a chess board. 
   Addressing is normalized, starting with the lowest bit as A1 and the
   highest bit as H8 (indexes A1=0, B1=1, ..., A2=8, ..., H8=63). */
typedef uint64_t Bitboard;

/* TODO: add a definition for a FEN game session context to replace the was_moved bitboards
         e.g. 1 bit for drawing_side, 8 bits for en-passants, 4 bits for rochades,
              6 bits for draws_since_last_pawn_draw, remaining bits for game_round
         -> new 32-bit integer bitwise type */

/* |    game round | halfdraws since pawn draw | rochades | en-passants | side |
   | ------------- | ------------------------- | -------- | ----------- | ---- |
   | xxxxxxxxxxxxx |                    xxxxxx |     xxxx |    xxxxxxxx |    x | */
typedef uint32_t ChessGameContext;

#endif
