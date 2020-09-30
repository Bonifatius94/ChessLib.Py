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

#ifndef CHESSPIECE_H
#define CHESSPIECE_H

/* ====================================================
            L O A D   D E P E N D E N C I E S
   ==================================================== */

#include "chesstypes.h"
#include <stdint.h>
#include <ctype.h>

/* ====================================================
               D E F I N E    M A K R O S
   ==================================================== */

#define CHESS_PIECE_NULL ((ChessPiece)0)
#define OPPONENT(color) ((ChessColor)((uint8_t)(color) ^ 1))

/* ====================================================
             D E F I N E    F U N C T I O N S
   ==================================================== */

ChessPiece create_piece(ChessPieceType type, ChessColor color, int was_moved);

ChessColor get_piece_color(ChessPiece piece);
int get_was_piece_moved(ChessPiece piece);
ChessPieceType get_piece_type(ChessPiece piece);

ChessColor color_from_char(char c);
ChessPieceType piece_type_from_char(char c);
char color_to_char(ChessColor color);
char piece_type_to_char(ChessPieceType type);
char* color_to_string(ChessColor color);
char* piece_type_to_string(ChessPieceType type);

#endif
