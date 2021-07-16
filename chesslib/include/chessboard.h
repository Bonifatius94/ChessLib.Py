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

#ifndef CHESSSBITBOARD_H
#define CHESSSBITBOARD_H

/* ====================================================
            L O A D   D E P E N D E N C I E S
   ==================================================== */

#include "chesspiece.h"
#include "chessposition.h"
#include "chessdraw.h"
#include "chesstypes.h"
#include "chesspieceatpos.h"

#include <stdint.h>

/* ====================================================
                D E F I N E    M A K R O S
   ==================================================== */

/* row masks */
#define ROW_1 0x00000000000000FFuLL
#define ROW_2 0x000000000000FF00uLL
#define ROW_3 0x0000000000FF0000uLL
#define ROW_4 0x00000000FF000000uLL
#define ROW_5 0x000000FF00000000uLL
#define ROW_6 0x0000FF0000000000uLL
#define ROW_7 0x00FF000000000000uLL
#define ROW_8 0xFF00000000000000uLL

/* column masks */
#define COL_A 0x0101010101010101uLL
#define COL_B 0x0202020202020202uLL
#define COL_C 0x0404040404040404uLL
#define COL_D 0x0808080808080808uLL
#define COL_E 0x1010101010101010uLL
#define COL_F 0x2020202020202020uLL
#define COL_G 0x4040404040404040uLL
#define COL_H 0x8080808080808080uLL

/* diagonal masks */
#define WHITE_FIELDS 0x55AA55AA55AA55AAuLL
#define BLACK_FIELDS 0xAA55AA55AA55AA55uLL

/* start formation positions mask */
#define START_POSITIONS 0xFFFF00000000FFFFuLL

/* position masks for kings and rooks on the start formation */
#define FIELD_A1 0x0000000000000001uLL
#define FIELD_C1 0x0000000000000004uLL
#define FIELD_D1 0x0000000000000008uLL
#define FIELD_E1 0x0000000000000010uLL
#define FIELD_F1 0x0000000000000020uLL
#define FIELD_G1 0x0000000000000040uLL
#define FIELD_H1 0x0000000000000080uLL
#define FIELD_A8 0x0100000000000000uLL
#define FIELD_C8 0x0400000000000000uLL
#define FIELD_D8 0x0800000000000000uLL
#define FIELD_E8 0x1000000000000000uLL
#define FIELD_F8 0x2000000000000000uLL
#define FIELD_G8 0x4000000000000000uLL
#define FIELD_H8 0x8000000000000000uLL

#define SIDE_OFFSET(color) (((uint8_t)(color)) * 6)
#define PIECE_OFFSET(piece) (((uint8_t)(piece)) - 1)
#define WHITE_MASK(color) (((Bitboard)(((int64_t)(color)) - 1)))
#define BLACK_MASK(color) (~WHITE_MASK((color)))

/* ====================================================
            D E F I N E     F U N C T I O N S
   ==================================================== */

/* TODO: add interface documentation */

Bitboard* create_empty_chessboard();
void copy_board(const Bitboard orig[], Bitboard* target);
void create_board(const Bitboard bitboards[], Bitboard* target);
void create_board_from_piecesatpos(const ChessPieceAtPos pieces_at_pos[],
    size_t pieces_count, Bitboard* target);

Bitboard is_captured_at(const Bitboard board[], ChessPosition pos);
ChessPiece get_piece_at(const Bitboard board[], ChessPosition pos);
int was_piece_moved(const Bitboard board[], ChessPosition pos);

void apply_draw(Bitboard* bitboards, ChessDraw draw);

void from_simple_board(const ChessPiece simple_board[], Bitboard* target);
void to_simple_board(const Bitboard board[], ChessPiece* target);

#endif
