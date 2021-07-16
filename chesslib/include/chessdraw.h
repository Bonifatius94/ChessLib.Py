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

#ifndef CHESSDRAW_H
#define CHESSDRAW_H

/* ====================================================
            L O A D   D E P E N D E N C I E S
   ==================================================== */

#include "chesspiece.h"
#include "chessposition.h"
#include "chessboard.h"
#include "chesstypes.h"

#include <stdint.h>
#include <stdlib.h>

/* ====================================================
               D E F I N E    M A K R O S
   ==================================================== */

#define DRAW_NULL ((ChessDraw)0)

/* ====================================================
             D E F I N E    F U N C T I O N S
   ==================================================== */

/* TODO: add interface documentation */

ChessDraw create_draw(const Bitboard board[], ChessPosition oldPos, ChessPosition newPos, ChessPieceType peasantPromotionType);
ChessDraw from_compact_draw(const Bitboard board[], CompactChessDraw comp_draw);
CompactChessDraw to_compact_draw(ChessDraw draw);

int get_is_first_move(ChessDraw draw);
ChessDrawType get_draw_type(ChessDraw draw);
ChessColor get_drawing_side(ChessDraw draw);
ChessPieceType get_drawing_piece_type(ChessDraw draw);
ChessPieceType get_taken_piece_type(ChessDraw draw);
ChessPieceType get_peasant_promotion_piece_type(ChessDraw draw);
ChessPosition get_old_position(ChessDraw draw);
ChessPosition get_new_position(ChessDraw draw);

#endif
