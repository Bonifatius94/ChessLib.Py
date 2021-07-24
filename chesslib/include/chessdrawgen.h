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

#ifndef CHESSDRAWGEN_H
#define CHESSDRAWGEN_H

#include "chesstypes.h"
#include "chessboard.h"
#include "chessdraw.h"
#include "chesspiece.h"
#include "chessposition.h"

#include <stdlib.h>
#include <string.h>

/* TODO: add interface documentation */

void get_all_draws(ChessDraw** out_draws, size_t* out_length, const Bitboard board[],
    ChessColor drawing_side, ChessDraw last_draw, int analyze_draw_into_check);

void get_board_positions(Bitboard bitboard, ChessPosition* out_positions, size_t* out_length);
ChessPosition get_board_position(Bitboard bitboard);

Bitboard get_capturable_fields(const Bitboard bitboards[], ChessColor side, ChessDraw last_draw);
Bitboard get_captured_fields(const Bitboard bitboards[], ChessColor side);

#endif
