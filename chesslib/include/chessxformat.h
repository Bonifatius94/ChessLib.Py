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

#ifndef CHESSXFORMAT_H
#define CHESSXFORMAT_H

#include <stdlib.h>
#include "chesstypes.h"
#include "chesspiece.h"
#include "chessboard.h"

/* A chess game session semantically at least as powerful as the FEN notation.
   It is supposed to help carrying out professional chess matches. */
typedef struct _CHESS_GAME_SESSION {
   Bitboard board[13];
   ChessColor drawing_side;
   int halfdraws_since_last_pawn_draw;
   int game_round;
} ChessGameSession;

#define INIT_GAME_SESSION {START_FORMATION, White, 0, 1}

int chess_board_from_fen(const char fen_str[], ChessGameSession* session);
int chess_board_to_fen(char** fen_str, const ChessGameSession* session);
int chess_draw_from_pgn(const char fen_str[], Bitboard* board);
int chess_draw_to_pgn(char** fen_str, Bitboard* board);

#endif