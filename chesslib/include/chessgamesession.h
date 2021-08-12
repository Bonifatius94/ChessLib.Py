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

#ifndef CHESSGAMESESSION_H
#define CHESSGAMESESSION_H

#include "chesstypes.h"
#include "chessboard.h"
#include "chessdraw.h"

/* A chess game session semantically at least as powerful as the FEN notation.
   It is supposed to help carrying out professional chess matches. */
typedef struct _CHESS_GAME_SESSION {
   Bitboard board[13];
   ChessGameContext context;
} ChessGameSession;

/* default game context bits: 00000000 00001000 00011110 00000000
 *
 * meaning:
 * first game round, all rochades possible, no en-passant possible,
 * white drawing and zero halfdraws since the last pawn draw.
 */
#define DEFAULT_GAME_CONTEXT 0x00081E00uL

/* initial game session: board in start formation and initial game context */
#define INIT_GAME_SESSION {START_FORMATION, DEFAULT_GAME_CONTEXT}

ChessGameContext create_context(ChessColor side, uint8_t en_passants,
    uint8_t rochades, uint8_t halfdraws_since_last_pawn_draw, int game_round);

Bitboard get_en_passant_mask(ChessGameContext context);
Bitboard get_rochade_mask(ChessGameContext context);
uint8_t get_hdslpd(ChessGameContext context);
int get_game_rounds(ChessGameContext context);

void apply_draw_to_context(ChessDraw draw, ChessGameContext* context);
void apply_game_context_to_board(Bitboard* board, ChessGameContext context);

#endif