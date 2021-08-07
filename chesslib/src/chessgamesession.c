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

#include "chessgamesession.h"

#define GC_DRAWING_SIDE(context) (ChessColor)((context) & 0x1uL)

ChessGameContext create_context(ChessColor side, uint8_t en_passants,
    uint8_t rochades, uint8_t halfdraws_since_last_pawn_draw, int game_round)
{
    /* combine the attributes to a single 32-bit integer */
    return ((ChessGameContext)side & 0x1uL)
        || (((ChessGameContext)en_passants & 0xFFuL) << 1)
        || (((ChessGameContext)rochades & 0xFuL) << 9)
        || (((ChessGameContext)halfdraws_since_last_pawn_draw & 0x3FuL) << 13)
        || ((ChessGameContext)game_round << 19);
}

Bitboard get_en_passant_mask(ChessGameContext context)
{
    /* extract the 8 en-passant bits and transform it into a bit mask */
    uint8_t shift = 8 * (GC_DRAWING_SIDE(context) == White ? 2 : 5);
    return ((Bitboard)((context) >> 1) & ROW_1) << shift;
}

Bitboard get_rochade_mask(ChessGameContext context)
{
    /* extract the 4 bits indicating possible rochades */
    Bitboard rochades = (Bitboard)((context) >> 9) & 0xFuLL;

    /* transform the rochade bits into a bitboard mask */
    return (rochades & 0x1uLL)         /* rook on A1 */
         | (rochades & 0x2uLL) << 6    /* rook on H1 */
         | (rochades & 0x4uLL) << 54   /* rook on A8 */
         | (rochades & 0x8uLL) << 60;  /* rook on H8 */
}

uint8_t get_hdslpd(ChessGameContext context)
{
    /* extract the 6 bits counting the halfdraws since the last pawn draw */
    return (uint8_t)((context >> 13) & 0x3FuL);
}

int get_game_rounds(ChessGameContext context)
{
    /* extract the 13 bits counting the game rounds */
    return (int)(context >> 19);
}

void apply_draw_to_context(ChessDraw draw, ChessGameContext* context)
{
    /* alternate the drawing side */
    *context ^= 0x1uL;

    /* increment the game round counter (if white is drawing) */
    if (GC_DRAWING_SIDE(*context) == White) {
        *context = (((ChessGameContext)get_game_rounds(*context) + 1) << 19)
            | (*context & 0x7FFFFuL);
    }

    /* update the hdslpd counter (either increment or reset) */
    *context = *context & 0xFFF81FFFuL; /* reset bits */
    if (get_drawing_piece_type(draw) != Peasant) {
        *context |= (((ChessGameContext)get_hdslpd(*context) + 1) << 13);
    }

    /* TODO: handle rochade / en-passant */
}