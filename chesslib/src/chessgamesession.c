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

const ChessGameContext SIDE_MASK       = 0x00000001uL;
const ChessGameContext EN_PASSANT_MASK = 0x000001FEuL;
const ChessGameContext ROCHADE_MASK    = 0x00001E00uL;
const ChessGameContext HDSLPD_MASK     = 0x0007E000uL;
const ChessGameContext GAME_ROUND_MASK = 0xFFF80000uL;

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
    return (uint8_t)((context & HDSLPD_MASK) >> 13);
}

int get_game_rounds(ChessGameContext context)
{
    /* extract the 13 bits counting the game rounds */
    return (int)((context & GAME_ROUND_MASK) >> 19);
}

void apply_draw_to_context(ChessDraw draw, ChessGameContext* context)
{
    ChessGameContext temp = *context;

    /* alternate the drawing side */
    *context ^= 0x1uL;

    /* increment the game round counter (if white is drawing) */
    if (GC_DRAWING_SIDE(*context) == White) {
        *context = (((ChessGameContext)get_game_rounds(temp) + 1) << 19)
            | (*context & ~GAME_ROUND_MASK);
    }

    /* update the hdslpd counter (either increment or reset) */
    *context = *context & ~HDSLPD_MASK; /* reset bits */
    if (get_drawing_piece_type(draw) != Peasant) {
        *context |= (((ChessGameContext)get_hdslpd(temp) + 1) << 13);
    }

    /* handle en-passant (if a peasant moved double-forward) */
    *context = *context & ~EN_PASSANT_MASK; /* reset bits */
    if (get_drawing_piece_type(draw) == Peasant
        && abs(get_new_position(draw) - get_old_position(draw)) == 16
        && ((get_drawing_side(draw) ? ROW_5 : ROW_4) & get_new_position(draw)))
    {
        /* determine the column of the possible en-passant and set the bit */
        *context |= ((ChessGameContext)0x1uL << (get_new_position(draw) % 8 + 1));
    }

    /* handle rochade (if a king or a rook moved) */
    *context = *context & ~ROCHADE_MASK; /* reset bits */
    if (get_drawing_piece_type(draw) == King
            && (get_old_position(draw) & (FIELD_E1 | FIELD_E8)))
    {
        /*  disable the rochade bits of the given side accordingly */
        *context |= (get_drawing_side(draw) ? 0x1800uL : 0x600uL);
    }
    if (get_drawing_piece_type(draw) == Rook
            && (get_old_position(draw) & (FIELD_A1 | FIELD_H1 | FIELD_A8 | FIELD_H8)))
    {
        /* disable the rochade bit of a single rook accordingly */
        *context |= (ChessGameContext)(
              ((get_old_position(draw) & FIELD_A1) << 9)    /* shift to 10th bit */
            | ((get_old_position(draw) & FIELD_H1) << 3)    /* shift to 11th bit */
            | ((get_old_position(draw) & FIELD_A8) >> 54)   /* shift to 12th bit */
            | ((get_old_position(draw) & FIELD_H8) >> 60)); /* shift to 13th bit */
            /* TODO: make sure the bit shifts are correct */
    }
}