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

#include "chessgamestate.h"

int can_achieve_checkmate(const Bitboard board[], ChessColor side);

ChessGameState get_game_state(const Bitboard board[], ChessDraw last_draw)
{
    ChessDraw *draws;
    size_t draws_count;
    ChessColor allied_side, enemy_side;
    int can_ally_checkmate, can_enemy_checkmate, is_ally_checked;
    Bitboard enemy_capturable_fields;

    /* determine allied and enemy side */
    allied_side = (last_draw == DRAW_NULL) ? White : OPPONENT(get_drawing_side(last_draw));
    enemy_side = OPPONENT(allied_side);

    /* analyze the chess piece types on the board => determine whether any player
       can even achieve a checkmate with his remaining pieces */
    can_ally_checkmate = can_achieve_checkmate(board, allied_side);
    can_enemy_checkmate = can_achieve_checkmate(board, enemy_side);

    /* quit game status analysis if ally has lost due to unsufficient pieces */
    if (!can_ally_checkmate && !can_enemy_checkmate) { return Tie; }

    /* find out if any allied chess piece can draw */
    get_all_draws(&draws, &draws_count, board, allied_side, last_draw, 1);

    /* find out whether the allied king is checked */
    enemy_capturable_fields = get_capturable_fields(board, allied_side, last_draw);
    is_ally_checked = (enemy_capturable_fields & board[SIDE_OFFSET(allied_side)]) > 0;

    /* none:      ally can draw and is not checked
       check:     ally is checked, but can at least draw
       stalemate: ally cannot draw but is also not checked (end of game)
       checkmate: ally is checked and cannot draw anymore (end of game) */

    return draws_count
        ? (is_ally_checked ? Check : None)
        : (is_ally_checked ? Checkmate : Tie);
}

int can_achieve_checkmate(const Bitboard board[], ChessColor side)
{
    /* minimal pieces required for checkmate:
       ======================================
        (1) king + queen
        (2) king + rook
        (3) king + 2 bishops (onto different chess field colors)
        (4) king + bishop + knight
        (5) king + 3 knights
        (6) king + peasant (with promotion) */

    int opt_1_2_6, opt_3, opt_4, opt_5;
    ChessPosition first_pos, second_pos;
    Bitboard knights;

    /* check for options 1, 2 or 6: any allied queen, rook or peasant existing */
    opt_1_2_6 = (
          board[SIDE_OFFSET(side) + PIECE_OFFSET(Queen)]
        | board[SIDE_OFFSET(side) + PIECE_OFFSET(Rook)]
        | board[SIDE_OFFSET(side) + PIECE_OFFSET(Peasant)]
    ) > 0;

    /* check for option 3: at least 2 bishops, but on different lanes */
    first_pos = get_board_position(board[SIDE_OFFSET(side) + PIECE_OFFSET(Bishop)]);
    second_pos = get_board_position(
        (0x1uLL << first_pos) ^ board[SIDE_OFFSET(side) + PIECE_OFFSET(Bishop)]);
    opt_3 = (first_pos % 2 == 0) && (second_pos % 2 == 1);

    /* check for option 4: at least 1 bishop and 1 knight */
    opt_4 = (board[SIDE_OFFSET(side) + PIECE_OFFSET(Bishop)] > 0 
        && board[SIDE_OFFSET(side) + PIECE_OFFSET(Knight)] > 0);

    /* check for option 5: at least 3 knights */
    knights = board[SIDE_OFFSET(side) + PIECE_OFFSET(Knight)];
    knights ^= 0x1uLL << get_board_position(knights);
    knights ^= 0x1uLL << get_board_position(knights);
    opt_5 = knights > 0;

    /* check if at least one of the 6 options is true */
    return opt_1_2_6 || opt_3 || opt_4 || opt_5;
}
