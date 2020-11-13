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

#include "chessboard.h"

ChessBoard create_board(const Bitboard bitboards[])
{
    /* TODO: check if memory allocation works */
    size_t i;
    ChessBoard board = (ChessBoard)malloc(13 * sizeof(Bitboard));
    if (!board) { return NULL; }
    for (i = 0; i < 13; i++) { board[i] = bitboards[i]; }
    return board;
}

ChessBoard create_board_from_piecesatpos(const ChessPieceAtPos pieces_at_pos[], size_t pieces_count)
{
    size_t i;
    ChessBoard board = (Bitboard*)calloc(13, sizeof(Bitboard));
    if (!board) { return NULL; }

    uint8_t board_index;
    ChessPosition pos;
    ChessPiece piece;

    /* assume pieces as already moved */
    board[12] = 0xFFFFFFFFFFFFFFFFuLL;

    /* loop through the pieces@pos array */
    for (i = 0; i < pieces_count; i++)
    {
        /* determine the piece and position */
        piece = get_pieceatpos_piece(pieces_at_pos[i]);
        pos = get_pieceatpos_position(pieces_at_pos[i]);

        /* determine the board to apply the piece to */
        board_index = SIDE_OFFSET(get_piece_color(piece)) + PIECE_OFFSET(get_piece_type(piece));

        /* apply the piece to the bitboard */
        board[board_index] |= 0x1uLL << pos;

        /* apply was_moved state of the chess piece to the bitboard */
        /* the chess pieces are assumed to be already moved, so only flip the bit if the piece was not moved */
        board[12] ^= (((uint64_t)(get_was_piece_moved(piece) ^ 1)) << pos) & START_POSITIONS;
    }

    return board;
}

Bitboard is_captured_at(ChessBoard board, ChessPosition pos)
{
    Bitboard mask, all_pieces;

    mask = 0x1uLL << pos;

	/* combine all bitboards to one bitboard by bitwise OR */
    all_pieces = board[0] | board[1] | board[2] | board[3] | board[4] | board[5]
        | board[6] | board[7] | board[8] | board[9] | board[10] | board[11];

    return (all_pieces & mask);
}

ChessPiece get_piece_at(ChessBoard board, ChessPosition pos)
{
    int i;
    ChessPiece piece = CHESS_PIECE_NULL;
    ChessPieceType type;
    ChessColor color;

    /* only create a chess piece if the board is captured at the given position */
    if (is_captured_at(board, pos))
    {
        type = Invalid;
        color = White;

        /* determine the piece type and color */
        for (i = 0; i < 12; i++)
        {
            if (board[i] & (0x1uLL << pos))
            {
                 type = (ChessPieceType)((i % 6) + 1);
                 color = (ChessColor)(i / 6);
                 break;
            }
        }

        piece = create_piece(type, color, was_piece_moved(board, pos) == 0 ? 0 : 1);
    }

    return piece;
}

int was_piece_moved(ChessBoard board, ChessPosition pos)
{
    return ((~START_POSITIONS | board[12]) & (0x1uLL << pos)) > 0;
}

ChessBoard apply_draw(ChessBoard board, ChessDraw draw)
{
    uint8_t i;
    ChessBoard new_board;

    new_board = (ChessBoard)malloc(13 * sizeof(Bitboard));
    if (!new_board) { return NULL; }

    for (i = 0; i < 13; i++) { new_board[i] = board[i]; }

    apply_draw_to_bitboards(new_board, draw);
    return new_board;

    /* TODO: check if the memory allocation actually works */
}

void apply_draw_to_bitboards(ChessBoard bitboards, ChessDraw draw)
{
    Bitboard old_pos, new_pos, white_mask, black_mask, target_column;
    uint8_t rooks_board_index, side_offset, drawing_board_index, taken_piece_bitboard_index, promotion_board_index;

    /* determine bitboard masks of the drawing piece's old and new position */
    old_pos = 0x1uLL <<  get_old_position(draw);
    new_pos = 0x1uLL << get_new_position(draw);

    /* determine the bitboard index of the drawing piece */
    side_offset = SIDE_OFFSET(get_drawing_side(draw));
    drawing_board_index = PIECE_OFFSET(get_drawing_piece_type(draw)) + side_offset;

    /* set was moved */
    if (get_is_first_move(draw) && (bitboards[drawing_board_index] & old_pos)) { bitboards[12] ^= (old_pos | new_pos) & START_POSITIONS; }
    else if (get_is_first_move(draw)) { bitboards[12] ^= (old_pos | new_pos) & START_POSITIONS; }

    /* move the drawing piece by flipping its' bits at the old and new position on the bitboard */
    bitboards[drawing_board_index] ^= old_pos | new_pos;

    /* handle rochade: move casteling rook accordingly, king will be moved by standard logic */
    if (get_draw_type(draw) == Rochade)
    {
        /* determine the rooks bitboard */
        rooks_board_index = PIECE_OFFSET(Rook) + side_offset;

        /* move the casteling rook by filpping bits at its' old and new position on the bitboard */
        bitboards[rooks_board_index] ^=
              ((new_pos & COL_C) << 1) | ((new_pos & COL_C) >> 2)  /* big rochade   */
            | ((new_pos & COL_G) << 1) | ((new_pos & COL_G) >> 1); /* small rochade */
    }

    /* handle catching draw: remove caught enemy piece accordingly */
    if (get_taken_piece_type(draw) != Invalid)
    {
        /* determine the taken piece's bitboard */
        taken_piece_bitboard_index = SIDE_OFFSET(OPPONENT(get_drawing_side(draw))) + PIECE_OFFSET(get_taken_piece_type(draw));

        /* handle en-passant: remove enemy peasant accordingly, drawing peasant will be moved by standard logic */
        if (get_draw_type(draw) == EnPassant)
        {
            /* determine the white and black mask */
            white_mask = WHITE_MASK(get_drawing_side(draw));
            black_mask = ~white_mask;

            /* catch the enemy peasant by flipping the bit at his position */
            target_column = COL_A << get_column(get_new_position(draw));
            bitboards[taken_piece_bitboard_index] ^=
                  (white_mask & target_column & ROW_5)  /* caught enemy white peasant */
                | (black_mask & target_column & ROW_4); /* caught enemy black peasant */
        }
        /* handle normal catch: catch the enemy piece by flipping the bit at its' position on the bitboard */
        else { bitboards[taken_piece_bitboard_index] ^= new_pos; }
    }

    /* handle peasant promotion: wipe peasant and put the promoted piece */
    if (get_peasant_promotion_piece_type(draw) != Invalid)
    {
        /* remove the peasant at the new position */
        bitboards[drawing_board_index] ^= new_pos;

        /* put the promoted piece at the new position instead */
        promotion_board_index = side_offset + PIECE_OFFSET(get_peasant_promotion_piece_type(draw));
        bitboards[promotion_board_index] ^= new_pos;
    }
}
