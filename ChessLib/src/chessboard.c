#include "chessboard.h"

ChessBoard create_board(const Bitboard bitboards[])
{
	/* TODO: check if memory allocation works */
    size_t i;
    ChessBoard board;
    for (i = 0; i < 13; i++) { board.bitboards[i] = bitboards[i]; }
	return board;
}

Bitboard is_captured_at(ChessBoard board, ChessPosition pos)
{
    Bitboard mask, all_pieces;

    mask = 0x1uLL << pos;

	/* combine all bitboards to one bitboard by bitwise OR */
    all_pieces = board.bitboards[0] | board.bitboards[1] | board.bitboards[2]
		| board.bitboards[3] | board.bitboards[4] | board.bitboards[5]
		| board.bitboards[6] | board.bitboards[7] | board.bitboards[8] 
		| board.bitboards[9] | board.bitboards[10] | board.bitboards[11];

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
			if (board.bitboards[i] & (0x1uLL << pos))
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

Bitboard was_piece_moved(ChessBoard board, ChessPosition pos)
{
	return ((~START_POSITIONS | board.bitboards[12]) & (0x1uLL << pos));
}

ChessBoard apply_draw(ChessBoard board, ChessDraw draw)
{
	uint8_t i;
	ChessBoard new_board;

	for (i = 0; i < 13; i++)
	{
        new_board.bitboards[i] = board.bitboards[i];
	}

	apply_draw_to_bitboards(new_board.bitboards, draw);
	return new_board;

	/* TODO: check if the memory allocation actually works */
}

void apply_draw_to_bitboards(Bitboard* bitboards, ChessDraw draw)
{
    Bitboard old_pos, new_pos, white_mask, black_mask, targetColumn;
    uint8_t rooks_board_index, side_offset, drawing_board_index, taken_piece_bitboard_index, promotion_board_index;

    /* determine bitboard masks of the drawing piece's old and new position */
    old_pos = 0x1uLL <<  get_old_position(draw);
    new_pos = 0x1uLL << get_new_position(draw);

    /* determine the bitboard index of the drawing piece */
    side_offset = SIDE_OFFSET(get_drawing_side(draw));
    drawing_board_index = PIECE_OFFSET(get_drawing_piece_type(draw)) + side_offset;

    /* set was moved */
    if (get_is_first_move(draw) && (bitboards[drawing_board_index] & old_pos)) { bitboards[12] |= (old_pos | new_pos); }
    else if (get_is_first_move(draw)) { bitboards[12] &= ~(old_pos | new_pos); }

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
            targetColumn = COL_A << get_column(get_new_position(draw));
            bitboards[taken_piece_bitboard_index] ^=
                  (white_mask & targetColumn & ROW_5)  /* caught enemy white peasant */
                | (black_mask & targetColumn & ROW_4); /* caught enemy black peasant */
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