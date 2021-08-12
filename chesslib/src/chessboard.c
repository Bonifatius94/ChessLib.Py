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

Bitboard* create_empty_chessboard()
{
    return (Bitboard*)calloc(13, sizeof(Bitboard));
}

ChessPiece* create_empty_simple_chessboard()
{
    return (ChessPiece*)calloc(64, sizeof(ChessPiece));
}

void copy_board(const Bitboard orig[], Bitboard* target)
{
    /* info: this function assumes the target to be properly formatted */
    size_t i;
    for (i = 0; i < 13; i++) { target[i] = orig[i]; }
}

void copy_simple_board(const ChessPiece orig[], ChessPiece* target)
{
    /* info: this function assumes the target to be properly formatted */
    size_t i;
    for (i = 0; i < 64; i++) { target[i] = orig[i]; }
}

void create_board_from_piecesatpos(const ChessPieceAtPos pieces_at_pos[],
    size_t pieces_count, Bitboard* target)
{
    size_t i; uint8_t board_index;
    ChessPosition pos; ChessPiece piece;

    /* initialize the target with zeros */
    for (i = 0; i < 12; i++) { target[i] = 0; }

    /* assume all pieces as already moved */
    target[12] = 0xFFFFFFFFFFFFFFFFuLL;

    /* loop through the pieces@pos array */
    for (i = 0; i < pieces_count; i++)
    {
        /* determine the piece and position */
        piece = get_pieceatpos_piece(pieces_at_pos[i]);
        pos = get_pieceatpos_position(pieces_at_pos[i]);

        /* determine the board to apply the piece to */
        board_index = SIDE_OFFSET(get_piece_color(piece)) + PIECE_OFFSET(get_piece_type(piece));

        /* apply the piece to the bitboard */
        target[board_index] |= 0x1uLL << pos;

        /* apply was_moved state of the chess piece to the bitboard;
           the chess pieces are assumed to be already moved, so only
           flip the bit if the piece was not moved */
        target[12] ^= (((Bitboard)(get_was_piece_moved(piece) ^ 1)) << pos) & START_POSITIONS;
    }
}

Bitboard is_captured_at(const Bitboard board[], ChessPosition pos)
{
    Bitboard mask, all_pieces;

    mask = 0x1uLL << pos;

	/* combine all bitboards to one bitboard by bitwise OR */
    all_pieces = board[0] | board[1] | board[2] | board[3] | board[4] | board[5]
        | board[6] | board[7] | board[8] | board[9] | board[10] | board[11];

    return (all_pieces & mask);
}

ChessPiece get_piece_at(const Bitboard board[], ChessPosition pos)
{
    int i;
    ChessPiece piece = CHESS_PIECE_NULL;
    ChessPieceType type; ChessColor color;

    /* only create a chess piece if the board is captured at the given position */
    if (is_captured_at(board, pos))
    {
        /* initialize piece type and color with default values such that 
           the resulting chess piece would be equal to CHESS_PIECE_NULL */
        type = Invalid;
        color = White;

        /* loop through all bitboards */
        for (i = 0; i < 12; i++)
        {
            /* piece was found if the bitboard is set at the given position */
            if (board[i] & (0x1uLL << pos))
            {
                 type = (ChessPieceType)((i % 6) + 1);
                 color = (ChessColor)(i / 6);
                 break;
            }
        }

        /* create a chess piece instance with the given features */
        piece = create_piece(type, color, was_piece_moved(board, pos) == 0 ? 0 : 1);
    }

    return piece;
}

int was_piece_moved(const Bitboard board[], ChessPosition pos)
{
    /* evaluate the was_moved flag at the given position
       hint: non-start positions always indicate that the piece was moved */
    return ((~START_POSITIONS | board[12]) & (0x1uLL << pos)) > 0;
}

void apply_draw(Bitboard* bitboards, ChessDraw draw)
{
    /* info: this function is implemented using XOR-only operations, so applying
             the same draw once again will flip all bits back (= revert operation) */

    Bitboard old_pos, new_pos, white_mask, black_mask, target_column;
    uint8_t rooks_board_index, side_offset, drawing_board_index, 
            taken_piece_bitboard_index, promotion_board_index;

    /* determine bitboard masks of the drawing piece's old and new position */
    old_pos = 0x1uLL << get_old_position(draw);
    new_pos = 0x1uLL << get_new_position(draw);

    /* determine the bitboard index of the drawing piece */
    side_offset = SIDE_OFFSET(get_drawing_side(draw));
    drawing_board_index = PIECE_OFFSET(get_drawing_piece_type(draw)) + side_offset;

    /* set was moved */
    if (get_is_first_move(draw) && (bitboards[drawing_board_index] & old_pos)) {
        bitboards[12] ^= (old_pos | new_pos) & START_POSITIONS;
    } else if (get_is_first_move(draw)) { 
        bitboards[12] ^= (old_pos | new_pos) & START_POSITIONS;
    }

    /* move the drawing piece by flipping its' bits
       at the old and new position on the bitboard */
    bitboards[drawing_board_index] ^= old_pos | new_pos;

    /* handle rochade: move casteling rook accordingly,
       king will be moved by standard logic */
    if (get_draw_type(draw) == Rochade)
    {
        /* determine the rooks bitboard */
        rooks_board_index = PIECE_OFFSET(Rook) + side_offset;

        /* move the casteling rook by filpping bits at
           its' old and new position on the bitboard */
        bitboards[rooks_board_index] ^=
              ((new_pos & COL_C) << 1) | ((new_pos & COL_C) >> 2)  /* big rochade   */
            | ((new_pos & COL_G) << 1) | ((new_pos & COL_G) >> 1); /* small rochade */
    }

    /* handle catching draw: remove caught enemy piece accordingly */
    if (get_taken_piece_type(draw) != Invalid)
    {
        /* determine the taken piece's bitboard */
        taken_piece_bitboard_index = SIDE_OFFSET(OPPONENT(get_drawing_side(draw)))
            + PIECE_OFFSET(get_taken_piece_type(draw));

        /* handle en-passant: remove enemy peasant accordingly,
           drawing peasant will be moved by standard logic */
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
        /* handle normal catch: catch the enemy piece by
           flipping the bit at its' position on the bitboard */
        else { bitboards[taken_piece_bitboard_index] ^= new_pos; }
    }

    /* handle peasant promotion: wipe peasant and put the promoted piece */
    if (get_peasant_promotion_piece_type(draw) != Invalid)
    {
        /* remove the peasant at the new position */
        bitboards[drawing_board_index] ^= new_pos;

        /* put the promoted piece at the new position instead */
        promotion_board_index = side_offset +
            PIECE_OFFSET(get_peasant_promotion_piece_type(draw));
        bitboards[promotion_board_index] ^= new_pos;
    }
}

void from_simple_board(const ChessPiece simple_board[], Bitboard* target)
{
    /* info: this function assumes the target to be properly formatted */

    uint8_t i, pos, white_pos, black_pos;
    ChessPieceType piece_type; ChessColor color;
    Bitboard bitboard; int set_bit;

    /* assume pieces as already moved */
    Bitboard was_moved = 0xFFFFFFFFFFFFFFFFuL;

    /* loop through all bitboards */
    for (i = 0; i < 12; i++)
    {
        /* determine the chess piece type and color of the iteration */
        piece_type = (ChessPieceType)((i % 6) + 1);
        color = (ChessColor)(i / 6);

        /* init empty bitboard */
        bitboard = 0;

        /* loop through all positions */
        for (pos = 0; pos < 64; pos++)
        {
            /* set piece bit if the position is captured */
            set_bit = simple_board[pos] != CHESS_PIECE_NULL 
                && get_piece_type(simple_board[pos]) == piece_type
                && get_piece_color(simple_board[pos]) == color;
            bitboard |= set_bit ? 0x1uL << pos : 0x0uL;
        }

        /* apply converted bitboard */
        target[i] = bitboard;
    }

    /* init was moved bitboard */
    for (i = 0; i < 16; i++)
    {
        white_pos = i;
        black_pos = (uint8_t)(i + 48);

        /* explicitly set was moved to 0 only for unmoved pieces */
        was_moved ^= simple_board[white_pos] != CHESS_PIECE_NULL && 
            !get_was_piece_moved(simple_board[white_pos]) ? 0x1uL << white_pos : 0x0uL;
        was_moved ^= simple_board[black_pos] != CHESS_PIECE_NULL && 
            !get_was_piece_moved(simple_board[black_pos]) ? 0x1uL << black_pos : 0x0uL;
    }

    /* apply converted bitboard */
    target[12] = was_moved;
}

void to_simple_board(const Bitboard board[], ChessPiece* target)
{
    /* info: this function assumes the target to be properly formatted */

    uint8_t i, pos; ChessPieceType piece_type;
    ChessColor color; Bitboard bitboard;

    /* initialize values in target array with zeros */
    for (pos = 0; pos < 64; pos++) { target[pos] = 0; }

    /* loop through all bitboards */
    for (i = 0; i < 12; i++)
    {
        /* determine the chess piece type and color of the iteration */
        piece_type = (ChessPieceType)((i % 6) + 1);
        color = (ChessColor)(i / 6);

        /* cache bitboard for shifting bitwise */
        bitboard = board[i];

        /* loop through all positions */
        for (pos = 0; pos < 64; pos++)
        {
            /* write piece to array if there is one */
            target[pos] = (bitboard & 0x1) > 0
                ? create_piece(piece_type, color, was_piece_moved(board, pos))
                : target[pos];

            /* shift bitboard */
            bitboard >>= 1;
        }
    }
}

void compress_pieces_array(const ChessPiece pieces[], uint8_t* compr_bytes)
{
    ChessPosition pos;
    const uint8_t mask = 0xF8u;
    uint8_t offset, index, piece_bits;

    /* loop through all positions */
    for (pos = 0; pos < 64; pos++)
    {
        /* get chess piece from array by position */
        piece_bits = pieces[pos] << 3;

        /* determine the output byte's index and bit offset */
        index = ((int)pos * 5) / 8;
        offset = ((int)pos * 5) % 8;

        /* write leading bits to byte at the piece's position */
        compr_bytes[index] |= (piece_bits & mask) >> offset;

        /* write overlapping bits to the next byte (only if needed) */
        if (offset > 3) { compr_bytes[index + 1] |= (uint8_t)((piece_bits & mask) << (8 - offset)); }
    }
}

uint8_t get_bits_at(const uint8_t data_bytes[], size_t arr_size, int bit_index, int length)
{
    uint8_t upper, lower; uint16_t combined; int bitOffset;

    /* load data bytes into cache */
    bitOffset = bit_index % 8;
    upper = data_bytes[bit_index / 8];
    lower = ((size_t)bit_index / 8 + 1 < arr_size) ? data_bytes[bit_index / 8 + 1] : (uint8_t)0x00;
    combined = (uint16_t)((uint16_t)upper << 8) | (lower & 0xFF);

    /* cut the desired bits from the combined bytes */
    return (uint8_t)(((uint16_t)(combined << bitOffset)) >> (8 + (8 - length)));
}

void uncompress_pieces_array(const uint8_t compr_bytes[], ChessPiece* out_pieces)
{
    ChessPosition pos;
    uint8_t piece_bits;

    /* loop through all positions */
    for (pos = 0; pos < 64; pos++)
    {
        piece_bits = get_bits_at(compr_bytes, 40, pos * 5, 5);
        out_pieces[pos] = piece_bits;
    }
}

/* ====================================================
             C H E S S    P O S I T I O N S
                 O N    B I T B O A R D
   ==================================================== */

Bitboard get_captured_fields(const Bitboard bitboards[], ChessColor side)
{
    uint8_t offset = SIDE_OFFSET(side);

    return bitboards[offset] | bitboards[offset + 1] | bitboards[offset + 2]
        | bitboards[offset + 3] | bitboards[offset + 4] | bitboards[offset + 5];
}

void get_board_positions(Bitboard bitboard, ChessPosition* out_positions, size_t* out_length)
{
    uint8_t pos;
    *out_length = 0;

    /* loop through all bits of the board */
    for (pos = 0; pos < 64; pos++)
    {
        if ((bitboard & 0x1uLL << pos)) { out_positions[(*out_length)++] = (ChessPosition)pos; }
    }
}

/**************************************************************************************************
   this returns the index of the highest bit set on the given bitboard.
   if the given bitboard has multiple bits set, only the position of the highest bit is returned.
   info: the result is mathematically equal to floor(log2(x))
 **************************************************************************************************/
ChessPosition get_board_position(Bitboard bitboard)
{
    /* code was taken from https://stackoverflow.com/questions/11376288/fast-computing-of-log2-for-64-bit-integers */

#ifdef __GNUC__
    /* use built-in leading zeros function for GCC Linux build
       (this compiles to the very fast 'bsr' instruction on x86 AMD processors) */
    return (ChessPosition)((unsigned)(8 * sizeof(unsigned long long) - __builtin_clzll(bitboard) - 1));
#else
    /* use abstract DeBruijn algorithm with table lookup */
    /* TODO: think of implementing this as assembler code */

    const uint8_t tab64[64] = {
        63,  0, 58,  1, 59, 47, 53,  2,
        60, 39, 48, 27, 54, 33, 42,  3,
        61, 51, 37, 40, 49, 18, 28, 20,
        55, 30, 34, 11, 43, 14, 22,  4,
        62, 57, 46, 52, 38, 26, 32, 41,
        50, 36, 17, 19, 29, 10, 13, 21,
        56, 45, 25, 31, 35, 16,  9, 12,
        44, 24, 15,  8, 23,  7,  6,  5
    };

    bitboard |= bitboard >> 1;
    bitboard |= bitboard >> 2;
    bitboard |= bitboard >> 4;
    bitboard |= bitboard >> 8;
    bitboard |= bitboard >> 16;
    bitboard |= bitboard >> 32;

    return (ChessPosition)tab64[((Bitboard)((bitboard - (bitboard >> 1)) * 0x07EDD5E59A4E28C2uLL)) >> 58];
#endif
}
