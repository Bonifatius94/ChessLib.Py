#include "chessdrawgen.h"

/* TODO: think about implementing this without vector<> struct in classic C manner with double array pointer ** and size_t* array size, etc. */

/* ====================================================
            H E L P E R    F U N C T I O N S
   ==================================================== */

/*vector<ChessDraw> get_draws(Bitboard bitboards[], ChessColor side, ChessPieceType type, ChessDraw last_draw);
vector<ChessDraw> eliminate_draws_into_check(ChessBoard board, vector<ChessDraw> draws, ChessColor drawing_side);*/

void get_draws(ChessDraw** out_draws, size_t* out_length, const Bitboard bitboards[], ChessColor side, ChessPieceType type, ChessDraw last_draw);
void eliminate_draws_into_check(ChessDraw** out_draws, size_t* out_length, ChessBoard board, const ChessDraw draws[], ChessColor drawing_side);

Bitboard get_king_draw_positions(const Bitboard bitboards[], ChessColor side, int rochade);
Bitboard get_standard_king_draw_positions(const Bitboard bitboards[], ChessColor side);
Bitboard get_rochade_king_draw_positions(const Bitboard bitboards[], ChessColor side);
Bitboard get_queen_draw_positions(const Bitboard bitboards[], ChessColor side, Bitboard drawing_pieces_filter);
Bitboard get_rook_draw_positions(const Bitboard bitboards[], ChessColor side, Bitboard drawing_pieces_filter, uint8_t piece_offset);
Bitboard get_bishop_draw_positions(const Bitboard bitboards[], ChessColor side, Bitboard drawing_pieces_filter, uint8_t piece_offset);
Bitboard get_knight_draw_positions(const Bitboard bitboards[], ChessColor side, Bitboard drawing_pieces_filter);
Bitboard get_peasant_draw_positions(const Bitboard bitboards[], ChessColor side, ChessDraw last_draw, Bitboard drawing_pieces_filter);

void get_positions(Bitboard bitboard, ChessPosition** out_positions, size_t* out_length);
ChessPosition get_position(Bitboard bitboard);
void concat_draws(ChessDraw** draws, size_t* count, const ChessDraw append[], size_t append_length);

Bitboard get_capturable_fields(const Bitboard bitboards[], ChessColor side, ChessDraw last_draw);
Bitboard get_captured_fields(const Bitboard bitboards[], ChessColor side);

/* ====================================================
               D R A W - G E N    M A I N
   ==================================================== */

void get_all_draws(ChessDraw** out_draws, size_t* out_length, ChessBoard board, ChessColor drawing_side, ChessDraw last_draw, int analyze_draw_into_check)
{
    uint8_t side_offset;
    ChessColor opponent;

    ChessDraw* temp_draws;
    size_t temp_len;

     /* determine the drawing side */
     side_offset = SIDE_OFFSET(drawing_side);
     opponent = OPPONENT(drawing_side);

     /* compute the draws for the pieces of each type (for non-king pieces, check first if the bitboard actually contains pieces) */
     king_draws    = get_draws(&temp_draws, &temp_len, board.bitboards, drawing_side, King, last_draw);
     queen_draws   = (board.bitboards[side_offset + PIECE_OFFSET(Queen)]   != 0x0uLL) ? get_draws(&temp_draws, &temp_len, board.bitboards, drawing_side, Queen, last_draw)   : NULL;
     rook_draws    = (board.bitboards[side_offset + PIECE_OFFSET(Rook)]    != 0x0uLL) ? get_draws(&temp_draws, &temp_len, board.bitboards, drawing_side, Rook, last_draw)    : NULL;
     bishop_draws  = (board.bitboards[side_offset + PIECE_OFFSET(Bishop)]  != 0x0uLL) ? get_draws(&temp_draws, &temp_len, board.bitboards, drawing_side, Bishop, last_draw)  : NULL;
     knight_draws  = (board.bitboards[side_offset + PIECE_OFFSET(Knight)]  != 0x0uLL) ? get_draws(&temp_draws, &temp_len, board.bitboards, drawing_side, Knight, last_draw)  : NULL;
     peasant_draws = (board.bitboards[side_offset + PIECE_OFFSET(Peasant)] != 0x0uLL) ? get_draws(&temp_draws, &temp_len, board.bitboards, drawing_side, Peasant, last_draw) : NULL;
     
     /* concatenate the draws as one vector */
     draws = vector<ChessDraw>(king_draws.size() + queen_draws.size() + rook_draws.size() + bishop_draws.size() + knight_draws.size() + peasant_draws.size());
     draws.insert(draws.end(), king_draws.begin(), king_draws.end());
     draws.insert(draws.end(), queen_draws.begin(), queen_draws.end());
     draws.insert(draws.end(), rook_draws.begin(), rook_draws.end());
     draws.insert(draws.end(), bishop_draws.begin(), bishop_draws.end());
     draws.insert(draws.end(), knight_draws.begin(), knight_draws.end());
     draws.insert(draws.end(), peasant_draws.begin(), peasant_draws.end());

     /* if flag is active, filter only draws that do not cause draws into check */
     return analyze_draw_into_check ? eliminate_draws_into_check(board, draws, drawing_side) : draws;
}

void eliminate_draws_into_check(ChessDraw** out_draws, size_t* length, ChessBoard board, ChessDraw draws_to_validate[], ChessColor drawing_side)
{
    uint8_t side_offset;
    size_t i, legal_draws_count;
    Bitboard sim_bitboards[13], king_mask, enemy_capturable_fields;
    ChessColor opponent;

    /* make a working copy of all local bitboards */
    for (i = 0; i < 13; i++) { sim_bitboards[i] = board.bitboards[i]; }

    /* init legal draws count with the amount of all draws (unvalidated) */
    legal_draws_count = sizeof(draws_to_validate) / sizeof(ChessDraw);
    side_offset = SIDE_OFFSET(drawing_side);
    opponent = OPPONENT(drawing_side);

    /* loop through draws and simulate each draw */
    for (i = 0; i < legal_draws_count; i++)
    {
        /* simulate the draw */
        apply_draw_to_bitboards(sim_bitboards, draws_to_validate[i]);
        king_mask = sim_bitboards[side_offset];

        /* calculate enemy answer draws (only fields that could be captured as one bitboard) */
        enemy_capturable_fields = get_capturable_fields(sim_bitboards, opponent, draws_to_validate[i]);

        /*    old code
           ========================================
        Bitboard enemy_capturable_fields =
              get_king_draw_positions(sim_bitboards, opponent, false)
            | get_queen_draw_positions(sim_bitboards, opponent)
            | getRookDrawBitboards(sim_bitboards, opponent)
            | getBishopDrawBitboards(sim_bitboards, opponent)
            | getKnightDrawBitboards(sim_bitboards, opponent)
            | getPeasantDrawBitboards(sim_bitboards, opponent); */

        /* revert the simulated draw (flip the bits back) */
        apply_draw_to_bitboards(sim_bitboards, draws_to_validate[i]);

        /* check if one of those draws would catch the allied king (bitwise AND) -> draw-into-check */
        if ((king_mask & enemy_capturable_fields) > 0)
        {
            /* overwrite the illegal draw with the last unevaluated draw in the array */
            draws_to_validate[i--] = draws_to_validate[--legal_draws_count];
        }
    }

    /* remove illegal draws */
    return vector<ChessDraw>(draws_to_validate.begin(), draws_to_validate.begin() + legal_draws_count);
}

void get_draws(ChessDraw** out_draws, size_t* out_length, const Bitboard bitboards[], ChessColor side, ChessPieceType type, ChessDraw last_draw)
{
    uint8_t index, piece_type;
    size_t count = 0, i, j, drawing_pieces_len;
    ChessPosition *drawing_pieces, *capturable_positions;
    ChessDraw *draws;

    ChessPosition pos;
    Bitboard filter, draw_bitboard;
    ChessBoard board;
    int contains_peasant_promotion;

    /* get drawing pieces */
    index = SIDE_OFFSET(side) + PIECE_OFFSET(type);
    get_positions(bitboards[index], &drawing_pieces, &drawing_pieces_len);

    /* init draws result set (max. draws) */
    *draws = (ChessDraw*)malloc(drawing_pieces.size() * 28);

    /* loop through drawing pieces */
    for (i = 0; i < drawing_pieces.size(); i++)
    {
        pos = drawing_pieces[i];

        /* only set the drawing piece to the bitboard, wipe all others */
        filter = 0x1uLL << pos;
        draw_bitboard = 0x0uLL;

        /* compute the chess piece's capturable positions as bitboard */
        switch (type)
        {
            case King:    draw_bitboard = get_king_draw_positions(bitboards, side, 1);                              break;
            case Queen:   draw_bitboard = get_queen_draw_positions(bitboards, side, filter);                        break;
            case Rook:    draw_bitboard = get_rook_draw_positions(bitboards, side, filter, PIECE_OFFSET(Rook));     break;
            case Bishop:  draw_bitboard = get_bishop_draw_positions(bitboards, side, filter, PIECE_OFFSET(Bishop)); break;
            case Knight:  draw_bitboard = get_knight_draw_positions(bitboards, side, filter);                       break;
            case Peasant: draw_bitboard = get_peasant_draw_positions(bitboards, side, last_draw, filter);           break;
            /*default: throw new ArgumentException("Invalid chess piece type detected!");*/
        }

        /* extract all capturable positions from the draws bitboard */
        capturable_positions = get_positions(draw_bitboard);

        /* check for peasant promotion */
        contains_peasant_promotion = (type == Peasant && ((side == White && (draw_bitboard & ROW_8)) || (side == Black && (draw_bitboard & ROW_1))));
        board = create_board(bitboards);

        /* convert the positions into chess draws */
        if (contains_peasant_promotion)
        {
            /* peasant will advance to level 8, all draws need to be peasant promotions */
            for (j = 0; j < capturable_positions.size(); j++)
            {
                /* add types that the piece can promote to (queen, rook, bishop, knight) */
                for (piece_type = 2; piece_type < 6; piece_type++) { draws[count++] = create_draw(board, drawing_pieces[i], capturable_positions[j], (ChessPieceType)piece_type); }
            }
        }
        else { for (j = 0; j < capturable_positions.size(); j++) { draws[count++] = create_draw(board, drawing_pieces[i], capturable_positions[j], Invalid); } }
    }

    /* cut trailing empty draws from draws array and return only the actual draws */
    out_draws = vector<ChessDraw>(count);
    out_draws.insert(out_draws.begin(), draws.begin(), draws.begin() + count);

    /* TODO: free memory that is no longer required */

    return out_draws;
}

Bitboard get_capturable_fields(const Bitboard bitboards[], ChessColor side, ChessDraw last_draw)
{
    Bitboard capturable_fields;

    /* combine the capturable fields of all pieces of the given side by bitwise OR */
    capturable_fields =
          get_king_draw_positions(bitboards, side, 0)
        | get_queen_draw_positions(bitboards, side, 0xFFFFFFFFFFFFFFFFuLL)
        | get_rook_draw_positions(bitboards, side, 0xFFFFFFFFFFFFFFFFuLL, PIECE_OFFSET(Rook))
        | get_bishop_draw_positions(bitboards, side, 0xFFFFFFFFFFFFFFFFuLL, PIECE_OFFSET(Bishop))
        | get_knight_draw_positions(bitboards, side, 0xFFFFFFFFFFFFFFFFuLL)
        | get_peasant_draw_positions(bitboards, side, last_draw, 0xFFFFFFFFFFFFFFFFuLL);

    return capturable_fields;
}

/* ====================================================
                K I N G    D R A W - G E N
   ==================================================== */

Bitboard get_king_draw_positions(const Bitboard bitboards[], ChessColor side, int rochade)
{
    Bitboard standard_draws, rochade_draws;

    // determine standard and rochade draws
    standard_draws = get_standard_king_draw_positions(bitboards, side);
    rochade_draws = rochade ? get_rochade_king_draw_positions(bitboards, side) : 0x0uLL;

    return standard_draws | rochade_draws;
}

Bitboard get_standard_king_draw_positions(const Bitboard bitboards[], ChessColor side)
{
    Bitboard bitboard, allied_pieces, standard_draws;

    /* get the king bitboard */
    bitboard = bitboards[SIDE_OFFSET(side)];

    /* determine allied pieces to eliminate blocked draws */
    allied_pieces = get_captured_fields(bitboards, side);

    /* compute all possible draws using bit-shift, moreover eliminate illegal overflow draws
       info: the top/bottom comments are related to white-side perspective */
    standard_draws =
          ((bitboard << 7) & ~(ROW_1 | COL_H | allied_pieces))  /* top left     */
        | ((bitboard << 8) & ~(ROW_1 |         allied_pieces))  /* top mid      */
        | ((bitboard << 9) & ~(ROW_1 | COL_A | allied_pieces))  /* top right    */
        | ((bitboard >> 1) & ~(COL_H |         allied_pieces))  /* side left    */
        | ((bitboard << 1) & ~(COL_A |         allied_pieces))  /* side right   */
        | ((bitboard >> 9) & ~(ROW_8 | COL_H | allied_pieces))  /* bottom left  */
        | ((bitboard >> 8) & ~(ROW_8 |         allied_pieces))  /* bottom mid   */
        | ((bitboard >> 7) & ~(ROW_8 | COL_A | allied_pieces)); /* bottom right */

    // TODO: cache draws to save computation

    return standard_draws;
}

Bitboard get_rochade_king_draw_positions(const Bitboard bitboards[], ChessColor side)
{
    uint8_t offset;
    Bitboard draws, king, rooks, was_moved, enemy_capturable_fields, free_king_passage;

    /* get the king and rook bitboard */
    offset = SIDE_OFFSET(side);
    king = bitboards[offset];
    rooks = bitboards[offset + PIECE_OFFSET(Rook)];
    was_moved = bitboards[12];

    /* enemy capturable positions (for validation) */
    enemy_capturable_fields = get_capturable_fields(bitboards, OPPONENT(side), DRAW_NULL);
    free_king_passage =
          ~((FIELD_C1 & enemy_capturable_fields) & ((FIELD_D1 & enemy_capturable_fields) >> 1) & ((FIELD_E1 & enemy_capturable_fields) >> 2))  /* white big rochade   */
        | ~((FIELD_G1 & enemy_capturable_fields) & ((FIELD_F1 & enemy_capturable_fields) << 1) & ((FIELD_E1 & enemy_capturable_fields) << 2))  /* white small rochade */
        | ~((FIELD_C8 & enemy_capturable_fields) & ((FIELD_D8 & enemy_capturable_fields) >> 1) & ((FIELD_E8 & enemy_capturable_fields) >> 2))  /* black big rochade   */
        | ~((FIELD_G8 & enemy_capturable_fields) & ((FIELD_F8 & enemy_capturable_fields) << 1) & ((FIELD_E8 & enemy_capturable_fields) << 2)); /* black small rochade */

    /* get rochade draws (king and rook not moved, king passage free) */
    draws =
          (((king & FIELD_E1 & ~was_moved) >> 2) & ((rooks & FIELD_A1 & ~was_moved) << 3) & free_king_passage)  /* white big rochade   */
        | (((king & FIELD_E1 & ~was_moved) << 2) & ((rooks & FIELD_H1 & ~was_moved) >> 2) & free_king_passage)  /* white small rochade */
        | (((king & FIELD_E8 & ~was_moved) >> 2) & ((rooks & FIELD_A8 & ~was_moved) << 3) & free_king_passage)  /* black big rochade   */
        | (((king & FIELD_E8 & ~was_moved) >> 2) & ((rooks & FIELD_H8 & ~was_moved) >> 2) & free_king_passage); /* black small rochade */

    /* TODO: cache draws to save computation */

    return draws;
}

/* ====================================================
               Q U E E N    D R A W - G E N
   ==================================================== */

Bitboard get_queen_draw_positions(const Bitboard bitboards[], ChessColor side, Bitboard drawing_pieces_filter)
{
    return get_rook_draw_positions(bitboards, side, drawing_pieces_filter, 1) | get_bishop_draw_positions(bitboards, side, drawing_pieces_filter, 1);
}

/* ====================================================
                R O O K    D R A W - G E N
   ==================================================== */

Bitboard get_rook_draw_positions(const Bitboard bitboards[], ChessColor side, Bitboard drawing_pieces_filter, uint8_t piece_offset)
{
    uint8_t i;
    Bitboard draws = 0uLL, bitboard, enemy_pieces, allied_pieces, b_rooks, l_rooks, r_rooks, t_rooks;

    /* get the bitboard */
    bitboard = bitboards[SIDE_OFFSET(side) + piece_offset] & drawing_pieces_filter;

    /* determine allied and enemy pieces (for collision / catch handling) */
    enemy_pieces = get_captured_fields(bitboards, OPPONENT(side));
    allied_pieces = get_captured_fields(bitboards, side);

    /* init empty draws bitboards, separated by field color */
    b_rooks = bitboard;
    l_rooks = bitboard;
    r_rooks = bitboard;
    t_rooks = bitboard;

    /* compute draws (try to apply 1-7 shifts in each direction) */
    for (i = 1; i < 8; i++)
    {
        /* simulate the computing of all draws:
           if there would be one or more overflows / collisions with allied pieces, remove certain rooks 
           from the rooks bitboard, so the overflow won't occur on the real draw computation afterwards */
        b_rooks ^= ((b_rooks >> (i * 8)) & (ROW_8 | allied_pieces)) << (i * 8); /* bottom */
        l_rooks ^= ((l_rooks >> (i * 1)) & (COL_H | allied_pieces)) << (i * 1); /* left   */
        r_rooks ^= ((r_rooks << (i * 1)) & (COL_A | allied_pieces)) >> (i * 1); /* right  */
        t_rooks ^= ((t_rooks << (i * 8)) & (ROW_1 | allied_pieces)) >> (i * 8); /* top    */

        /* compute all legal draws and apply them to the result bitboard */
        draws |= b_rooks >> (i * 8) | l_rooks >> (i * 1) | r_rooks << (i * 1) | t_rooks << (i * 8);

        /* handle catches the same way as overflow / collision detection (this has to be done afterwards 
           as the catches are legal draws that need to occur onto the result bitboard) */
        b_rooks ^= ((b_rooks >> (i * 8)) & enemy_pieces) << (i * 8); /* bottom */
        l_rooks ^= ((l_rooks >> (i * 1)) & enemy_pieces) << (i * 1); /* left   */
        r_rooks ^= ((r_rooks << (i * 1)) & enemy_pieces) >> (i * 1); /* right  */
        t_rooks ^= ((t_rooks << (i * 8)) & enemy_pieces) >> (i * 8); /* top    */
    }

    /* TODO: implement hyperbola quintessence */

    return draws;
}

/* ====================================================
             B I S H O P    D R A W - G E N
   ==================================================== */

Bitboard get_bishop_draw_positions(const Bitboard bitboards[], ChessColor side, Bitboard drawing_pieces_filter, uint8_t piece_pffset)
{
    uint8_t i;
    Bitboard draws = 0uLL, bitboard, enemy_pieces, allied_pieces, br_bishops, bl_bishops, tr_bishops, tl_bishops;

    /* get the bitboard */
    bitboard = bitboards[SIDE_OFFSET(side) + piece_pffset] & drawing_pieces_filter;

    /* determine allied and enemy pieces (for collision / catch handling) */
    enemy_pieces = get_captured_fields(bitboards, OPPONENT(side));
    allied_pieces = get_captured_fields(bitboards, side);

    /* init empty draws bitboards, separated by field color */
    br_bishops = bitboard;
    bl_bishops = bitboard;
    tr_bishops = bitboard;
    tl_bishops = bitboard;

    /* compute draws (try to apply 1-7 shifts in each direction) */
    for (i = 1; i < 8; i++)
    {
        /* simulate the computing of all draws:
           if there would be one or more overflows / collisions with allied pieces, remove certain bishops 
           from the bishops bitboard, so the overflow won't occur on the real draw computation afterwards */
        br_bishops ^= ((br_bishops >> (i * 7)) & (ROW_8 | COL_A | allied_pieces)) << (i * 7); /* bottom right */
        bl_bishops ^= ((bl_bishops >> (i * 9)) & (ROW_8 | COL_H | allied_pieces)) << (i * 9); /* bottom left  */
        tr_bishops ^= ((tr_bishops << (i * 9)) & (ROW_1 | COL_A | allied_pieces)) >> (i * 9); /* top right    */
        tl_bishops ^= ((tl_bishops << (i * 7)) & (ROW_1 | COL_H | allied_pieces)) >> (i * 7); /* top left     */

        /* compute all legal draws and apply them to the result bitboard */
        draws |= br_bishops >> (i * 7) | bl_bishops >> (i * 9) | tr_bishops << (i * 9) | tl_bishops << (i * 7);

        /* handle catches the same way as overflow / collision detection (this has to be done afterwards 
           as the catches are legal draws that need to occur onto the result bitboard) */
        br_bishops ^= ((br_bishops >> (i * 7)) & enemy_pieces) << (i * 7); /* bottom right */
        bl_bishops ^= ((bl_bishops >> (i * 9)) & enemy_pieces) << (i * 9); /* bottom left  */
        tr_bishops ^= ((tr_bishops << (i * 9)) & enemy_pieces) >> (i * 9); /* top right    */
        tl_bishops ^= ((tl_bishops << (i * 7)) & enemy_pieces) >> (i * 7); /* top left     */
    }

    /* TODO: implement hyperbola quintessence */

    return draws;
}

/* ====================================================
             K N I G H T    D R A W - G E N
   ==================================================== */

Bitboard get_knight_draw_positions(const Bitboard bitboards[], ChessColor side, Bitboard drawing_pieces_filter = 0xFFFFFFFFFFFFFFFFuLL)
{
    Bitboard bitboard, allied_pieces, draws;

    /* get bishops bitboard */
    bitboard = bitboards[SIDE_OFFSET(side) + PIECE_OFFSET(Knight)] & drawing_pieces_filter;

    /* determine allied pieces to eliminate blocked draws */
    allied_pieces = get_captured_fields(bitboards, side);

    /* compute all possible draws using bit-shift, moreover eliminate illegal overflow draws */
    draws =
          ((bitboard <<  6) & ~(ROW_1 | COL_H | COL_G | allied_pieces))  /* top left  (1-2)    */
        | ((bitboard << 10) & ~(ROW_1 | COL_A | COL_B | allied_pieces))  /* top right (1-2)    */
        | ((bitboard << 15) & ~(ROW_1 | COL_H | ROW_2 | allied_pieces))  /* top left  (2-1)    */
        | ((bitboard << 17) & ~(ROW_1 | COL_A | ROW_2 | allied_pieces))  /* top right (2-1)    */
        | ((bitboard >> 10) & ~(ROW_8 | COL_H | COL_G | allied_pieces))  /* bottom left  (1-2) */
        | ((bitboard >>  6) & ~(ROW_8 | COL_A | COL_B | allied_pieces))  /* bottom right (1-2) */
        | ((bitboard >> 17) & ~(ROW_8 | COL_H | ROW_7 | allied_pieces))  /* bottom left  (2-1) */
        | ((bitboard >> 15) & ~(ROW_8 | COL_A | ROW_7 | allied_pieces)); /* bottom right (2-1) */

    /* TODO: cache draws to save computation */

    return draws;
}

/* ====================================================
             P E A S A N T    D R A W - G E N
   ==================================================== */

Bitboard get_peasant_draw_positions(const Bitboard bitboards[], ChessColor side, ChessDraw last_draw, Bitboard drawing_pieces_filter = 0xFFFFFFFFFFFFFFFFuLL)
{
    Bitboard draws = 0x0uLL, bitboard, allied_pieces, enemy_pieces, blocking_pieces, enemy_peasants, 
        was_moved_mask, white_mask, black_mask, last_draw_new_pos, last_draw_old_pos;

    /* get peasants bitboard */
    bitboard = bitboards[SIDE_OFFSET(side) + PIECE_OFFSET(Peasant)] & drawing_pieces_filter;

    /* get all fields captured by enemy pieces as bitboard */
    allied_pieces = get_captured_fields(bitboards, side);
    enemy_pieces = get_captured_fields(bitboards, OPPONENT(side));
    blocking_pieces = allied_pieces | enemy_pieces;
    enemy_peasants = bitboards[SIDE_OFFSET(OPPONENT(side)) + PIECE_OFFSET(Peasant)];
    was_moved_mask = bitboards[12];

    /* initialize white and black masks (-> calculate draws for both sides, but nullify draws of the wrong side using the mask) */
    white_mask = WHITE_MASK(side);
    black_mask = ~white_mask;

    /* get one-foreward draws */
    draws |= 
          (white_mask & (bitboard << 8) & ~blocking_pieces)
        | (black_mask & (bitboard >> 8) & ~blocking_pieces);

    /* get two-foreward draws */
    draws |= 
          (white_mask & ((((bitboard & ROW_2 & ~was_moved_mask) << 8) & ~blocking_pieces) << 8) & ~blocking_pieces)
        | (black_mask & ((((bitboard & ROW_7 & ~was_moved_mask) >> 8) & ~blocking_pieces) >> 8) & ~blocking_pieces);

    /* handle en-passant (in case of en-passant, put an extra peasant that gets caught by the standard catch logic) */
    last_draw_new_pos = 0x1uLL << (last_draw != DRAW_NULL ? get_new_position(last_draw) : -1);
    last_draw_old_pos = 0x1uLL << (last_draw != DRAW_NULL ? get_old_position(last_draw) : -1);
    bitboard |=
          (white_mask & ((last_draw_new_pos & enemy_peasants) >> 8) & ((ROW_2 & last_draw_old_pos) << 8))
        | (black_mask & ((last_draw_new_pos & enemy_peasants) << 8) & ((ROW_2 & last_draw_old_pos) >> 8));

    /* get right / left catch draws */
    draws |= 
          (white_mask & ((((bitboard & ~COL_H) << 9) & enemy_pieces) | (((bitboard & ~COL_A) << 7) & enemy_pieces)))
        | (black_mask & ((((bitboard & ~COL_A) >> 9) & enemy_pieces) | (((bitboard & ~COL_H) >> 7) & enemy_pieces)));

    return draws;
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

void get_positions(Bitboard bitboard, ChessPosition** out_positions, size_t* out_length)
{
    uint8_t pos;
    size_t count = 0, i;

    /* init position cache for worst-case */
    ChessPosition temp_pos[28];
     
    /* loop through all bits of the board */
    for (pos = 0; pos < 64; pos++)
    {
        if ((bitboard & 0x1uLL << pos)) { temp_pos[count++] = (ChessPosition)pos; }
    }

    /* copy the positions to the results array (without empty entries) */
    *out_length = count;
    *out_positions = (ChessPosition*)malloc(count * sizeof(ChessPosition));
    /* TODO: check if the allocation is correct */

    for (i = 0; i < count; i++) { (*out_positions)[i] = temp_pos[i]; }
}

/**************************************************************************************************
   this returns the index of the highest bit set on the given bitboard.
   if the given bitboard has multiple bits set, only the position of the highest bit is returned.
   info: the result is mathematically equal to floor(log2(x)) 
 **************************************************************************************************/
ChessPosition get_position(Bitboard bitboard)
{
    /* code was taken from https://stackoverflow.com/questions/11376288/fast-computing-of-log2-for-64-bit-integers */

#ifdef __GNUC__
    /* use built-in leading zeros function for GCC Linux build (this compiles to the very fast 'bsr' instruction on x86 AMD processors) */
    return (ChessPosition)((unsigned)(8 * sizeof(unsigned long long) - __builtin_clzll((X)) - 1))
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

    return (ChessPosition)tab64[((Bitboard)((bitboard - (bitboard >> 1)) * 0x07EDD5E59A4E28C2)) >> 58];
#endif
}

/* ====================================================
                A R R A Y    C O N C A T
   ==================================================== */

void concat_draws(ChessDraw** draws, size_t* count, const ChessDraw append[], size_t append_length)
{
    size_t i;
    ChessDraw* old_draws;

    /* exit if there is nothing to be appended */
    if (append == NULL) { return; }

    /* extend memory by reallocation to fit the draws to be appended */
    old_draws = *draws;
    
    *draws = (ChessDraw*)realloc(old_draws, *count * sizeof(ChessDraw));
    /* TODO: check if the allocation is correct */
    
    /* copy the draws to the output array */
    for (i = 0; i < append_length; i++) { (*draws)[*count + i] = append[i]; }
    *count += append_length;
}