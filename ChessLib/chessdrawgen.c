#include "chessdrawgen.h"

size_t get_draws();
size_t concat_draws(ChessDraw** out_draws, ChessDraw a[], ChessDraw b[]);

 size_t get_possible_draws(ChessDraw** out_draws, ChessBitboard board, ChessColor drawing_side, ChessDraw last_draw, int analyze_draw_into_check)
{
     // determine the drawing side
     uint8_t sideOffset = SIDE_OFFSET(drawing_side);
     ChessColor opponent = OPPONENT(drawing_side);

     // compute the draws for the pieces of each type (for non-king pieces, check first if the bitboard actually contains pieces)
     size_t draws_count = 0;

     // TODO: init out_draws with size of the hypothetical max. draws amount (king=10, queen=28, rook=14, bishop=14, knight=8, peasant=4)
     ChessDraw* temp_draws = malloc(170 * sizeof(ChessDraw));
     // or alternatively use resize array inside get_draws to append the draws properly

     draws_count += getDraws(board.bitboards, drawing_side, King, last_draw);
     draws_count += (board.bitboards[sideOffset + PIECE_OFFSET(Queen)]   != 0x0uL) ? get_draws((out_draws + draws_count), board.bitboards, drawing_side, Queen, last_draw)   : 0;
     draws_count += (board.bitboards[sideOffset + PIECE_OFFSET(Rook)]    != 0x0uL) ? get_draws((out_draws + draws_count), board.bitboards, drawing_side, Rook, last_draw)    : 0;
     draws_count += (board.bitboards[sideOffset + PIECE_OFFSET(Bishop)]  != 0x0uL) ? get_draws((out_draws + draws_count), board.bitboards, drawing_side, Bishop, last_draw)  : 0;
     draws_count += (board.bitboards[sideOffset + PIECE_OFFSET(Knight)]  != 0x0uL) ? get_draws((out_draws + draws_count), board.bitboards, drawing_side, Knight, last_draw)  : 0;
     draws_count += (board.bitboards[sideOffset + PIECE_OFFSET(Peasant)] != 0x0uL) ? get_draws((out_draws + draws_count), board.bitboards, drawing_side, Peasant, last_draw) : 0;

     // if flag is active, filter only draws that do not cause draws into check
     if (analyze_draw_into_check)
     {
         // make a working copy of all local bitboards
         uint64_t simBitboards[13];
         for (int i = 0; i < 13; i++) { simBitboards[i] = board.bitboards[i]; }

         // init legal draws count with the amount of all draws (unvalidated)
         size_t legalDrawsCount = sizeof(draws) / sizeof(ChessDraw);

         // loop through draws and simulate each draw
         for (byte i = 0; i < legalDrawsCount; i++)
         {
             // simulate the draw
             applyDraw(simBitboards, draws[i]);
             uint64_t kingMask = simBitboards[sideOffset];

             // calculate enemy answer draws (only fields that could be captured as one bitboard)
             uint64_t enemyCapturableFields =
                 getKingDrawBitboards(simBitboards, opponent, false)
                 | getQueenDrawBitboards(simBitboards, opponent)
                 | getRookDrawBitboards(simBitboards, opponent)
                 | getBishopDrawBitboards(simBitboards, opponent)
                 | getKnightDrawBitboards(simBitboards, opponent)
                 | getPeasantDrawBitboards(simBitboards, opponent);

             // revert the simulated draw (flip the bits back, this actually works LOL!!!)
             applyDraw(simBitboards, draws[i]);
             // TODO: test if cloning the board is actually faster than reverting the draw (use the better option)

             // check if one of those draws would catch the allied king (bitwise AND) -> draw-into-check
             if ((kingMask & enemyCapturableFields) > 0)
             {
                 // overwrite the illegal draw with the last unevaluated draw in the array
                 draws[i--] = draws[--legalDrawsCount];
             }
         }

         // remove illegal draws
         draws = draws.SubArray(0, legalDrawsCount);
     }

     return draws;
}

size_t concat_draws(ChessDraw** out_draws, ChessDraw draws_a[], ChessDraw draws_b[])
{
    // TODO: check if this works

    size_t len_a = sizeof(draws_a) / sizeof(ChessDraw);
    size_t len_b = sizeof(draws_b) / sizeof(ChessDraw);

    *out_draws = malloc(len_a * len_b * sizeof(ChessDraw));
    memcpy(*out_draws, draws_a, len_a * sizeof(ChessDraw));
    memcpy(*out_draws + len_a, draws_b, len_b * sizeof(ChessDraw));
    // TODO: use secure memcpy instead

    return len_a + len_b;
}