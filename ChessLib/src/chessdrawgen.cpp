#include "chessdrawgen.h"
using namespace std;

vector<ChessDraw> get_draws(Bitboard bitboards[], ChessColor side, ChessPieceType type, ChessDraw lastDraw);
vector<ChessDraw> eliminate_draws_into_check(ChessBoard board, vector<ChessDraw> draws);

Bitboard get_king_draw_positions(Bitboard bitboards[], ChessColor side, ChessPieceType type, ChessDraw lastDraw);
Bitboard get_queen_draw_positions(Bitboard bitboards[], ChessColor side, ChessPieceType type, ChessDraw lastDraw);
// TODO: reuse method stubs from C#

vector<ChessDraw> get_all_draws(ChessBoard board, ChessColor drawing_side, ChessDraw last_draw, int analyze_draw_into_check)
{
     // determine the drawing side
     uint8_t side_offset = SIDE_OFFSET(drawing_side);
     ChessColor opponent = OPPONENT(drawing_side);
     
     // compute the draws for the pieces of each type (for non-king pieces, check first if the bitboard actually contains pieces)
     vector<ChessDraw> king_draws    = get_draws(board.bitboards, drawing_side, King, last_draw);
     vector<ChessDraw> queen_draws   = (board.bitboards[side_offset + PIECE_OFFSET(Queen)]   != 0x0uLL) ? get_draws(board.bitboards, drawing_side, Queen, last_draw)   : vector<ChessDraw>();
     vector<ChessDraw> rook_draws    = (board.bitboards[side_offset + PIECE_OFFSET(Rook)]    != 0x0uLL) ? get_draws(board.bitboards, drawing_side, Rook, last_draw)    : vector<ChessDraw>();
     vector<ChessDraw> bishop_draws  = (board.bitboards[side_offset + PIECE_OFFSET(Bishop)]  != 0x0uLL) ? get_draws(board.bitboards, drawing_side, Bishop, last_draw)  : vector<ChessDraw>();
     vector<ChessDraw> knight_draws  = (board.bitboards[side_offset + PIECE_OFFSET(Knight)]  != 0x0uLL) ? get_draws(board.bitboards, drawing_side, Knight, last_draw)  : vector<ChessDraw>();
     vector<ChessDraw> peasant_draws = (board.bitboards[side_offset + PIECE_OFFSET(Peasant)] != 0x0uLL) ? get_draws(board.bitboards, drawing_side, Peasant, last_draw) : vector<ChessDraw>();
     
     // concatenate the draws as one vector
     vector<ChessDraw> draws(king_draws.size() + queen_draws.size() + rook_draws.size() + bishop_draws.size() + knight_draws.size() + peasant_draws.size());
     draws.insert(draws.end(), king_draws.begin(), king_draws.end());
     draws.insert(draws.end(), queen_draws.begin(), queen_draws.end());
     draws.insert(draws.end(), rook_draws.begin(), rook_draws.end());
     draws.insert(draws.end(), bishop_draws.begin(), bishop_draws.end());
     draws.insert(draws.end(), knight_draws.begin(), knight_draws.end());
     draws.insert(draws.end(), peasant_draws.begin(), peasant_draws.end());

     // if flag is active, filter only draws that do not cause draws into check
     return analyze_draw_into_check ? eliminate_draws_into_check(board, draws) : draws;
}

vector<ChessDraw> eliminate_draws_into_check(ChessBoard board, vector<ChessDraw> draws, ChessColor drawing_side)
{
    // make a working copy of all local bitboards
    Bitboard simBitboards[13];
    for (int i = 0; i < 13; i++) { simBitboards[i] = board.bitboards[i]; }

    // init legal draws count with the amount of all draws (unvalidated)
    size_t legalDrawsCount = sizeof(draws) / sizeof(ChessDraw);
    uint8_t side_offset = SIDE_OFFSET(drawing_side);
    ChessColor opponent = OPPONENT(drawing_side);

    // loop through draws and simulate each draw
    for (size_t i = 0; i < legalDrawsCount; i++)
    {
        // simulate the draw
        apply_draw_to_bitboards(simBitboards, draws[i]);
        Bitboard king_mask = simBitboards[side_offset];

        // calculate enemy answer draws (only fields that could be captured as one bitboard)
        Bitboard enemyCapturableFields =
              getKingDrawBitboards(simBitboards, opponent, false)
            | getQueenDrawBitboards(simBitboards, opponent)
            | getRookDrawBitboards(simBitboards, opponent)
            | getBishopDrawBitboards(simBitboards, opponent)
            | getKnightDrawBitboards(simBitboards, opponent)
            | getPeasantDrawBitboards(simBitboards, opponent);

        // revert the simulated draw (flip the bits back)
        apply_draw_to_bitboards(simBitboards, draws[i]);

        // check if one of those draws would catch the allied king (bitwise AND) -> draw-into-check
        if ((king_mask & enemyCapturableFields) > 0)
        {
            // overwrite the illegal draw with the last unevaluated draw in the array
            draws[i--] = draws[--legalDrawsCount];
            // TODO: check if this works
        }
    }

    // remove illegal draws
    return vector<ChessDraw>(draws.begin(), draws.begin() + legalDrawsCount);
}

vector<ChessDraw> get_draws(Bitboard bitboards[], ChessColor side, ChessPieceType type, ChessDraw lastDraw)
{
    // get drawinh pieces
    byte index = (byte)(side.SideOffset() + type.PieceTypeOffset());
    var drawingPieces = getPositions(bitboards[index]);

    // init draws result set (max. draws)
    var draws = new ChessDraw[drawingPieces.Length * 28];
    byte count = 0;

    // loop through drawing pieces
    for (byte i = 0; i < drawingPieces.Length; i++)
    {
        var pos = (byte)drawingPieces[i].GetHashCode();

        // only set the drawing piece to the bitboard, wipe all others
        ulong filter = 0x1uL << pos;
        ulong drawBitboard;

        // compute the chess piece's capturable positions as bitboard
        switch (type)
        {
        case ChessPieceType.King:    drawBitboard = getKingDrawBitboards(bitboards, side, true);                break;
        case ChessPieceType.Queen:   drawBitboard = getQueenDrawBitboards(bitboards, side, filter);             break;
        case ChessPieceType.Rook:    drawBitboard = getRookDrawBitboards(bitboards, side, filter);              break;
        case ChessPieceType.Bishop:  drawBitboard = getBishopDrawBitboards(bitboards, side, filter);            break;
        case ChessPieceType.Knight:  drawBitboard = getKnightDrawBitboards(bitboards, side, filter);            break;
        case ChessPieceType.Peasant: drawBitboard = getPeasantDrawBitboards(bitboards, side, lastDraw, filter); break;
        default: throw new ArgumentException("Invalid chess piece type detected!");
        }

        // extract all capturable positions from the draws bitboard
        var capturablePositions = getPositions(drawBitboard);

        // check for peasant promotion
        bool containsPeasantPromotion = type == ChessPieceType.Peasant
            && ((side == ChessColor.White && (drawBitboard & ROW_8) > 0) || (side == ChessColor.Black && (drawBitboard & ROW_1) > 0));

        // convert the positions into chess draws
        if (containsPeasantPromotion)
        {
            // peasant will advance to level 8, all draws need to be peasant promotions
            for (byte j = 0; j < capturablePositions.Length; j++)
            {
                // add types that the piece can promote to (queen, rook, bishop, knight)
                for (byte pieceType = 2; pieceType < 6; pieceType++) { draws[count++] = new ChessDraw(this, drawingPieces[i], capturablePositions[j], (ChessPieceType)pieceType); }
            }
        }
        else { for (byte j = 0; j < capturablePositions.Length; j++) { draws[count++] = new ChessDraw(this, drawingPieces[i], capturablePositions[j]); } }
    }

    return draws.SubArray(0, count);
}

Bitboard getCapturedFields(Bitboard bitboards[], ChessColor side)
{
    uint8_t offset = SIDE_OFFSET(side);

    return bitboards[offset] | bitboards[offset + 1] | bitboards[offset + 2]
        | bitboards[offset + 3] | bitboards[offset + 4] | bitboards[offset + 5];
}

size_t get_positions(ChessPosition** out_pos, Bitboard bitboard)
{
    // init position cache for worst-case
    size_t count = 0;
    *out_pos = malloc(32 * sizeof(ChessPosition));
     
    // loop through all bits of the board
    for (uint8_t pos = 0; pos < 64; pos++)
    {
        if ((bitboard & 0x1uLL << pos)) { (*out_pos)[count++] = (ChessPosition)pos; }
    }

    // return the resulting array (without empty entries)
    return count;
}

// this returns the numeric value of the highest bit set on the given bitboard
// if the given bitboard has multiple bits set, only the position of the highest bit is returned
// info: the result is mathematically equal to floor(log2(x))
ChessPosition get_position(Bitboard bitboard)
{
    // code was taken from https://stackoverflow.com/questions/11376288/fast-computing-of-log2-for-64-bit-integers

#ifdef __GNUC__
    // use built-in leading zeros function for GCC Linux build (this compiles to the very fast 'bsr' instruction on x86 AMD processors)
    return (ChessPosition)((unsigned)(8 * sizeof(unsigned long long) - __builtin_clzll((X)) - 1))
#else
    // use abstract DeBruijn algorithm with table lookup
    // TODO: think of implementing this as assembler code

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