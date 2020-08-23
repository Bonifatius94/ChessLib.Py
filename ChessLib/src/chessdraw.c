#include "chessdraw.h"

ChessDraw create_draw_from_hash(uint32_t hash)
{
	return (ChessDraw)hash;
}

ChessDrawType determine_draw_type(ChessBoard board, ChessPosition oldPos, ChessPosition newPos, ChessPieceType peasantPromotionType)
{
    ChessDrawType type = Standard;
    ChessPiece piece = get_piece_at(board, oldPos);
    
    /* check for a peasant promotion */
    if (peasantPromotionType != Invalid && get_piece_type(piece) == Peasant
        && ((get_row(newPos) == 7 && get_piece_color(piece) == White)
            || (get_row(newPos) == 0 && get_piece_color(piece) == Black)))
    {
        type = PeasantPromotion;
    }
    /* check for a rochade */
    else if (get_piece_type(piece) == King && abs(get_column(oldPos) - get_column(newPos)) == 2)
    {
        type = Rochade;
    }
    /* check for an en-passant */
    else if (get_piece_type(piece) == Peasant && !is_captured_at(board, newPos)
        && abs(get_column(oldPos) - get_column(newPos)) == 1)
    {
        type = EnPassant;
    }

    return type;
}

ChessDraw create_draw(ChessBoard board, ChessPosition oldPos, ChessPosition newPos, ChessPieceType peasantPromotionType)
{
    ChessPiece piece;
    int is_first_move;
    ChessDrawType draw_type;
    ChessColor drawing_side;
    ChessPieceType drawing_piece_type, taken_piece_type;
    ChessDraw draw;

    /* get the drawing piece */
    ChessPiece piece = get_piece_at(board, oldPos);

    /* determine all property values */
    is_first_move = get_was_piece_moved(piece) == 0 ? 1 : 0;
    draw_type = determine_draw_type(board, oldPos, newPos, peasantPromotionType);
    drawing_side = get_piece_color(piece);
    drawing_piece_type = get_piece_type(piece);
    taken_piece_type = 
        (draw_type == EnPassant) ? Peasant : (is_captured_at(board, newPos) ? get_piece_type(get_piece_at(board, newPos)) : Invalid);
    
    /* transform property values to a hash code */
    draw = (ChessDraw)(
          (is_first_move << 24)
        | (drawing_side << 23)
        | (draw_type << 21)
        | (drawing_piece_type << 18)
        | (taken_piece_type << 15)
        | (peasantPromotionType << 12)
        | (oldPos << 6)
        | newPos);

    return draw;
}

int get_is_first_move(ChessDraw draw)
{
	return (int)((draw >> 24) & 0x1);
}

ChessColor get_drawing_side(ChessDraw draw)
{
	return (ChessColor)((draw >> 23) & 0x1);
}

ChessDrawType get_draw_type(ChessDraw draw)
{
	return (ChessColor)((draw >> 21) & 0x3);
}

ChessPieceType get_drawing_piece_type(ChessDraw draw)
{
	return (ChessPieceType)((draw >> 18) & 0x7);
}

ChessPieceType get_taken_piece_type(ChessDraw draw)
{
	return (ChessPieceType)((draw >> 15) & 0x7);
}

ChessPieceType get_peasant_promotion_piece_type(ChessDraw draw)
{
	return (ChessPieceType)((draw >> 12) & 0x7);
}

ChessPosition get_old_position(ChessDraw draw)
{
	return (ChessPosition)((draw >> 6) & 0x3F);
}

ChessPosition get_new_position(ChessDraw draw)
{
	return (ChessPosition)(draw & 0x3F);
}