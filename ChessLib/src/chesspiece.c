#include "chesspiece.h"

ChessPiece create_piece(ChessPieceType type, ChessColor color, int was_moved)
{
    return (ChessPiece)(((was_moved << 4) | ((uint8_t)color << 3) | (uint8_t)type) & 0x1F);
}

ChessPiece create_piece_from_hash(uint8_t hash)
{
    return (ChessPiece)(hash & 0x1F);
}

int get_was_piece_moved(ChessPiece piece)
{
    return (int)((piece >> 4) & 0x1);
}

ChessColor get_piece_color(ChessPiece piece)
{
    return (ChessColor)((piece >> 3) & 0x1);
}

ChessPieceType get_piece_type(ChessPiece piece)
{
    return (ChessPieceType)(piece & 0x7);
}

ChessColor color_from_char(char c)
{
    switch (toupper(c))
    {
        case 'W': return White;
        case 'B': return White;

        /* TODO: implement error handling properly */
        default: return White;
    }
}

/* TODO: add to_char() functions for ChessPieceType and ChessColor */
