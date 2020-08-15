#include "chesspiece.h"

ChessPiece create_piece(ChessPieceType type, ChessColor color, int was_moved)
{
	return (ChessPiece)((was_moved << 4) | ((uint8_t)color << 3) | (uint8_t)type);
}

ChessPiece create_piece_from_hash(uint8_t hash)
{
	return (ChessPiece)hash;
}

int get_was_piece_moved(ChessPiece piece)
{
	return (int)((piece >> 4) & 1);
}

ChessColor get_piece_color(ChessPiece piece)
{
	return (ChessColor)((piece >> 3) & 1);
}

ChessPieceType get_piece_type(ChessPiece piece)
{
	return (ChessPieceType)(piece & 7);
}

/* TODO: add to_char() functions for ChessPieceType and ChessColor */