#include "chesspieceatpos.h"

ChessPieceAtPos create_pieceatpos(ChessPosition pos, ChessPiece piece)
{
	ChessPieceAtPos pieceAtPos = { (uint16_t)((pos << 5) | piece) };
	return pieceAtPos;
}

ChessPiece get_piece(ChessPieceAtPos pieceAtPos)
{
	return create_piece_from_hash((uint8_t)(pieceAtPos & 0x1F));
}

ChessPosition get_position(ChessPieceAtPos pieceAtPos)
{
	return create_position_from_hash((uint8_t)(pieceAtPos >> 5));
}