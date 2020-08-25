#include "chesspieceatpos.h"

ChessPieceAtPos create_pieceatpos(ChessPosition pos, ChessPiece piece)
{
	ChessPieceAtPos pieceAtPos = { (uint16_t)((pos << 5) | piece) };
	return pieceAtPos;
}

ChessPiece get_pieceatpos_piece(ChessPieceAtPos pieceAtPos)
{
	return (uint8_t)(pieceAtPos & 0x1F);
}

ChessPosition get_pieceatpos_position(ChessPieceAtPos pieceAtPos)
{
	return (uint8_t)(pieceAtPos >> 5);
}