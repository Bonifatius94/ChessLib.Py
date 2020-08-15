//#include "chesspieceatpos.h"
//
//ChessPieceAtPos create_pieceatpos_from_hash(uint16_t hash)
//{
//	ChessPieceAtPos pieceAtPos = { hash };
//	return pieceAtPos;
//}
//
//ChessPieceAtPos create_pieceatpos(ChessPosition pos, ChessPiece piece)
//{
//	ChessPieceAtPos pieceAtPos = { (uint16_t)((pos.hash << 5) | piece.hash) };
//	return pieceAtPos;
//}
//
//ChessPiece get_piece(ChessPieceAtPos pieceAtPos)
//{
//	return create_piece_from_hash((uint8_t)(pieceAtPos.hash & 0x1F));
//}
//
//ChessPosition get_position(ChessPieceAtPos pieceAtPos)
//{
//	return create_position_from_hash((uint8_t)(pieceAtPos.hash >> 5));
//}