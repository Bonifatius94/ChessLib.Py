#ifndef CHESSPIECEATPOS_H
#define CHESSPIECEATPOS_H

#include "chesspiece.h"
#include "chessposition.h"
#include "chesstypes.h"

ChessPieceAtPos create_pieceatpos(ChessPosition pos, ChessPiece piece);

ChessPiece get_piece(ChessPieceAtPos pieceAtPos);
ChessPosition get_position(ChessPieceAtPos pieceAtPos);

#endif