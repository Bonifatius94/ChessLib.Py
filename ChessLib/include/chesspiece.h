
/* info: this file defines  */

#ifndef CHESSPIECE_H
#define CHESSPIECE_H

/* ====================================================
            L O A D   D E P E N D E N C I E S
   ==================================================== */

#include "chesstypes.h"
#include <stdint.h>

/* ====================================================
               D E F I N E    M A K R O S
   ==================================================== */

#define CHESS_PIECE_NULL create_piece_from_hash(0)
#define OPPONENT(color) ((ChessColor)((uint8_t)(color) ^ 1))

/* ====================================================
             D E F I N E    F U N C T I O N S
   ==================================================== */

ChessPiece create_piece_from_hash(uint8_t hash);
ChessPiece create_piece(ChessPieceType type, ChessColor color, int was_moved);

ChessColor get_piece_color(ChessPiece piece);
int get_was_piece_moved(ChessPiece piece);
ChessPieceType get_piece_type(ChessPiece piece);

#endif