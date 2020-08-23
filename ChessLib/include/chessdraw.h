
/* info: this file defines  */

#ifndef CHESSDRAW_H
#define CHESSDRAW_H

/* ====================================================
            L O A D   D E P E N D E N C I E S
   ==================================================== */

#include "chesspiece.h"
#include "chessposition.h"
#include "chessboard.h"
#include "chesstypes.h"

#include <stdint.h>
#include <stdlib.h>

/* ====================================================
               D E F I N E    M A K R O S
   ==================================================== */

#define DRAW_NULL create_draw_from_hash(0)

/* ====================================================
             D E F I N E    F U N C T I O N S
   ==================================================== */

ChessDraw create_draw(ChessBoard board, ChessPosition oldPos, ChessPosition newPos, ChessPieceType peasantPromotionType);

int get_is_first_move(ChessDraw draw);
ChessDrawType get_draw_type(ChessDraw draw);
ChessColor get_drawing_side(ChessDraw draw);
ChessPieceType get_drawing_piece_type(ChessDraw draw);
ChessPieceType get_taken_piece_type(ChessDraw draw);
ChessPieceType get_peasant_promotion_piece_type(ChessDraw draw);
ChessPosition get_old_position(ChessDraw draw);
ChessPosition get_new_position(ChessDraw draw);

#endif