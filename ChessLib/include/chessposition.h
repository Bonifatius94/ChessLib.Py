#ifndef CHESS_POSITION_H
#define CHESS_POSITION_H

/* ====================================================
            L O A D   D E P E N D E N C I E S
   ==================================================== */

#include "chesstypes.h"
#include <stdint.h>

/* ====================================================
             D E F I N E    F U N C T I O N S
   ==================================================== */

//ChessPosition create_position_from_hash(uint8_t hash);
ChessPosition create_position(int row, int column);

int8_t get_row(ChessPosition position);
int8_t get_column(ChessPosition position);

#endif