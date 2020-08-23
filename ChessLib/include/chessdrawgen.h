#ifndef CHESSDRAWGEN_H
#define CHESSDRAWGEN_H

#include "chesstypes.h"
#include "chessboard.h"
#include "chessdraw.h"
#include "chesspiece.h"
#include "chessposition.h"

#include <stdlib.h>
#include <string.h>

void get_all_draws(ChessDraw** out_draws, size_t* out_length, ChessBoard board, ChessColor drawing_side, ChessDraw last_draw, int analyze_draw_into_check);

#endif