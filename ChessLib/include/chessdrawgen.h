#ifndef CHESSDRAWGEN_H
#define CHESSDRAWGEN_H

#include "chesstypes.h"
#include "chessboard.h"
#include "chessdraw.h"
#include "chesspiece.h"
#include "chessposition.h"

#include <stdlib.h>
#include <string.h>
#include <vector>

// TODO: add method stubs from ChessBoard.cs

size_t get_possible_draws(ChessDraw** out_draws, ChessBoard board, ChessColor drawing_side, ChessDraw last_draw, int analyze_draw_into_check);

#endif