#include "chessposition.h"

ChessPosition create_position(int8_t row, int8_t column)
{
    return (ChessPosition)((row << 3) | column);
}

int8_t get_row(ChessPosition position)
{
    return (position >> 3);
}

int8_t get_column(ChessPosition position)
{
    return (position & 7);
}
