#include "chessposition.h"

//ChessPosition create_position_from_hash(uint8_t hash)
//{
//	return (ChessPosition)hash;
//}

ChessPosition create_position(int row, int column)
{
	return (ChessPosition)((row << 3) & column);
}

int8_t get_row(ChessPosition position)
{
	return (position >> 3);
}

int8_t get_column(ChessPosition position)
{
	return (position & 7);
}