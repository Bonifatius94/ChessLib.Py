#ifndef CHESSTYPES_H
#define CHESSTYPES_H

#include <stdint.h>

/* include size_t type for gcc */
#ifdef __GNUC__
    #include <stddef.h>
#endif

typedef enum _CHESS_PIECE_TYPE {
    Invalid = 0,
    King = 1,
    Queen = 2,
    Rook = 3,
    Bishop = 4,
    Knight = 5,
    Peasant = 6
} ChessPieceType;

typedef enum _CHESS_COLOR {
    White = 0,
    Black = 1
} ChessColor;

typedef enum _CHESS_DRAW_TYPE {
    Standard = 0,
    Rochade = 1,
    EnPassant = 2,
    PeasantPromotion = 3
} ChessDrawType;

/* | was moved | color | piece type |
   |         x |     x |        xxx | */
typedef uint8_t ChessPiece;

/* |  row | column |
   |  xxx |    xxx | */
typedef uint8_t ChessPosition;

/* | position | piece |
   |   xxxxxx | xxxxx | */
typedef uint16_t ChessPieceAtPos;

/* |  position | piece |
   |    xxxxxx | xxxxx | */
/*typedef uint16_t ChessPieceAtPos;*/

/* |  unused | is first move | side | draw type | piece type | taken piece type | promotion type | old position | new position |
   | xxxxxxx |             x |    x |        xx |        xxx |              xxx |            xxx |       xxxxxx |       xxxxxx | */
typedef uint32_t ChessDraw;

/* A chess bitboard with each bit representing a field onto a chess board. Addressing is normalized, starting with the lowest bit as A1 and the highest bit as H8 (indexes A1=0, B1=1, ..., A2=8, ..., H8=63). */
typedef uint64_t Bitboard;

/* The chess board represented as 13 bitboards of unsigned 64-bit integers. 
   The chess field allocation is normalized by the index values corresponding to ChessPosition type starting with the lowest bit as A1 and ending with the highest bit as H8 (indexes A1=0, B1=1, ..., A2=8, ..., H8=63).
   The first 12 boards show positions of the chess pieces, the last board keeps track of was_moved states.
   The boards holding information on chess pieces are ordered by the occurance of the piece in the ChessPieceType enum (King=0, Queens=1, Rooks=2, Bishops=3, Knights=4, Peasants=5).
   All boards with indices 0-5 belong to the white side, the next 6 boards with indices 6-11 belong to the black side.
*/
typedef struct _CHESS_BITBOARD {
    /* the bitboards for each  */
    Bitboard bitboards[13];
} ChessBoard;

/*typedef Bitboard * ChessBoard;*/

#endif
