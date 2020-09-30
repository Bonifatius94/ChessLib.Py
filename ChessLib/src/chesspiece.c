/*
 * MIT License
 * 
 * Copyright(c) 2020 Marco Tröster
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "chesspiece.h"

ChessPiece create_piece(ChessPieceType type, ChessColor color, int was_moved)
{
    return (ChessPiece)(((was_moved << 4) | ((uint8_t)color << 3) | (uint8_t)type) & 0x1F);
}

ChessPiece create_piece_from_hash(uint8_t hash)
{
    return (ChessPiece)(hash & 0x1F);
}

int get_was_piece_moved(ChessPiece piece)
{
    return (int)((piece >> 4) & 0x1);
}

ChessColor get_piece_color(ChessPiece piece)
{
    return (ChessColor)((piece >> 3) & 0x1);
}

ChessPieceType get_piece_type(ChessPiece piece)
{
    return (ChessPieceType)(piece & 0x7);
}

ChessColor color_from_char(char c)
{
    switch (toupper(c))
    {
        case 'W': return White;
        case 'B': return Black;
        default:  return White;
    }
}

char color_to_char(ChessColor color)
{
    switch (color)
    {
        case White: return 'W';
        case Black: return 'B';
        default:    return 'W';
    }
}

ChessPieceType piece_type_from_char(char c)
{
    switch (toupper(c))
    {
        case 'I': return (ChessPieceType)Invalid;
        case 'K': return (ChessPieceType)King;
        case 'Q': return (ChessPieceType)Queen;
        case 'R': return (ChessPieceType)Rook;
        case 'B': return (ChessPieceType)Bishop;
        case 'N': return (ChessPieceType)Knight;
        case 'P': return (ChessPieceType)Peasant;
        default:  return (ChessPieceType)Invalid;
    }
}

char piece_type_to_char(ChessPieceType type)
{
    switch (type)
    {
        case Invalid: return 'I';
        case King:    return 'K';
        case Queen:   return 'Q';
        case Rook:    return 'R';
        case Bishop:  return 'B';
        case Knight:  return 'N';
        case Peasant: return 'P';
        default:      return 'I';
    }
}

char* color_to_string(ChessColor color)
{
    switch (color)
    {
        case White: return "White";
        case Black: return "Black";
        default:    return "White";
    }
}

char* piece_type_to_string(ChessPieceType type)
{
    switch (type)
    {
        case Invalid: return "Invalid";
        case King:    return "King";
        case Queen:   return "Queen";
        case Rook:    return "Rook";
        case Bishop:  return "Bishop";
        case Knight:  return "Knight";
        case Peasant: return "Peasant";
        default:      return "Invalid";
    }
}