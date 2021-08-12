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

int position_from_string(const char* pos_str, ChessPosition* pos)
{
    uint8_t row, column;

    /* make sure the first character is within [a-h] or [A-H] */
    if (!isalpha(pos_str[0])
        || toupper(pos_str[0]) - 'A' >= 8
        || toupper(pos_str[0]) - 'A' < 0)
    { return 0; }

    /* make sure the second character is within [1-8] */
    if (!isdigit(pos_str[1]) || pos_str[1] - '1' >= 8) { return 0; }

    /* make sure the third character is a zero-terminal */
    if (pos_str[2] != '\0') { return 0; }

    /* finally, do the actual parsing */
    row = pos_str[1] - '1';
    column = toupper(pos_str[0]) - 'A';
    *pos = create_position(row, column);

    /* parsing successful! */
    return 1;
}

void position_to_string(ChessPosition position, char* pos_str)
{
    /* serialize position as string*/
    pos_str[0] = 'A' + get_column(position);
    pos_str[1] = '1' + get_row(position);
    pos_str[2] = '\0';
}