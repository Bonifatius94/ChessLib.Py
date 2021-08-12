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

#ifndef CHESSXFORMAT_H
#define CHESSXFORMAT_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "chesstypes.h"
#include "chesspiece.h"
#include "chessboard.h"
#include "chessgamesession.h"

/* TODO: move the board hashing code here */

int chess_session_from_fen(const char fen_str[], ChessGameSession* session);
int chess_session_to_fen(char* fen_str, const ChessGameSession* session);
int chess_session_from_pgn(const char pgn_str[], ChessGameSession* session);
int chess_session_to_pgn(char* pgn_str, const ChessGameSession* session);

#endif