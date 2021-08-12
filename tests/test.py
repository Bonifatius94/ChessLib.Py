
# ================================================================================= #
# MIT License                                                                       #
#                                                                                   #
# Copyright(c) 2020 Marco Tröster                                                   #
#                                                                                   #
# Permission is hereby granted, free of charge, to any person obtaining a copy      #
# of this software and associated documentation files (the "Software"), to deal     #
# in the Software without restriction, including without limitation the rights      #
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell         #
# copies of the Software, and to permit persons to whom the Software is             #
# furnished to do so, subject to the following conditions:                          #
#                                                                                   #
# The above copyright notice and this permission notice shall be included in all    #
# copies or substantial portions of the Software.                                   #
#                                                                                   #
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR        #
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,          #
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE       #
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER            #
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,     #
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE     #
# SOFTWARE.                                                                         #
# ================================================================================= #

import sys
import chesslib
import numpy as np
from asserts import assert_true, assert_equal


def test_module():

    # test base datatypes
    test_chesscolor()
    test_chessposition()
    test_chesspiece()
    test_chessdraw_null()
    test_chessboard_start()
    test_chesspieceatpos()
    test_create_chessboard()
    test_create_chessdraw()

    # test gameplay functions
    test_drawgen()
    test_apply_draw()
    test_game_state()
    test_board_hash()
    test_board_from_fen()

    # # test visualization functions
    test_visualize_board()
    test_visualize_draw()


def test_chesscolor():

    print("testing chess color struct")

    # test white side (expected integer with value=0)
    white_side = chesslib.ChessColor_White
    assert_equal(white_side, 0)

    # test black side (expected integer with value=1)
    black_side = chesslib.ChessColor_Black
    assert_equal(black_side, 1)

    print("test passed!")


def test_chessposition():

    print("testing chess position struct")

    # create column char mappings
    col_chars = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

    # loop through all rows and columns
    for row in range(8):
        for col in range(8):
            # test the creation of a new chessposition of the given (row, column) tuple
            pos_string = col_chars[col] + str(row + 1)
            pos = chesslib.ChessPosition(pos_as_str=pos_string)

            # make sure that the position's numeric value is equal to the expected index
            assert_equal(pos, row * 8 + col)

    print("test passed!")


def test_chesspiece():

    print("testing chess piece struct")

    # create piece type char mappings (invalid, king, queen, rook, bishop, knight, pawn)
    piece_type_chars = ['I', 'K', 'Q', 'R', 'B', 'N', 'P']

    # loop through all possible pieces
    for type in range(1, 7):
        for was_moved in range(2):
            for color in range(2):
                # test the creation of the new chesspiece
                color_char = 'W' if color == 0 else 'B'
                type_char = piece_type_chars[type]
                piece = chesslib.ChessPiece(color=color_char, piece_type=type_char, was_moved=was_moved)

                # make sure that the numeric value is correctly encoded
                assert_equal(piece, type + 8 * color + 16 * was_moved)

    print("test passed!")


def test_chesspieceatpos():

    print("testing chess piece at pos struct")

    # create column char mappings
    col_chars = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

    # create piece type char mappings (invalid, king, queen, rook, bishop, knight, pawn)
    piece_type_chars = ['I', 'K', 'Q', 'R', 'B', 'N', 'P']

    # loop through all rows and columns
    for row in range(8):
        for col in range(8):
            # test the creation of a new chessposition of the given (row, column) tuple
            pos_string = col_chars[col] + str(row + 1)
            pos = chesslib.ChessPosition(pos_string)

            # loop through all possible pieces
            for type in range(1, 7):
                for was_moved in range(2):
                    for color in range(2):
                        # test the creation of the new chesspiece
                        color_char = 'W' if color == 0 else 'B'
                        type_char = piece_type_chars[type]
                        piece = chesslib.ChessPiece(color_char, type_char, was_moved)

                        # test the creation of a pieceatpos struct with the given piece and pos
                        piece_at_pos = chesslib.ChessPieceAtPos(piece=piece, pos=pos)

                        # make sure that the numeric value of pieceatpos is correctly encoded
                        assert_equal(piece_at_pos, pos * 32 + piece)

    print("test passed!")


def test_chessdraw_null():

    print("testing chessdraw null value")

    # test if the expected null value is returned
    assert_equal(chesslib.ChessDraw_Null, 0)

    print("test passed!")


def test_create_chessdraw():

    # code snippet to be tested
    # ==========================
    # if (!PyArg_ParseTuple(args, "Okk|kii", &chessboard, &old_pos, &new_pos, 
    # &prom_type, &is_compact_format, &is_simple_board)) { return NULL; }

    print("testing creation of chess draws")

    # assign the expected draw codes for 'white peasant E2-E4' in start formation
    exp_draw = 18350876
    exp_compact_draw = 18350876 & 0x7FFF

    # standard creation, draw E2-E4 from start formation, board in bitboards format
    board = chesslib.ChessBoard_StartFormation()
    gen_draw = chesslib.ChessDraw(board=board, old_pos=chesslib.ChessPosition('E2'), new_pos=chesslib.ChessPosition('E4'))
    assert_equal(gen_draw, exp_draw)
    assert_equal(sys.getrefcount(board), 2)

    # compact creation, draw E2-E4 from start formation, board in bitboards format
    # board = chesslib.ChessBoard_StartFormation()
    gen_draw = chesslib.ChessDraw(board, chesslib.ChessPosition('E2'), chesslib.ChessPosition('E4'), is_compact_draw=True)
    assert_equal(gen_draw, exp_compact_draw)
    assert_equal(sys.getrefcount(board), 2)

    # standard creation, draw E2-E4 from start formation, board in simple format
    board = chesslib.ChessBoard_StartFormation(True)
    gen_draw = chesslib.ChessDraw(board, chesslib.ChessPosition('E2'), chesslib.ChessPosition('E4'), is_simple=True)
    assert_equal(gen_draw, exp_draw)
    assert_equal(sys.getrefcount(board), 2)

    # compact creation, draw E2-E4 from start formation, board in simple format
    # board = chesslib.ChessBoard_StartFormation(True)
    gen_draw = chesslib.ChessDraw(board, chesslib.ChessPosition('E2'), chesslib.ChessPosition('E4'), is_compact_draw=True, is_simple=True)
    assert_equal(gen_draw, exp_compact_draw)
    # print(gen_draw, exp_compact_draw)
    assert_equal(sys.getrefcount(board), 2)

    # TODO: add a peasant prom. test case

    print("test passed!")


def test_create_chessboard():

    print("test chessboard creation from pieceatpos array")

    # define the pieces to be put onto the board at the given positions
    pieces_at_pos = np.array([
        chesslib.ChessPieceAtPos(chesslib.ChessPiece('W', 'K', False), chesslib.ChessPosition('E1')),
        chesslib.ChessPieceAtPos(chesslib.ChessPiece('B', 'K', True ), chesslib.ChessPosition('G8')),
        chesslib.ChessPieceAtPos(chesslib.ChessPiece('W', 'R', True ), chesslib.ChessPosition('A3')),
        chesslib.ChessPieceAtPos(chesslib.ChessPiece('B', 'Q', False), chesslib.ChessPosition('D8')),
        chesslib.ChessPieceAtPos(chesslib.ChessPiece('W', 'R', True ), chesslib.ChessPosition('B7')),
        chesslib.ChessPieceAtPos(chesslib.ChessPiece('B', 'B', True ), chesslib.ChessPosition('C5')),
    ], np.uint16)
    # TODO: add pawns to the collection for edge case testing

    # define the expected board
    exp_board = np.array([
        0x0000000000000010, # white king at E1
        0x0000000000000000, # nothing
        0x0002000000010000, # white rooks at A3, B7
        0x0000000000000000, # nothing
        0x0000000000000000, # nothing
        0x0000000000000000, # nothing
        0x4000000000000000, # black king at G8
        0x0800000000000000, # black queen at D8
        0x0000000000000000, # nothing
        0x0000000400000000, # black bishop at C5
        0x0000000000000000, # nothing
        0x0000000000000000, # nothing
        0xF7FFFFFFFFFFFFEF  # unmoved at E1, D8
    ], dtype=np.uint64)

    # create the chessboard from pieces at pos array and make sure it was created correctly
    board = chesslib.ChessBoard(pieces_list=pieces_at_pos)
    assert_true(np.array_equal(exp_board, board))

    # make sure the pieces@pos reference counters were decremented properly
    assert_equal(sys.getrefcount(pieces_at_pos), 2)

    # define the expected board
    exp_simple_board = np.array([
		0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x13, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x1C, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x13, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x0A, 0x00, 0x00, 0x19, 0x00,
    ], dtype=np.uint8)

    # create the chessboard from pieces at pos array (simple format) and make sure it was created correctly
    simple_board = chesslib.ChessBoard(pieces_at_pos, is_simple=True)
    assert_true(np.array_equal(exp_simple_board, simple_board))

    # make sure the pieces@pos reference counters were decremented properly
    assert_equal(sys.getrefcount(pieces_at_pos), 2)

    print("test passed!")


def test_chessboard_start():

    print("testing chess board start formation value")

    # initialize expected bitboards array in start formation
    exp_start_formation = np.array([
        0x0000000000000010,
        0x0000000000000008,
        0x0000000000000081,
        0x0000000000000024,
        0x0000000000000042,
        0x000000000000FF00,
        0x1000000000000000,
        0x0800000000000000,
        0x8100000000000000,
        0x2400000000000000,
        0x4200000000000000,
        0x00FF000000000000,
        0x0000FFFFFFFF0000
    ], dtype=np.uint64)

    # test if the expected board in start formation is returned
    start = chesslib.ChessBoard_StartFormation()
    assert_true(np.array_equal(start, exp_start_formation))

    # test simple chess board format
    # initialize the expected chess pieces array in start formation
    exp_simple_start_formation = np.array([
        0x3, 0x5, 0x4, 0x2, 0x1, 0x4, 0x5, 0x3,
        0x6, 0x6, 0x6, 0x6, 0x6, 0x6, 0x6, 0x6,
        0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
        0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
        0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
        0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
        0xE, 0xE, 0xE, 0xE, 0xE, 0xE, 0xE, 0xE,
        0xB, 0xD, 0xC, 0xA, 0x9, 0xC, 0xD, 0xB,
    ], dtype=np.uint8)

    # test if the expected board in start formation is returned
    simple_start = chesslib.ChessBoard_StartFormation(is_simple=True)
    assert_true(np.array_equal(simple_start, exp_simple_start_formation))

    print("test passed!")


def test_drawgen():

    print("testing draw-gen")

    # get all draws for starting position (white side)
    board = chesslib.ChessBoard_StartFormation()
    simple_board = chesslib.ChessBoard_StartFormation(True)
    draws = chesslib.GenerateDraws(board=board, drawing_side=chesslib.ChessColor_White, last_draw=chesslib.ChessDraw_Null, analyze_check=True)

    # make sure that simple board and bitboards representation produce the same output
    assert_true(set(draws) == set(chesslib.GenerateDraws(simple_board, chesslib.ChessColor_White, chesslib.ChessDraw_Null, True, is_simple=True)))

    # make sure the board reference counters were decremented properly
    assert_equal(sys.getrefcount(board), 2)
    assert_equal(sys.getrefcount(simple_board), 2)

    # define the expected draws (16 peasant draws, 4 knight draws)
    expected_draws = np.array([
        18088016, 18088018, 18088341, 18088343, 18350608, 18350616, 18350673, 18350681,
        18350738, 18350746, 18350803, 18350811, 18350868, 18350876, 18350933, 18350941,
        18350998, 18351006, 18351063, 18351071
    ], np.uint32)

    # make sure that the generated draws equal the expected draws
    assert_true(set(draws) == set(expected_draws))

    print("test passed!")
    print("testing draw-gen (compact draws)")

    # get all draws for starting position (white side)
    comp_draws = chesslib.GenerateDraws(board, chesslib.ChessColor_White, chesslib.ChessDraw_Null, True, is_compact_draw=True)

    # make sure that simple board and bitboards representation produce the same output
    assert_true(set(comp_draws) == set(chesslib.GenerateDraws(simple_board, chesslib.ChessColor_White, chesslib.ChessDraw_Null, True, True, True)))

    # make sure the board reference counters were decremented properly
    assert_equal(sys.getrefcount(board), 2)
    assert_equal(sys.getrefcount(simple_board), 2)

    # define the expected draws (only lowest 15 bits of the long draw format)
    expected_comp_draws = expected_draws & 0x7FFF

    # make sure that the generated draws equal the expected draws
    assert_true(set(comp_draws) == set(expected_comp_draws))

    # TODO: add more unit tests that at least cover the correct parsing of all parameters
    # TODO: try to test the assignment all possible draw attribute values at least once (not all perms of course)

    print("test passed!")


def test_apply_draw():

    print("test applying draws to chess boards")

    # get board in start formation and opening draw 'white peasant E2-E4'
    board = chesslib.ChessBoard_StartFormation()
    simple_board = chesslib.ChessBoard_StartFormation(True)
    draw = 0x0118070C

    # try applying the draw
    board_after = chesslib.ApplyDraw(board, draw)
    simple_board_after = chesslib.ApplyDraw(board=simple_board, draw=draw, is_simple=True)

    # make sure the board reference counters were decremented properly
    assert_equal(sys.getrefcount(board), 2)
    assert_equal(sys.getrefcount(simple_board), 2)

    # define the expected board after applying the draw
    exp_board_after = np.array([
        0x0000000000000010,
        0x0000000000000008,
        0x0000000000000081,
        0x0000000000000024,
        0x0000000000000042,
        0x000000001000EF00,
        0x1000000000000000,
        0x0800000000000000,
        0x8100000000000000,
        0x2400000000000000,
        0x4200000000000000,
        0x00FF000000000000,
        0x0000FFFFFFFF1000
    ], dtype=np.uint64)

    # define the expected board after applying the draw
    exp_simple_board_after = np.array([
        0x03, 0x05, 0x04, 0x2, 0x01, 0x04, 0x05, 0x03,
        0x06, 0x06, 0x06, 0x6, 0x00, 0x06, 0x06, 0x06,
        0x00, 0x00, 0x00, 0x0, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x0, 0x16, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x0, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x0, 0x00, 0x00, 0x00, 0x00,
        0x0E, 0x0E, 0x0E, 0xE, 0x0E, 0x0E, 0x0E, 0x0E,
        0x0B, 0x0D, 0x0C, 0xA, 0x09, 0x0C, 0x0D, 0x0B,
    ], dtype=np.uint8)

    # make sure that the new board is as expected
    assert_true(np.array_equal(exp_board_after, board_after))
    assert_true(np.array_equal(exp_simple_board_after, simple_board_after))

    # test if ApplyDraw() function is revertible
    rev_board = chesslib.ApplyDraw(board_after, draw)
    simple_rev_board = chesslib.ApplyDraw(simple_board_after, draw, True)

    # make sure the board reference counters were decremented properly
    assert_equal(sys.getrefcount(board), 2)
    assert_equal(sys.getrefcount(simple_board), 2)

    # make sure that the reverted board is the same as the original board
    assert_true(np.array_equal(board, rev_board))
    assert_true(np.array_equal(simple_board, simple_rev_board))

    print("test passed!")


def test_board_hash():

    print("testing board to hash function")

    # create board in start formation
    board = chesslib.ChessBoard_StartFormation()
    simple_board = chesslib.ChessBoard_StartFormation(True)

    # make sure that simple and bitboards representations create the same hash
    assert_true(np.array_equal(chesslib.Board_ToHash(board), chesslib.Board_ToHash(simple_board, True)))

    # compute the board's 40-byte hash
    hash = chesslib.Board_ToHash(board=board)
    assert_equal(sys.getrefcount(board), 2)
    hash = chesslib.Board_ToHash(simple_board, is_simple=True)
    assert_equal(sys.getrefcount(simple_board), 2)

    # define the expected hash
    exp_hash = np.array([
        0x19, 0x48, 0x20, 0x90, 0xA3,
        0x31, 0x8C, 0x63, 0x18, 0xC6,
        0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00,
        0x73, 0x9C, 0xE7, 0x39, 0xCE,
        0x5B, 0x58, 0xA4, 0xB1, 0xAB
    ], dtype=np.uint8)

    # make sure that the computed hash is correct
    assert_true(np.array_equal(exp_hash, hash))

    # convert the board's 40-byte hash back to a board instance
    board_copy = chesslib.Board_FromHash(hash=hash)

    # make sure that the board converted from 40-byte hash is the same as the original board
    assert_true(np.array_equal(board_copy, board))
    assert_equal(sys.getrefcount(hash), 2)

    # convert the board's 40-byte hash back to a simple board instance
    simple_board_copy = chesslib.Board_FromHash(hash=hash, is_simple=True)

    # make sure that the board converted from 40-byte hash is the same as the original simple board
    assert_true(np.array_equal(simple_board_copy, simple_board))
    assert_equal(sys.getrefcount(hash), 2)

    # TODO: figure out why np.frombuffer(hash) causes a segfault when reading ndarray bytes

    print("test passed!")


def test_game_state():

    print("testing game state function")

    # create a chess board with a check-mate position
    checkmate_board_pieces = np.array([
        chesslib.ChessPieceAtPos(chesslib.ChessPiece('W', 'K', False), chesslib.ChessPosition('E1')),
        chesslib.ChessPieceAtPos(chesslib.ChessPiece('W', 'R', True), chesslib.ChessPosition('H1')),
        chesslib.ChessPieceAtPos(chesslib.ChessPiece('W', 'R', True), chesslib.ChessPosition('G7')),
        chesslib.ChessPieceAtPos(chesslib.ChessPiece('B', 'K', False), chesslib.ChessPosition('E8')),
    ], dtype=np.uint16)
    checkmate_board = chesslib.ChessBoard(checkmate_board_pieces)
    checkmate_draw = chesslib.ChessDraw(checkmate_board,
        chesslib.ChessPosition('H1'), chesslib.ChessPosition('H8'))
    checkmate_board = chesslib.ApplyDraw(checkmate_board, checkmate_draw)

    # test the GameState() function to detect the mate
    state = chesslib.GameState(board=checkmate_board, last_draw=checkmate_draw)
    assert_equal(state, chesslib.GameState_Checkmate)

    # create a chess board with a stalemate position
    stalemate_board_pieces = np.array([
        chesslib.ChessPieceAtPos(chesslib.ChessPiece('W', 'K', True), chesslib.ChessPosition('E6')),
        chesslib.ChessPieceAtPos(chesslib.ChessPiece('W', 'R', True), chesslib.ChessPosition('C1')),
        chesslib.ChessPieceAtPos(chesslib.ChessPiece('W', 'R', True), chesslib.ChessPosition('F7')),
        chesslib.ChessPieceAtPos(chesslib.ChessPiece('B', 'K', False), chesslib.ChessPosition('E8')),
    ], dtype=np.uint16)
    stalemate_board = chesslib.ChessBoard(stalemate_board_pieces)
    stalemate_draw = chesslib.ChessDraw(stalemate_board,
        chesslib.ChessPosition('C1'), chesslib.ChessPosition('D1'))
    stalemate_board = chesslib.ApplyDraw(stalemate_board, stalemate_draw)

    # test the GameState() function to detect the stalemate as tie
    state = chesslib.GameState(stalemate_board, stalemate_draw)
    assert_equal(state, chesslib.GameState_Tie)

    # create a chess board with a position of insufficient pieces for a checkmate
    insuff_board_pieces = np.array([
        chesslib.ChessPieceAtPos(chesslib.ChessPiece('W', 'K', True), chesslib.ChessPosition('E6')),
        chesslib.ChessPieceAtPos(chesslib.ChessPiece('B', 'K', False), chesslib.ChessPosition('E8')),
    ], dtype=np.uint16)
    insuff_board = chesslib.ChessBoard(insuff_board_pieces)
    insuff_draw = chesslib.ChessDraw(insuff_board,
        chesslib.ChessPosition('E8'), chesslib.ChessPosition('D8'))
    insuff_board = chesslib.ApplyDraw(insuff_board, insuff_draw)

    # test the GameState() function to detect the insufficient pieces as tie
    state = chesslib.GameState(insuff_board, insuff_draw)
    assert_equal(state, chesslib.GameState_Tie)

    # create a chess board with a simple check that can still be defended
    check_board_pieces = np.array([
        chesslib.ChessPieceAtPos(chesslib.ChessPiece('W', 'K', True), chesslib.ChessPosition('E1')),
        chesslib.ChessPieceAtPos(chesslib.ChessPiece('W', 'R', True), chesslib.ChessPosition('D2')),
        chesslib.ChessPieceAtPos(chesslib.ChessPiece('B', 'K', False), chesslib.ChessPosition('E8')),
    ], dtype=np.uint16)
    check_board = chesslib.ChessBoard(check_board_pieces)
    check_draw = chesslib.ChessDraw(check_board,
        chesslib.ChessPosition('D2'), chesslib.ChessPosition('E2'))
    check_board = chesslib.ApplyDraw(check_board, check_draw)

    # test the GameState() function to detect a simple check
    state = chesslib.GameState(check_board, check_draw)
    assert_equal(state, chesslib.GameState_Check)

    # tests with inverted piece colors, so the logic is not only working for one side

    # create a chess board with a check-mate position
    checkmate_board_pieces = np.array([
        chesslib.ChessPieceAtPos(chesslib.ChessPiece('B', 'K', False), chesslib.ChessPosition('E1')),
        chesslib.ChessPieceAtPos(chesslib.ChessPiece('B', 'R', True), chesslib.ChessPosition('H1')),
        chesslib.ChessPieceAtPos(chesslib.ChessPiece('B', 'R', True), chesslib.ChessPosition('G7')),
        chesslib.ChessPieceAtPos(chesslib.ChessPiece('W', 'K', False), chesslib.ChessPosition('E8')),
    ], dtype=np.uint16)
    checkmate_board = chesslib.ChessBoard(checkmate_board_pieces)
    checkmate_draw = chesslib.ChessDraw(checkmate_board,
        chesslib.ChessPosition('H1'), chesslib.ChessPosition('H8'))
    checkmate_board = chesslib.ApplyDraw(checkmate_board, checkmate_draw)

    # test the GameState() function to detect the mate
    state = chesslib.GameState(checkmate_board, checkmate_draw)
    assert_equal(state, chesslib.GameState_Checkmate)

    # create a chess board with a stalemate position
    stalemate_board_pieces = np.array([
        chesslib.ChessPieceAtPos(chesslib.ChessPiece('B', 'K', True), chesslib.ChessPosition('E6')),
        chesslib.ChessPieceAtPos(chesslib.ChessPiece('B', 'R', True), chesslib.ChessPosition('C1')),
        chesslib.ChessPieceAtPos(chesslib.ChessPiece('B', 'R', True), chesslib.ChessPosition('F7')),
        chesslib.ChessPieceAtPos(chesslib.ChessPiece('W', 'K', False), chesslib.ChessPosition('E8')),
    ], dtype=np.uint16)
    stalemate_board = chesslib.ChessBoard(stalemate_board_pieces)
    stalemate_draw = chesslib.ChessDraw(stalemate_board,
        chesslib.ChessPosition('C1'), chesslib.ChessPosition('D1'))
    stalemate_board = chesslib.ApplyDraw(stalemate_board, stalemate_draw)

    # test the GameState() function to detect the stalemate as tie
    state = chesslib.GameState(stalemate_board, stalemate_draw)
    assert_equal(state, chesslib.GameState_Tie)

    # create a chess board with a position of insufficient pieces for a checkmate
    insuff_board_pieces = np.array([
        chesslib.ChessPieceAtPos(chesslib.ChessPiece('B', 'K', True), chesslib.ChessPosition('E6')),
        chesslib.ChessPieceAtPos(chesslib.ChessPiece('W', 'K', False), chesslib.ChessPosition('E8')),
    ], dtype=np.uint16)
    insuff_board = chesslib.ChessBoard(insuff_board_pieces)
    insuff_draw = chesslib.ChessDraw(insuff_board,
        chesslib.ChessPosition('E8'), chesslib.ChessPosition('D8'))
    insuff_board = chesslib.ApplyDraw(insuff_board, insuff_draw)

    # test the GameState() function to detect the insufficient pieces as tie
    state = chesslib.GameState(insuff_board, insuff_draw)
    assert_equal(state, chesslib.GameState_Tie)

    # create a chess board with a simple check that can still be defended
    check_board_pieces = np.array([
        chesslib.ChessPieceAtPos(chesslib.ChessPiece('B', 'K', True), chesslib.ChessPosition('E1')),
        chesslib.ChessPieceAtPos(chesslib.ChessPiece('B', 'R', True), chesslib.ChessPosition('D2')),
        chesslib.ChessPieceAtPos(chesslib.ChessPiece('W', 'K', False), chesslib.ChessPosition('E8')),
    ], dtype=np.uint16)
    check_board = chesslib.ChessBoard(check_board_pieces)
    check_draw = chesslib.ChessDraw(check_board,
        chesslib.ChessPosition('D2'), chesslib.ChessPosition('E2'))
    check_board = chesslib.ApplyDraw(check_board, check_draw)

    # test the GameState() function to detect a simple check
    state = chesslib.GameState(check_board, check_draw)
    assert_equal(state, chesslib.GameState_Check)

    print("test passed!")


exp_board_str = \
"""   -----------------------------------------
 8 | BR | BN | BB | BQ | BK | BB | BN | BR |
   -----------------------------------------
 7 | BP | BP | BP | BP | BP | BP | BP | BP |
   -----------------------------------------
 6 |    |    |    |    |    |    |    |    |
   -----------------------------------------
 5 |    |    |    |    |    |    |    |    |
   -----------------------------------------
 4 |    |    |    |    |    |    |    |    |
   -----------------------------------------
 3 |    |    |    |    |    |    |    |    |
   -----------------------------------------
 2 | WP | WP | WP | WP | WP | WP | WP | WP |
   -----------------------------------------
 1 | WR | WN | WB | WQ | WK | WB | WN | WR |
   -----------------------------------------
     A    B    C    D    E    F    G    H"""


def test_visualize_board():

    print("testing visualize chess board")

    # generate a chess board in start formation
    board = chesslib.ChessBoard_StartFormation()
    simple_board = chesslib.ChessBoard_StartFormation(True)

    # convert the board to a printable string
    board_str = chesslib.VisualizeBoard(board=board)

    # make sure that the simple board and bitboards representations produce the same output
    assert_equal(board_str, chesslib.VisualizeBoard(simple_board, True))

    # make sure that the expected content is retrieved
    assert_equal(board_str, exp_board_str)

    print("test passed!")


def test_visualize_draw():

    print("testing visualize chess draw")

    # generate the chess draw white peasant E2-E4
    draw = 0x0118070C

    # convert the board to a printable string
    draw_str = chesslib.VisualizeDraw(draw=draw)

    # make sure that the expected content is retrieved
    assert_equal(draw_str, 'White Peasant E4-E2')

    print("test passed!")

    # TODO: add more tests for edge cases (rochade, en-passant, promotion)


def test_board_from_fen():

    # TODO: make this test work

    print("testing FEN to chess board")

    # FEN string representing the initial game state
    fen_str = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
    board_exp = chesslib.ChessBoard_StartFormation()

    # parse the FEN string and make sure it's correct
    board, context = chesslib.Board_FromFen(fen_str)
    assert_true(np.array_equal(board, board_exp))
    assert_equal(531968, context)

    # FEN string representing the game state after first draw 'white pawn F2-F4'
    fen_str = 'rnbqkbnr/pppppppp/8/8/5P2/8/PPPPP1PP/RNBQKBNR b KQkq f3 10 12'
    board_exp = chesslib.ApplyDraw(board, chesslib.ChessDraw(
        board, chesslib.ChessPosition('F2'), chesslib.ChessPosition('F4')))

    # parse the FEN string and make sure it's correct
    board, context = chesslib.Board_FromFen(fen_str)
    assert_true(np.array_equal(board, board_exp))
    assert_equal(6381121, context)

    # FEN string representing the game state after first draw 'white pawn F2-F4'
    fen_str = '5k2/8/8/8/8/8/8/3K4 b - - 41 71'
    board_exp = np.array([
        1 << chesslib.ChessPosition('D1'),
        0, 0, 0, 0, 0,
        1 << chesslib.ChessPosition('F8'),
        0, 0, 0, 0, 0,
        0xFFFF00000000FFFF # info: was_moved mask cannot be 100% accurately restored
                           #       -> draw-gen needs to be the same
    ], dtype=np.uint64)

    # parse the FEN string and make sure it's correct
    board, context = chesslib.Board_FromFen(fen_str)
    assert_true(np.array_equal(board[:12], board_exp[:12]))
    assert_equal(37560321, context)

    print("test passed!")


if __name__ == '__main__':
    test_module()
