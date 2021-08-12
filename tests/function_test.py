
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
import chesslib as cl
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

    # test serialization functions
    test_board_hash()

    # # test visualization functions
    test_visualize_board()
    test_visualize_draw()


def test_chesscolor():

    print("testing chess color struct")

    # test white side (expected integer with value=0)
    white_side = cl.ChessColor_White
    assert_equal(white_side, 0)

    # test black side (expected integer with value=1)
    black_side = cl.ChessColor_Black
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
            pos = cl.ChessPosition(pos_string)

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
                piece = cl.ChessPiece(color_char, type_char, was_moved)

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
            pos = cl.ChessPosition(pos_string)

            # loop through all possible pieces
            for type in range(1, 7):
                for was_moved in range(2):
                    for color in range(2):
                        # test the creation of the new chesspiece
                        color_char = 'W' if color == 0 else 'B'
                        type_char = piece_type_chars[type]
                        piece = cl.ChessPiece(color_char, type_char, was_moved)

                        # test the creation of a pieceatpos struct with the given piece and pos
                        piece_at_pos = cl.ChessPieceAtPos(piece, pos)

                        # make sure that the numeric value of pieceatpos is correctly encoded
                        assert_equal(piece_at_pos, pos * 32 + piece)

    print("test passed!")


def test_chessdraw_null():

    print("testing chessdraw null value")

    # test if the expected null value is returned
    assert_equal(cl.ChessDraw_Null, 0)

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
    board = cl.ChessBoard_StartFormation()
    gen_draw = cl.ChessDraw(board, cl.ChessPosition('E2'), cl.ChessPosition('E4'))
    assert_equal(gen_draw, exp_draw)
    assert_equal(sys.getrefcount(board), 2)

    # compact creation, draw E2-E4 from start formation, board in bitboards format
    # board = cl.ChessBoard_StartFormation()
    gen_draw = cl.ChessDraw(board, cl.ChessPosition('E2'), cl.ChessPosition('E4'), 0, True)
    assert_equal(gen_draw, exp_compact_draw)
    assert_equal(sys.getrefcount(board), 2)

    # standard creation, draw E2-E4 from start formation, board in simple format
    board = cl.ChessBoard_StartFormation(True)
    gen_draw = cl.ChessDraw(board, cl.ChessPosition('E2'), cl.ChessPosition('E4'), 0, False, True)
    assert_equal(gen_draw, exp_draw)
    assert_equal(sys.getrefcount(board), 2)

    # compact creation, draw E2-E4 from start formation, board in simple format
    # board = cl.ChessBoard_StartFormation(True)
    gen_draw = cl.ChessDraw(board, cl.ChessPosition('E2'), cl.ChessPosition('E4'), 0, True, True)
    assert_equal(gen_draw, exp_compact_draw)
    print(gen_draw, exp_compact_draw)
    assert_equal(sys.getrefcount(board), 2)

    # TODO: add a peasant prom. test case

    print("test passed!")


def test_create_chessboard():

    print("test chessboard creation from pieceatpos array")

    # define the pieces to be put onto the board at the given positions
    pieces_at_pos = np.array([
        cl.ChessPieceAtPos(cl.ChessPiece('W', 'K', False), cl.ChessPosition('E1')),
        cl.ChessPieceAtPos(cl.ChessPiece('B', 'K', True ), cl.ChessPosition('G8')),
        cl.ChessPieceAtPos(cl.ChessPiece('W', 'R', True ), cl.ChessPosition('A3')),
        cl.ChessPieceAtPos(cl.ChessPiece('B', 'Q', False), cl.ChessPosition('D8')),
        cl.ChessPieceAtPos(cl.ChessPiece('W', 'R', True ), cl.ChessPosition('B7')),
        cl.ChessPieceAtPos(cl.ChessPiece('B', 'B', True ), cl.ChessPosition('C5')),
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
    board = cl.ChessBoard(pieces_at_pos)
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
    simple_board = cl.ChessBoard(pieces_at_pos, True)
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
    start = cl.ChessBoard_StartFormation()
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
    simple_start = cl.ChessBoard_StartFormation(True)
    assert_true(np.array_equal(simple_start, exp_simple_start_formation))

    print("test passed!")


def test_drawgen():

    print("testing draw-gen")

    # get all draws for starting position (white side)
    board = cl.ChessBoard_StartFormation()
    simple_board = cl.ChessBoard_StartFormation(True)
    draws = cl.GenerateDraws(board, cl.ChessColor_White, cl.ChessDraw_Null, True)

    # make sure that simple board and bitboards representation produce the same output
    assert_true(set(draws) == set(cl.GenerateDraws(simple_board, cl.ChessColor_White, cl.ChessDraw_Null, True, False, True)))

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
    comp_draws = cl.GenerateDraws(board, cl.ChessColor_White, cl.ChessDraw_Null, True, True)

    # make sure that simple board and bitboards representation produce the same output
    assert_true(set(comp_draws) == set(cl.GenerateDraws(simple_board, cl.ChessColor_White, cl.ChessDraw_Null, True, True, True)))

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
    board = cl.ChessBoard_StartFormation()
    simple_board = cl.ChessBoard_StartFormation(True)
    draw = 0x0118070C

    # try applying the draw
    board_after = cl.ApplyDraw(board, draw)
    simple_board_after = cl.ApplyDraw(simple_board, draw, True)

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
    rev_board = cl.ApplyDraw(board_after, draw)
    simple_rev_board = cl.ApplyDraw(simple_board_after, draw, True)

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
    board = cl.ChessBoard_StartFormation()
    simple_board = cl.ChessBoard_StartFormation(True)

    # make sure that simple and bitboards representations create the same hash
    assert_true(np.array_equal(cl.Board_ToHash(board), cl.Board_ToHash(simple_board, True)))

    # compute the board's 40-byte hash
    hash = cl.Board_ToHash(board)
    assert_equal(sys.getrefcount(board), 2)
    hash = cl.Board_ToHash(simple_board, True)
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
    board_copy = cl.Board_FromHash(hash)

    # make sure that the board converted from 40-byte hash is the same as the original board
    assert_true(np.array_equal(board_copy, board))
    assert_equal(sys.getrefcount(hash), 2)

    # convert the board's 40-byte hash back to a simple board instance
    simple_board_copy = cl.Board_FromHash(hash, True)

    # make sure that the board converted from 40-byte hash is the same as the original simple board
    assert_true(np.array_equal(simple_board_copy, simple_board))
    assert_equal(sys.getrefcount(hash), 2)

    # TODO: figure out why np.frombuffer(hash) causes a segfault when reading ndarray bytes

    print("test passed!")


def test_game_state():

    print("testing game state function")

    # create a chess board with a check-mate position
    checkmate_board_pieces = np.array([
        cl.ChessPieceAtPos(cl.ChessPiece('W', 'K', False), cl.ChessPosition('E1')),
        cl.ChessPieceAtPos(cl.ChessPiece('W', 'R', True), cl.ChessPosition('H1')),
        cl.ChessPieceAtPos(cl.ChessPiece('W', 'R', True), cl.ChessPosition('G7')),
        cl.ChessPieceAtPos(cl.ChessPiece('B', 'K', False), cl.ChessPosition('E8')),
    ], dtype=np.uint16)
    checkmate_board = cl.ChessBoard(checkmate_board_pieces)
    checkmate_draw = cl.ChessDraw(checkmate_board,
        cl.ChessPosition('H1'), cl.ChessPosition('H8'))
    checkmate_board = cl.ApplyDraw(checkmate_board, checkmate_draw)

    # test the GameState() function to detect the mate
    state = cl.GameState(checkmate_board, checkmate_draw)
    assert_equal(state, cl.GameState_Checkmate)

    # create a chess board with a stalemate position
    stalemate_board_pieces = np.array([
        cl.ChessPieceAtPos(cl.ChessPiece('W', 'K', True), cl.ChessPosition('E6')),
        cl.ChessPieceAtPos(cl.ChessPiece('W', 'R', True), cl.ChessPosition('C1')),
        cl.ChessPieceAtPos(cl.ChessPiece('W', 'R', True), cl.ChessPosition('F7')),
        cl.ChessPieceAtPos(cl.ChessPiece('B', 'K', False), cl.ChessPosition('E8')),
    ], dtype=np.uint16)
    stalemate_board = cl.ChessBoard(stalemate_board_pieces)
    stalemate_draw = cl.ChessDraw(stalemate_board,
        cl.ChessPosition('C1'), cl.ChessPosition('D1'))
    stalemate_board = cl.ApplyDraw(stalemate_board, stalemate_draw)

    # test the GameState() function to detect the stalemate as tie
    state = cl.GameState(stalemate_board, stalemate_draw)
    assert_equal(state, cl.GameState_Tie)

    # create a chess board with a position of insufficient pieces for a checkmate
    insuff_board_pieces = np.array([
        cl.ChessPieceAtPos(cl.ChessPiece('W', 'K', True), cl.ChessPosition('E6')),
        cl.ChessPieceAtPos(cl.ChessPiece('B', 'K', False), cl.ChessPosition('E8')),
    ], dtype=np.uint16)
    insuff_board = cl.ChessBoard(insuff_board_pieces)
    insuff_draw = cl.ChessDraw(insuff_board,
        cl.ChessPosition('E8'), cl.ChessPosition('D8'))
    insuff_board = cl.ApplyDraw(insuff_board, insuff_draw)

    # test the GameState() function to detect the insufficient pieces as tie
    state = cl.GameState(insuff_board, insuff_draw)
    assert_equal(state, cl.GameState_Tie)

    # create a chess board with a simple check that can still be defended
    check_board_pieces = np.array([
        cl.ChessPieceAtPos(cl.ChessPiece('W', 'K', True), cl.ChessPosition('E1')),
        cl.ChessPieceAtPos(cl.ChessPiece('W', 'R', True), cl.ChessPosition('D2')),
        cl.ChessPieceAtPos(cl.ChessPiece('B', 'K', False), cl.ChessPosition('E8')),
    ], dtype=np.uint16)
    check_board = cl.ChessBoard(check_board_pieces)
    check_draw = cl.ChessDraw(check_board,
        cl.ChessPosition('D2'), cl.ChessPosition('E2'))
    check_board = cl.ApplyDraw(check_board, check_draw)

    # test the GameState() function to detect a simple check
    state = cl.GameState(check_board, check_draw)
    assert_equal(state, cl.GameState_Check)

    # tests with inverted piece colors, so the logic is not only working for one side

    # create a chess board with a check-mate position
    checkmate_board_pieces = np.array([
        cl.ChessPieceAtPos(cl.ChessPiece('B', 'K', False), cl.ChessPosition('E1')),
        cl.ChessPieceAtPos(cl.ChessPiece('B', 'R', True), cl.ChessPosition('H1')),
        cl.ChessPieceAtPos(cl.ChessPiece('B', 'R', True), cl.ChessPosition('G7')),
        cl.ChessPieceAtPos(cl.ChessPiece('W', 'K', False), cl.ChessPosition('E8')),
    ], dtype=np.uint16)
    checkmate_board = cl.ChessBoard(checkmate_board_pieces)
    checkmate_draw = cl.ChessDraw(checkmate_board,
        cl.ChessPosition('H1'), cl.ChessPosition('H8'))
    checkmate_board = cl.ApplyDraw(checkmate_board, checkmate_draw)

    # test the GameState() function to detect the mate
    state = cl.GameState(checkmate_board, checkmate_draw)
    assert_equal(state, cl.GameState_Checkmate)

    # create a chess board with a stalemate position
    stalemate_board_pieces = np.array([
        cl.ChessPieceAtPos(cl.ChessPiece('B', 'K', True), cl.ChessPosition('E6')),
        cl.ChessPieceAtPos(cl.ChessPiece('B', 'R', True), cl.ChessPosition('C1')),
        cl.ChessPieceAtPos(cl.ChessPiece('B', 'R', True), cl.ChessPosition('F7')),
        cl.ChessPieceAtPos(cl.ChessPiece('W', 'K', False), cl.ChessPosition('E8')),
    ], dtype=np.uint16)
    stalemate_board = cl.ChessBoard(stalemate_board_pieces)
    stalemate_draw = cl.ChessDraw(stalemate_board,
        cl.ChessPosition('C1'), cl.ChessPosition('D1'))
    stalemate_board = cl.ApplyDraw(stalemate_board, stalemate_draw)

    # test the GameState() function to detect the stalemate as tie
    state = cl.GameState(stalemate_board, stalemate_draw)
    assert_equal(state, cl.GameState_Tie)

    # create a chess board with a position of insufficient pieces for a checkmate
    insuff_board_pieces = np.array([
        cl.ChessPieceAtPos(cl.ChessPiece('B', 'K', True), cl.ChessPosition('E6')),
        cl.ChessPieceAtPos(cl.ChessPiece('W', 'K', False), cl.ChessPosition('E8')),
    ], dtype=np.uint16)
    insuff_board = cl.ChessBoard(insuff_board_pieces)
    insuff_draw = cl.ChessDraw(insuff_board,
        cl.ChessPosition('E8'), cl.ChessPosition('D8'))
    insuff_board = cl.ApplyDraw(insuff_board, insuff_draw)

    # test the GameState() function to detect the insufficient pieces as tie
    state = cl.GameState(insuff_board, insuff_draw)
    assert_equal(state, cl.GameState_Tie)

    # create a chess board with a simple check that can still be defended
    check_board_pieces = np.array([
        cl.ChessPieceAtPos(cl.ChessPiece('B', 'K', True), cl.ChessPosition('E1')),
        cl.ChessPieceAtPos(cl.ChessPiece('B', 'R', True), cl.ChessPosition('D2')),
        cl.ChessPieceAtPos(cl.ChessPiece('W', 'K', False), cl.ChessPosition('E8')),
    ], dtype=np.uint16)
    check_board = cl.ChessBoard(check_board_pieces)
    check_draw = cl.ChessDraw(check_board,
        cl.ChessPosition('D2'), cl.ChessPosition('E2'))
    check_board = cl.ApplyDraw(check_board, check_draw)

    # test the GameState() function to detect a simple check
    state = cl.GameState(check_board, check_draw)
    assert_equal(state, cl.GameState_Check)

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
    board = cl.ChessBoard_StartFormation()
    simple_board = cl.ChessBoard_StartFormation(True)

    # convert the board to a printable string
    board_str = cl.VisualizeBoard(board)

    # make sure that the simple board and bitboards representations produce the same output
    assert_equal(board_str, cl.VisualizeBoard(simple_board, True))

    # make sure that the expected content is retrieved
    assert_equal(board_str, exp_board_str)

    print("test passed!")


def test_visualize_draw():

    print("testing visualize chess draw")

    # generate the chess draw white peasant E2-E4
    draw = 0x0118070C

    # convert the board to a printable string
    draw_str = cl.VisualizeDraw(draw)

    # make sure that the expected content is retrieved
    assert_equal(draw_str, 'White Peasant E4-E2')

    print("test passed!")

    # TODO: add more tests for edge cases (rochade, en-passant, promotion)


if __name__ == '__main__':
    test_module()
