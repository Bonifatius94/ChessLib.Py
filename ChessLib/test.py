
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

from asserts import assert_true, assert_equal
import chesslib
import numpy as np


def test_module():

    # test base datatypes
    test_chesscolor()
    test_chessposition()
    test_chesspiece()
    test_chessdraw_null()
    test_chessboard_start()
    test_chesspieceatpos()
    test_chessboard()

    # test gameplay functions
    test_drawgen()
    test_apply_draw()
    #test_game_state()
    test_board_to_hash()

    # test visualization functions
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
            pos = chesslib.ChessPosition(pos_string)

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
                piece = chesslib.ChessPiece(color_char, type_char, was_moved)

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
                        piece_at_pos = chesslib.ChessPieceAtPos(piece, pos)

                        # make sure that the numeric value of pieceatpos is correctly encoded
                        assert_equal(piece_at_pos, pos * 32 + piece)

    print("test passed!")


def test_chessdraw_null():

    print("testing chessdraw null value")

    # test if the expected null value is returned
    assert_equal(chesslib.ChessDraw_Null, 0)

    print("test passed!")


def test_chessdraw():
    # optional feature, can be ignored as draws are generated by the draw-gen function
    return 0


def test_chessboard():

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

    # create the chessboard from pieces at pos array
    board = chesslib.ChessBoard(pieces_at_pos)

    # define the expected board
    exp_board = [
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
    ]

    # make sure that the board is the same as expected
    for i in range(13):
        assert_equal(exp_board[i], board[i])

    print("test passed!")


def test_chessboard_start():

    print("testing chess board start formation value")

    # initialize expected bitboards array in start formation
    START_FORMATION = [
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
    ]

    # test if the expected board in start formation is returned
    start = chesslib.ChessBoard_StartFormation()
    #print(start)

    for i in range(13):
        assert_equal(start[i], START_FORMATION[i])

    print("test passed!")


def test_drawgen():

    print("testing draw-gen")

    # get all draws for starting position (white side)
    start_formation = chesslib.ChessBoard_StartFormation()
    draws = chesslib.GenerateDraws(start_formation, chesslib.ChessColor_White, chesslib.ChessDraw_Null, True)

    # define the expected draws
    expected_draws = np.array([
        18088016, 18088018, 18088341, 18088343, 18350608, 18350616, 18350673, 18350681,
        18350738, 18350746, 18350803, 18350811, 18350868, 18350876, 18350933, 18350941,
        18350998, 18351006, 18351063, 18351071],
        np.uint32
    )

    # make sure that the generated draws equal the expected draws
    assert_true(set(draws) == set(expected_draws))

    # TODO: add more unit tests that at least cover the correct parsing of all parameters

    print("test passed!")


def test_apply_draw():

    print("test applying draws to chess boards")

    # get board in start formation and opening draw 'white peasant E2-E4'
    board = chesslib.ChessBoard_StartFormation()
    draw = 0x0118070C
    #draw = chesslib.ChessDraw(board, chesslib.ChessPosition('E2'), chesslib.ChessPosition('E4'), 0)

    # try applying the draw
    new_board = chesslib.ApplyDraw(board, draw)

    # define the expected board after applying the draw
    exp_new_board = [
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
    ]

    # make sure that the new board is as expected
    for i in range(13):
        assert_equal(exp_new_board[i], new_board[i])

    # test if ApplyDraw() function is revertible
    rev_board = chesslib.ApplyDraw(new_board, draw)

    # make sure that the reverted board is the same as the original board
    for i in range(13):
        assert_equal(board[i], rev_board[i])

    print("test passed!")


def test_board_to_hash():

    print("testing board to hash function")

    # create board in start formation
    board = chesslib.ChessBoard_StartFormation()

    # compute the board's hash
    hash = bytes(chesslib.Board_Hash(board))

    # define the expected hash
    exp_hash = bytes([
        0x19, 0x48, 0x20, 0x90, 0xA3,
        0x31, 0x8C, 0x63, 0x18, 0xC6,
        0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00,
        0x73, 0x9C, 0xE7, 0x39, 0xCE,
        0x5B, 0x58, 0xA4, 0xB1, 0xAB
    ])

    # make sure that the computed hash is correct
    for i in range(40):
        assert_equal(exp_hash[i], hash[i])

    print("test passed!")


# def test_game_state():

#     print("test game state")

#     # TODO: implement this test!!!!!

#     # prepare a regular position
#     board_regular = ...
#     last_draw_regular = ...

#     # prepare a check position
#     board_check = ...
#     last_draw_check = ...

#     # prepare a checkmate position
#     board_checkmate = ...
#     last_draw_checkmate = ...

#     # prepare a tie position
#     board_tie = ...
#     last_draw_tie = ...

#     # make sure that all game states are detected correctly
#     assert_equal(chesslib.GameState_None, chesslib.GameState(board_regular, last_draw_regular))
#     assert_equal(chesslib.GameState_Check, chesslib.GameState(board_check, last_draw_check))
#     assert_equal(chesslib.GameState_Checkmate, chesslib.GameState(board_checkmate, last_draw_checkmate))
#     assert_equal(chesslib.GameState_Tie, chesslib.GameState(board_tie, last_draw_tie))

#     print("test passed!")


def test_visualize_board():

    print("testing visualize chess board")

    # generate a chess board in start formation
    board = chesslib.ChessBoard_StartFormation()

    # convert the board to a printable string
    board_str = chesslib.VisualizeBoard(board)

    # make sure that the expected content is retrieved
    exp_board_str = "   -----------------------------------------\n 8 | BR | BN | BB | BQ | BK | BB | BN | BR |\n   -----------------------------------------\n 7 | BP | BP | BP | BP | BP | BP | BP | BP |\n   -----------------------------------------\n 6 |    |    |    |    |    |    |    |    |\n   -----------------------------------------\n 5 |    |    |    |    |    |    |    |    |\n   -----------------------------------------\n 4 |    |    |    |    |    |    |    |    |\n   -----------------------------------------\n 3 |    |    |    |    |    |    |    |    |\n   -----------------------------------------\n 2 | WP | WP | WP | WP | WP | WP | WP | WP |\n   -----------------------------------------\n 1 | WR | WN | WB | WQ | WK | WB | WN | WR |\n   -----------------------------------------\n     A    B    C    D    E    F    G    H"
    assert_equal(board_str, exp_board_str)

    print("test passed!")


def test_visualize_draw():

    print("testing visualize chess draw")

    # generate a chess board in start formation
    draw = 0x0118070C

    # convert the board to a printable string
    draw_str = chesslib.VisualizeDraw(draw)

    # make sure that the expected content is retrieved
    assert_equal(draw_str, 'White Peasant E4-E2')

    print("test passed!")

    # TODO: add more tests for edge cases (rochade, en-passant, promotion)


# run all tests
test_module()
