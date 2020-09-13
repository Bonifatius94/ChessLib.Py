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
    #test_chesspieceatpos()
    #test_chessboard()

    # test draw-gen
    test_drawgen()

    # test conversion functions
    test_board_to_hash()


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

    # info: ignore this test as the piece at pos struct is not really required

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
                        piece_at_pos = chesslib.ChessPieceAtPos(piece, pos)

                        # make sure that the numeric value is correctly encoded
                        print(piece, pos)
                        assert_equal(piece_at_pos, pos * 16 + piece)

    print("test passed!")


def test_chessdraw_null():

    print("testing chessdraw null value")

    # test if the expected null value is returned
    assert_equal(chesslib.ChessDraw_Null, 0)

    print("test passed!")


def test_chessdraw():
    # can be ignored as draws are generated by the draw-gen function
    return 0


def test_chessboard():
    # can be ignored as the start formation is always used
    return 0


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

    print("test passed!")


def test_board_to_hash():

    print("testing board to hash function")

    # create board in start formation
    board = chesslib.ChessBoard_StartFormation()

    # compute the board's hash
    hash = bytes(chesslib.Board_Hash(board))
    #print(bytes(hash).hex())

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


# run all tests
test_module()
