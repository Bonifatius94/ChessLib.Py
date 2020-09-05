from asserts import assert_true, assert_equal
import chesslib


def test_module():

    # test base datatypes
    test_chesscolor()
    test_chessposition()
    test_chesspiece()
    test_chesspieceatpos()
    test_chessdraw()
    test_chessboard()

    # test draw-gen
    test_drawgen()


def test_chesscolor():

    # test white side (expected integer with value=0)
    white_side = chesslib.ChessColor_White()
    print(white_side)
    assert_equal(white_side, 0)

    # test black side (expected integer with value=1)
    black_side = chesslib.ChessColor_Black()
    print(black_side)
    assert_equal(black_side, 1)


def test_chessposition():

    # create column char mappings
    col_chars = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

    # loop through all rows and columns
    for row in range(8):
        for col in range(8):
            # test the creation of a new chessposition of the given (row, column) tuple
            pos_string = col_chars[col] + str(row + 1)
            pos = chesslib.ChessPosition(pos_string)
            print(pos_string, pos)

            # make sure that the position's numeric value is equal to the expected index
            assert_equal(pos, row * 8 + col)


def test_chesspiece():
    return 0


def test_chesspieceatpos():
    return 0


def test_chessdraw():
    return 0


def test_chessboard():
    return 0


def test_drawgen():

    # get all draws for starting position (white side)
    #start_formation = chesslib.ChessBoard_StartFormation()
    #drawing_side = chesslib.ChessColor_White()
    #last_draw = chesslib.ChessDraw_Null()
    #print(start_formation, drawing_side, last_draw)

    #draws = chesslib.GenerateDraws(start_formation, drawing_side, last_draw, True)
    #print(draws)

    return 0


test_module()
