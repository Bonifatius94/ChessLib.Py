import chesslib


def test_drawgen():

    # get all draws for starting position (white side)
    start_formation = chesslib.ChessBoard_StartFormation
    drawing_side = chesslib.ChessColor_White
    last_draw = chesslib.ChessDraw_Null
    print(start_formation, drawing_side, last_draw)

    draws = chesslib.GenerateDraws(start_formation, drawing_side, last_draw, True)
    #print(draws)

test_drawgen()
