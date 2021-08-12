
import numpy as np
import chesslib as cl
from asserts import assert_true, assert_equal


def test():
    # test_chessposition()
    # test_chesspiece()
    # test_chesspieceatpos()
    # test_chessdraw()
    test_chessboard()


def test_chessposition():
    for i in range(10000000):
        pos = cl.ChessPosition('A1')
        assert_true(pos == 0)


def test_chesspiece():
    for i in range(10000000):
        piece = cl.ChessPiece('W', 'K', False)
        assert_true(piece == 1)


def test_chesspieceatpos():
    piece = cl.ChessPiece('W', 'K', False)
    pos = cl.ChessPosition('A1')

    for i in range(10000000):
        pieceatpos = cl.ChessPieceAtPos(piece, pos)
        assert_true(pieceatpos == 1)


def test_chessdraw():
    board = cl.ChessBoard_StartFormation()
    old_pos = cl.ChessPosition('E2')
    new_pos = cl.ChessPosition('E4')

    for i in range(10000000):
        draw = cl.ChessDraw(board, old_pos, new_pos)
        assert_true(draw == 18350876)


def test_chessboard():
    board = cl.ChessBoard_StartFormation()
    old_pos = cl.ChessPosition('E2')
    new_pos = cl.ChessPosition('E4')

    pieces_at_pos = np.array([
        cl.ChessPieceAtPos(cl.ChessPiece('W', 'K', False), cl.ChessPosition('E1')),
        cl.ChessPieceAtPos(cl.ChessPiece('B', 'K', True ), cl.ChessPosition('G8')),
        cl.ChessPieceAtPos(cl.ChessPiece('W', 'R', True ), cl.ChessPosition('A3')),
        cl.ChessPieceAtPos(cl.ChessPiece('B', 'Q', False), cl.ChessPosition('D8')),
        cl.ChessPieceAtPos(cl.ChessPiece('W', 'R', True ), cl.ChessPosition('B7')),
        cl.ChessPieceAtPos(cl.ChessPiece('B', 'B', True ), cl.ChessPosition('C5')),
    ], np.uint16)

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

    for i in range(1000000):
        board = cl.ChessBoard(pieces_at_pos)
        assert_true(np.array_equal(exp_board, board))


if __name__ == '__main__':
    test()
