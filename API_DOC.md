
# ChessLib API Documentation

## API Overview

### Data Types

| Type             | Description                                                                           |
| ---------------- | ------------------------------------------------------------------------------------- |
| ChessColor       | An enum with types white=0, black=1.                                                  |
| ChessPieceType   | An enum with types invalid=0, king=1, queen=2, rook=3, bishop=4, knight=5 and pawn=6. |
| ChessDrawType    | An enum with types standard=0, rochade=1, enpassant=2, PeasantProm=3.                 |
| ChessGameState   | An enum with types none='N', check='C', mate='M', tie='T'.                            |
| ChessPosition    | Bitboard index within [0, 63]. Row-wise indexing: 0=A1, 1=B1, ..., 8=A2, ..., 63=H8.  |
| ChessPiece       | Tuple of ChessPieceType, ChessColor and was_moved flag, bitwise aligned as 5 bits.    |
| ChessPieceAtPos  | Short integer combining ChessPieceType and ChessPosition, bitwise aligned as 11 bits. |
| CompactChessDraw | Tuple of OldPos, NewPos and PromType, bitwise aligned as 15 bits.                     |
| ChessDraw        | Extends CompactChessDraw with all additional context info, used for applying draws.   |
| Bitboard         | Indicates if a piece is standing at a position / was moved (for all 64 positions).    |
| ChessBoard       | A flat NumPy array consisting of 13 bitboards with dtype=np.uint64.                   |
| SimpleChessBoard | A flat NumPy array consisting of 64 chess pieces with dtype=np.uint8.                 |
| BoardHash        | A flat NumPy array compressing a SimpleChessBoard to 40 bytes with dtype=np.uint8.    |
| ChessPieceList   | A flat NumPy array of ChessPieceAtPos objects with dtype=np.uint16 / dtype=np.int16.  |

See the [chess types](./chesslib/include/chesstypes.h) header file for more detailed information.

### API Constants

| Type                | Description                                                             |
| ------------------- | ----------------------------------------------------------------------- |
| version             | The chesslib package version installed (as string).                     |
| ChessColor_White    | Representing the ChessColor enum's state white=0.                       |
| ChessColor_Black    | Representing the ChessColor enum's state black=1.                       |
| ChessDraw_Null      | Representing the null-value for ChessDraw equal to the numeric value 0. |
| GameState_None      | Representing the ChessGameState enum's state none='N'.                  |
| GameState_Check     | Representing the ChessGameState enum's state check='C'.                 |
| GameState_Checkmate | Representing the ChessGameState enum's state checkmate='M'.             |
| GameState_Tie       | Representing the ChessGameState enum's state tie='T'.                   |

See the [chesslib module](./chesslib/src/chesslibmodule.c) code file for more detailed information.

### API Functions

| Type                      | Description                                                                                    |
| ------------------------- | ---------------------------------------------------------------------------------------------- |
| ChessPosition             | Transform the literal chess position representation (e.g. 'A5' or 'H8') to a bitboard index.   |
| ChessPiece                | Create a chess piece given its color, type and whether it was already moved.                   |
| ChessPieceAtPos           | Attach positional information to a chess piece.                                                |
| ChessDraw                 | Create a chess draw given a chess board and the old/new position of the piece to be moved.     |
| ChessBoard                | Create a chess board given a ChessPieceList, use is_simple_board=True for a SimpleChessBoard.  |
| ChessBoard_StartFormation | Create a chess board in start formation, use is_simple_board=True for a SimpleChessBoard.      |
| GenerateDraws             | Generate all possible draws for the given chess board and drawing side.                        |
| ApplyDraw                 | Apply the draw to the given chess board and return the resulting chess board as new instance.  |
| GameState                 | Determine the current game state given the chess board and the last draw made.                 |
| Board_ToHash              | Compress the given chess board to a 40-byte hash.                                              |
| Board_FromHash            | Uncompress a chess board from a 40-byte hash, use is_simple_board=True for a SimpleChessBoard. |
| VisualizeBoard            | Visualize the given chess board as a human-readable textual representation.                    |
| VisualizeDraw             | Visualize the given chess draw as a human-readable textual representation.                     |

See the [chesslib module](./chesslib/src/chesslibmodule.c) code file for more detailed information.

## Detailed Information
TODO: add a more detailed info section outlining usage examples etc.
