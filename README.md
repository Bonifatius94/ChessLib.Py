# ChessLib Python Extension
![PyPi Release](https://github.com/Bonifatius94/ChessLib.Py/workflows/PyPi%20Release/badge.svg)
![Docker CI](https://github.com/Bonifatius94/ChessLib.Py/workflows/Docker%20CI/badge.svg)

## About
This project provides an efficient chess draw generation extension for Python3.
The main purpose of this project is enabling further TensorFlow AI projects and learning 
how to write an efficient Python3 extension (using good old C).

## How to Build / Test
The commands for building the Python3 extension module and testing it properly are 
wrapped as a Docker image. Therefore just build the Dockerfile and use the image
as base for your Python3 application importing the module. 

Alternatively you could run the commands from the Dockerfile onto an Ubuntu-like 
machine and build the binaries on your own. I'm using the default distutils tools,
so making your own builds should not be too hard to achieve.

```sh
# install docker (e.g. Ubuntu 18.04)
sudo apt-get update && sudo apt-get install -y git docker.io
sudo usermod -aG docker $USER && reboot

# download the project's source code
git clone https://github.com/Bonifatius94/ChessLib.Py
cd ChessLib.Py

# build the chesslib Python3 module using the commands from the Dockerfile
# this also includes running the unit tests (Docker build fails if tests don't pass)
docker build . -t "chesslib-python3:1.0"

# run a test command using the chesslib
docker run "chesslib-python3:1.0" python3 test.py
```

## Usage
Install the [official Python package](https://pypi.org/project/chesslib/) using pip:
```sh
pip install chesslib
```

The following sample outlines the usage of the ChessLib:
```py
import chesslib
import numpy as np
import random


test():

    # create a new chess board in start formation
    board = chesslib.ChessBoard_StartFormation()
    
    # generate all possible draws
    draws = chesslib.GenerateDraws(board, chesslib.ChessColor_White, chesslib.ChessDraw_Null, True)
    
    # apply one of the possible draws
    draw_to_apply = draws[random.randint(0, len(draws) - 1)]
    new_board = chesslib.ApplyDraw(board, draw_to_apply)
    
    # write the draw's name
    print(chesslib.VisualizeDraw(draw_to_apply))
    
    # visualize the board before / after applying the draw
    print(chesslib.VisualizeBoard(board))
    print(chesslib.VisualizeBoard(new_board))
    
    # revert the draw (just call ApplyDraw again with the new board)
    rev_board = chesslib.ApplyDraw(new_board, draw_to_apply)
    
    # get the board's 40-byte-hash and create a new board instance from the hash
    board_hash = chesslib.Board_ToHash(board)
    board_reloaded = chesslib.Board_FromHash(board_hash)
    
    # see ChessLib/test.py file for more examples
```

## Roadmap

Following features are planned for the near future:
- [ ] change Board_ToHash() / Board_FromHash() exchange format to Python type 'bytes' or 'bytearray' for better compatibility
- [ ] improve code coverage of unit tests
- [x] implement CI/CD GitHub pipelines for DockerHub and PyPi releases
- [ ] fix all memory leaks of the lib
- [ ] think of performence testing / performance improvements (especially draw-gen)

Following optional / fancy improvements are to be considered:
- [x] add fancy status badges, beautify README
- [ ] add API documentation compatible with common Python linters

## Copyright
You may use this project under the MIT licence's conditions.
