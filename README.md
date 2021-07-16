![PyPi Release](https://github.com/Bonifatius94/ChessLib.Py/workflows/PyPi%20Release/badge.svg)
![Docker CI](https://github.com/Bonifatius94/ChessLib.Py/workflows/Docker%20CI/badge.svg)

# ChessLib Python Extension

## About
This project provides an efficient chess draw generation extension for Python3.
The main purpose of this project is enabling further TensorFlow AI projects and learning 
how to write an efficient Python3 extension (using good old C).

## Usage
Install the [official Python package](https://pypi.org/project/chesslib/) using pip:
```sh
pip install chesslib
```

Use the chesslib package like in the following example:
```py
import chesslib
import numpy as np
import random


def test():

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
    
    # see tests/ folder for more examples
```

## How to Develop

For a quickstart, set up your dev machine as a VM (e.g. Ubuntu 20.04 hosted by VirtualBox). After 
successfully creating the VM, use following commands to install all essential dev tools (git, 
docker, good text editor).

```sh
# install docker (e.g. Ubuntu 20.04)
sudo apt-get update && sudo apt-get install -y git docker.io
sudo usermod -aG docker $USER && reboot

# download the project's source code (if you haven't done before)
git clone https://github.com/Bonifatius94/ChessLib.Py
cd ChessLib.Py

# install the 'Visual Studio Code' text editor (optional)
sudo snap install code --classic
```

The commands for dev-testing the chesslib are wrapped within a Docker environment.
Therefore build the 'Dockerfile-dev' file which takes your source code and performs 
all required CI steps (build + install + unit tests). Afterwards you may attach to the 
Docker image with the bash console interactively and run commands, etc.

```sh
# make a dev build using the 'Dockerfile-dev' build environment
# this runs all CI steps (build + install + unit tests)
docker build . --file Dockerfile-dev -t "chesslib-dev"

# attach to the build environment's interactive bash console (leave the session with 'exit')
docker run "chesslib-dev" -it bash

# mount a Python test script into the build environment and run it
docker run -v $PWD:/scripts "chesslib-dev" python3 /scripts/test.py
```

## Copyright
This software is available under the MIT licence's terms.
