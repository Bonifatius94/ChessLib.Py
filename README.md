# ChessLib Python Extension

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

## Copyright
You may use this project under the MIT licence's conditions.
