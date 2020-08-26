node {
    checkout scm

    docker.withServer('tcp://192.168.178.166:2375') {
        docker.image('ubuntu:18.04').withRun('') {

            sh "echo Install Dependencies"
            sh "echo ===================="
            sh "apt-get update && apt-get install -y cmake git build-essential python3 python3-dev python3-pip"
            sh "pip3 install numpy"
            sh "rm -rf /var/lib/apt/lists/*"

            sh "echo Build CMake Project"
            sh "echo ===================="
            sh "mkdir build && cd build && cmake .. && make -j4"
        }
    }
}