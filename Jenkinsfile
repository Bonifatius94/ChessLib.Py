node {
    checkout scm

    docker.withServer('tcp://192.168.178.166:2375') {
        docker.image('ubuntu:18.04').withRun('') {
            sh "echo Hello World!"
        }
    }
}