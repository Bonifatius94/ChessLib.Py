#!/usr/bin/env groovy

pipeline {
    
    // declare docker image as host system for builds
    agent {
        docker { image 'ubuntu:18.04' }
    }
    
    // define build/test stages
    stages {
        
        // install cmake/gcc toolchain including dev-tools for python3 and numpy
        stage('install-dependencies') {
            steps {
                sh "apt-get update && apt-get install -y cmake git build-essential python3 python3-dev python3-pip"
                sh "pip3 install numpy"
                sh "rm -rf /var/lib/apt/lists/*"
            }
        }
        
        // build cmake project
        stage('build') {
            steps {
                sh "mkdir build && cd build && cmake .. && make -j4"
            }
        }
        
        // run cmake unit-tests
        stage('test') {
            steps {
                sh "echo 'TODO: implement cmake unit tests'"
            }
        }
        
        // create a release package
        stage('package') {
            steps {
                sh "echo 'TODO: implement packaging the cmake build output'"
            }
        }
    }
}