#!/usr/bin/env bash

module load gnu7
cd build && CC=gcc CXX=g++ cmake3 -DCMAKE_BUILD_DIR=Debug .. && make

