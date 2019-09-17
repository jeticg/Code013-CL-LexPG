#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

DIR=$DIR/..

rm -rf build
rm -rf __pycache__
rm -rf */__pycache__
rm -rf */*/__pycache__

rm -rf *.so
rm -rf */*.so
rm -rf */*/*.so

rm -rf *.c
rm -rf */*.c
rm -rf */*/*.c
