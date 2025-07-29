#!/usr/bin/bash

make clean
make

# compute-sanitizer --tool=memcheck ./a.out
./a.out