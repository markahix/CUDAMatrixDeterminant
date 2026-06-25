#!/bin/bash
nvcc -Iinclude src/main.cu -o bin/matrixdet


time bin/matrixdet testdata/tmp_mat_20.dat > test.txt
time bin/matrixdet testdata/tmp_mat_100.dat >> test.txt
time bin/matrixdet testdata/covarmat.dat >> test.txt
# echo "Now for the big matrix... \n\n" >> test.txt

# ./a.out testdata/covarmat.dat
