#!/bin/bash
nvcc -Iinclude src/main.cu -o bin/matrixdet
rm test.txt
bin/matrixdet testdata/tmp_mat.dat > test.txt

# echo "Now for the big matrix... \n\n" >> test.txt

# ./a.out testdata/covarmat.dat
