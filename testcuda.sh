#!/bin/bash
nvcc -Iinclude src/main.cu && time ./a.out testdata/tmp_mat.dat


