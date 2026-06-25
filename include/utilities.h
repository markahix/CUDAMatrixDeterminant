#ifndef UTILITIES_H
#define UTILITIES_H

#include <stdio.h>
#include <stdlib.h> // For malloc and free
#include <cuda_runtime.h> // For CUDA API functions
#include <string>
#include <iostream>
#include <assert.h>
#include <fstream>
#include <sstream>
#include <cmath>

#define BLOCK_SIZE 32
typedef float DATA_TYPE;  //change only this to mess with all subsequent array and single-value datatypes.

// CUDA error check macro
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error at %s:%d: %s %s\\n", file, line, func, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Function to print a matrix (for small matrices)
void printMatrix(float* matrix, int rows, int cols) {
    if (rows > 10 || cols > 10) { // Limit printing for large matrices
        printf("Matrix too large to print.\n");
        for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {
            printf("%.2f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
        return;
    }
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%.2f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

int get_matrix_dimensions(std::string filename)
{
    std::ifstream infile(filename,std::ios::in);
    std::string line;
    std::stringstream buffer;
    double value;
    int dimension=0;
    getline(infile,line);
    infile.close();
    buffer.str(line);
    while (buffer >> value)
    {
        dimension++;
    }
    return dimension;
}

void load_matrix_from_file(std::string filename, DATA_TYPE* h_a, int dimension)
{
    std::ifstream infile(filename,std::ios::in);
    std::string line;
    std::stringstream buffer;
    int idx=0;
    DATA_TYPE value;
    if (!infile.is_open())
    {
        printf("Unable to open covariance matrix file.\n");
        exit(1);
    }
    while(getline(infile,line))
    {
        buffer.str(line);
        while(buffer >> value)
        {
            h_a[idx] = value;
            idx++;
        }
        buffer.clear();
    }
    infile.close();
}

#endif