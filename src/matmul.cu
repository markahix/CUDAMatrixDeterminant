// /*
//  *  file name: matrix.cu
//  *
//  *  matrix.cu contains the code that realize some common used matrix operations in CUDA
//  *  
//  *  this is a toy program for learning CUDA, some functions are reusable in other project
//  *  
//  */
// #include <stdio.h>
// #include <iostream>
// #include <stdlib.h>
// #include <assert.h>
// #include <fstream>
// #include <string>
// #include <sstream>

// #define BLOCK_SIZE 64

// __global__ void gpu_matrix_mult(double *a, double *b, double *c, int m)
// { 
//     int row = blockIdx.y * blockDim.y + threadIdx.y; 
//     int col = blockIdx.x * blockDim.x + threadIdx.x;
//     double sum = 0;
//     if( col < m && row < m) 
//     {
//         for(int i = 0; i < m; i++) 
//         {
//             sum += a[row * m + i] * b[i * m + col];
//         }
//         c[row * m + col] = sum;
//     }
// } 

// /*
// *********************************************************************
// function name: gpu_square_matrix_mult

// description: dot product of two matrix (not only square) in GPU

// parameters: 
//             &a GPU device pointer to a n X n matrix (A)
//             &b GPU device pointer to a n X n matrix (B)
//             &c GPU device output purpose pointer to a n X n matrix (C) 
//             to store the result
// Note:
//     grid and block should be configured as:

//         dim3 dim_grid((n - 1) / BLOCK_SIZE + 1, (n - 1) / BLOCK_SIZE + 1, 1);
//         dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE, 1);

// return: none
// *********************************************************************
// */
// __global__ void gpu_square_matrix_mult(double *d_a, double *d_b, double *d_result, int n) 
// {
//     __shared__ int tile_a[BLOCK_SIZE][BLOCK_SIZE];
//     __shared__ int tile_b[BLOCK_SIZE][BLOCK_SIZE];

//     int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
//     int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
//     double tmp = 0;
//     int idx;

//     for (int sub = 0; sub < gridDim.x; ++sub) 
//     {
//         idx = row * n + sub * BLOCK_SIZE + threadIdx.x;
//         if(idx >= n*n)
//         {
//             // n may not divisible by BLOCK_SIZE
//             tile_a[threadIdx.y][threadIdx.x] = 0;
//         }
//         else
//         {
//             tile_a[threadIdx.y][threadIdx.x] = d_a[idx];
//         }

//         idx = (sub * BLOCK_SIZE + threadIdx.y) * n + col;
//         if(idx >= n*n)
//         {
//             tile_b[threadIdx.y][threadIdx.x] = 0;
//         }  
//         else
//         {
//             tile_b[threadIdx.y][threadIdx.x] = d_b[idx];
//         }
//         __syncthreads();

//         for (int k = 0; k < BLOCK_SIZE; ++k) 
//         {
//             tmp += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
//         }
//         __syncthreads();
//     }
//     if(row < n && col < n)
//     {
//         d_result[row * n + col] = tmp;
//     }
// }

// int get_matrix_dimensions(std::string filename)
// {
//     std::ifstream infile(filename,std::ios::in);
//     std::string line;
//     std::stringstream buffer;
//     double value;
//     int dimension=0;
//     getline(infile,line);
//     infile.close();
//     buffer.str(line);
//     while (buffer >> value)
//     {
//         dimension++;
//     }
//     return dimension;
// }

// void load_matrix_from_file(std::string filename, double* h_a, int dimension)
// {
//     std::ifstream infile(filename,std::ios::in);
//     std::string line;
//     std::stringstream buffer;
//     int idx=0;
//     double value;
//     if (!infile.is_open())
//     {
//         printf("Unable to open covariance matrix file.\n");
//         exit(1);
//     }
//     while(getline(infile,line))
//     {
//         buffer.str(line);
//         while(buffer >> value)
//         {
//             h_a[idx] = value;
//             idx++;
//         }
//         buffer.clear();
//     }
//     infile.close();
// }
// void print_mat(double* mat, int dimension)
// {
//     for (int i = 0; i < dimension; i++)
//     {
//         for (int j = 0; j < dimension; j++)
//         {
//             printf("%f ",mat[i*dimension+j]);
//         }
//         printf("\n");
//     }
// }

// void Build_u(double* h_a, double* h_b, int dimension)
// {
//     // the u_matrix of h_a is being put into h_b
//     int idx = 0;
//     int diag_increment = dimension + 1;
//     for (int i = 0; i < dimension; i++)
//     {
//         for (int j = 0; j < dimension; j++)
//         {
//             idx = i*dimension + j;
//             if (j < i)
//             {
//                 h_b[idx] = 0;
//             }
//             if (j > i)
//             {
//                 h_b[idx] = -h_a[idx];
//             }
//             if (j == i)
//             {
//                 h_b[idx] = 0;
//                 for (int k = idx+diag_increment; k < dimension*dimension; k+=diag_increment)
//                 {
//                     h_b[idx] += h_a[k];
//                 }
//             }
//         }
//     }
// }

// bool is_mat_det(double* h_c, int dimension)
// {
//     for (int i=1; i < dimension*dimension; i++)
//     {
//         if (h_c[i] != 0)
//         {
//             return false;
//         }
//     }
//     return true;
// }

// void algorithm_loop(double* h_a, double* h_b, double* h_c, int dimension)
// {

//     unsigned int grid_rows = (dimension + BLOCK_SIZE - 1) / BLOCK_SIZE;
//     unsigned int grid_cols = (dimension + BLOCK_SIZE - 1) / BLOCK_SIZE;
//     dim3 dimGrid(grid_cols, grid_rows);
//     dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

//     // allocate DEVICE memory
//     double *d_a;
//     double *d_b;
//     double *d_c;
//     cudaMalloc((void **) &d_a, sizeof(double)*dimension*dimension);
//     cudaMalloc((void **) &d_b, sizeof(double)*dimension*dimension);
//     cudaMalloc((void **) &d_c, sizeof(double)*dimension*dimension);

//     // copy A and B into DEVICE
//     cudaMemcpy(d_a, h_a, sizeof(double)*dimension*dimension, cudaMemcpyHostToDevice);
//     cudaMemcpy(d_b, h_b, sizeof(double)*dimension*dimension, cudaMemcpyHostToDevice);
//     cudaDeviceSynchronize();

//     gpu_matrix_mult<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, dimension);
//     cudaDeviceSynchronize();

//     // copy C from DEVICE to HOST 
//     cudaMemcpy(h_c, d_c, sizeof(double)*dimension*dimension, cudaMemcpyDeviceToHost);

//     cudaFree(d_a);
//     cudaFree(d_b);
//     cudaFree(d_c);

// }

// int main(int argc, char const *argv[])
// {
//     std::string filename = "tmp_mat.dat";
//     int m = get_matrix_dimensions(filename);

//     // allocate memory in host RAM
//     double *h_a;
//     double *h_b;
//     double *h_c;
//     cudaMallocHost((void **) &h_a, sizeof(double)*m*m);
//     cudaMallocHost((void **) &h_b, sizeof(double)*m*m);
//     cudaMallocHost((void **) &h_c, sizeof(double)*m*m);
    
//     // load Covariance Matrix into h_a
//     load_matrix_from_file(filename, h_a, m);
//     printf("Finished reading matrix file.\n\n");

//     // Build initial U_matrix
//     // u_matrix of h_a --> h_b
//     Build_u(h_a, h_b, m);
   
//     // loop time until determinant encountered.
//     int iterations = 0;
    
//     while(true) //eesh.
//     {
//         if (iterations %10 == 0) // start to count execution time of GPU version
//         {
//             printf("Beginning Iteration %d\n",iterations);
//         }
//         iterations++;

//         algorithm_loop(h_a, h_b, h_c, m);
        
//         // h_b * h_a --> h_c
//         // gpu_square_matrix_mult<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, m);    

      
//         print_mat(h_c, m);
//         std::cout << std::endl;

//         // if h_c[0] has value and everything after is a zero, break and print determinant.
//         if (is_mat_det(h_c, m))
//         {
//             break;
//         }
//         // u_matrix of h_c --> h_b
//         Build_u(h_c, h_b, m);
//     }

//     printf("Determinant of Matrix is :%f\n",h_c[0]);

//     // free memory
//     cudaFreeHost(h_a);
//     cudaFreeHost(h_b);
//     cudaFreeHost(h_c);
//     // cudaFreeHost(h_cc);
//     return 0;
// }