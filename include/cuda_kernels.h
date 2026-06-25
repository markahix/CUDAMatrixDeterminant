#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H
#include "utilities.h"

__global__ void initialize_matrix(DATA_TYPE* A,int dim)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < dim && col < dim) 
    {
        A[row*dim+col] = 0.0;
    }
    __syncthreads();
}

__global__ void matrixMulKernel(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, int dim) //square matrices assumed here.
{
    // Calculate the row and column of the C element to be computed by this thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if the thread is within the bounds of the output matrix C
    if (row < dim && col < dim) 
    {
        DATA_TYPE sum = 0.0;
        for (int i = 0; i < dim; i++) 
        {
            sum += A[row * dim + i] * B[i * dim + col];
        }
        C[row * dim + col] = sum;
    }
    __syncthreads();
}

__global__ void BuildUMatrixKernel(DATA_TYPE* A, DATA_TYPE* B, int dim)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < dim && col < dim) 
    {
        int idx = row*dim + col;
        if (row > col)
        {
            B[idx] = 0;
        }
        if (row < col)
        {
            B[idx] = -A[idx];
        }
        if (row == col)
        {
            DATA_TYPE sum = 0.0;
            for (int i = row+1; i < dim; i++)
            {
                sum+= A[i*dim+i];
            }
            B[idx] = sum;
        }
    }
    __syncthreads();
}

__global__ void AdditiveMatrix(DATA_TYPE* vA, DATA_TYPE* vB, DATA_TYPE* mC, int dim)
{
    //vA is the target row from the original matrix, vB is the multipliers.
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < dim && col < dim)
    {
        mC[row * dim + col] += vA[col] * vB[row];
    }
}


void TriangularizeMatrix(DATA_TYPE* A, int dim)
{
    size_t sizeM = dim*dim*sizeof(DATA_TYPE);
    size_t sizeV = dim*sizeof(DATA_TYPE);
    
    DATA_TYPE* multipliers, *additive_matrix, *target_row;
    DATA_TYPE* vA, *vB;

    // allocate memory on host and device.
    vA = (DATA_TYPE*)malloc(sizeV);
    vB = (DATA_TYPE*)malloc(sizeV);
    cudaMalloc((void**)&multipliers, sizeV);
    cudaMalloc((void**)&additive_matrix, sizeM);
    cudaMalloc((void**)&target_row, sizeV);
    cudaMemcpy(additive_matrix,A,sizeM,cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(
        (dim + dimBlock.x - 1) / dimBlock.x, // Grid width (cols of C)
        (dim + dimBlock.y - 1) / dimBlock.y  // Grid height (rows of C)
    );

    // each iteration should work through one row, moving it downward.
    for (int i=0; i < dim - 1; i++)
    {
        // iterate down the matrix by row.
        // fill target_row with current row.
        for(int j=0; j < dim; j++)
        {
            vB[j] = A[i*dim+j];
        }

        // fill the values of multipliers with 0.0 up to the current index
        for (int j = 0; j <= i; j++)
        {
            vA[j] = 0.0;
        }

        for (int j = i+1; j < dim; j++)
        {
            // generate multipliers for each row n after i.
            vA[j] = -A[j*dim + i]/A[i*dim+i];
        }
        // generate additive matrix from the target row and the multipliers.
        // // print target_row
        // std::cout << "target row"<<std::endl;
        // for (int q = 0; q < dim; q++)
        // {
        //     std::cout << vB[q]<< " ";
        // }
        // std::cout << std::endl << std::endl;
        // std::cout << "multipliers"<<std::endl;
        // for (int q = 0; q < dim; q++)
        // {
        //     std::cout << vA[q]<< " ";
        // }
        // std::cout << std::endl << std::endl;

        // print multipliers


        cudaMemcpy(target_row,vB,sizeV,cudaMemcpyHostToDevice);
        cudaMemcpy(multipliers,vA,sizeV,cudaMemcpyHostToDevice);
        AdditiveMatrix<<<dimGrid, dimBlock>>>(target_row, multipliers, additive_matrix, dim);
        cudaMemcpy(A,additive_matrix,sizeM,cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
    }
    
}

#endif

