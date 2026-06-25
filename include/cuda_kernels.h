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

__global__ void GenerateMultipliers(DATA_TYPE* A, DATA_TYPE* multipliers, int dim, int i)
{
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    if (row << dim)
    {
        if (row <=i)
        {
            multipliers[row] = 0.0;
        }
        else
        {
            multipliers[row] = -A[row*dim + i]/A[i*dim+i];
        }
    }
}

__global__ void GenerateTargetRow(DATA_TYPE* A, DATA_TYPE* target_row, int dim, int i)
{
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    if (col << dim)
    {
        target_row[col] = A[i*dim + col];
    }
}



void TriangularizeMatrix(DATA_TYPE* A, int dim)
{
    size_t sizeM = dim*dim*sizeof(DATA_TYPE);
    size_t sizeV = dim*sizeof(DATA_TYPE);
    
    DATA_TYPE* multipliers, *additive_matrix, *target_row;
    // DATA_TYPE* vA, *vB;

    // allocate memory on host and device.
    // vA = (DATA_TYPE*)malloc(sizeV);
    // vB = (DATA_TYPE*)malloc(sizeV);
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
        if (i%100 == 0)
        {
            std::cout << "Iteration "<< i <<std::endl;
        }

        // fill target_row with current row.
        GenerateTargetRow<<<dimGrid, dimBlock>>>(additive_matrix, target_row, dim, i);
        cudaDeviceSynchronize();

        // fill the values of multipliers with 0.0 up to the current index
        GenerateMultipliers<<<dimGrid, dimBlock>>>(additive_matrix, multipliers, dim, i);
        cudaDeviceSynchronize();

        // cudaMemcpy(target_row,vB,sizeV,cudaMemcpyHostToDevice);
        // cudaMemcpy(multipliers,vA,sizeV,cudaMemcpyHostToDevice);
        AdditiveMatrix<<<dimGrid, dimBlock>>>(target_row, multipliers, additive_matrix, dim);
        cudaDeviceSynchronize();

        // cudaMemcpy(A,additive_matrix,sizeM,cudaMemcpyDeviceToHost);
    }
    cudaMemcpy(A,additive_matrix,sizeM,cudaMemcpyDeviceToHost);
    cudaFree(additive_matrix);
    cudaFree(multipliers);
    cudaFree(target_row);
}

#endif

