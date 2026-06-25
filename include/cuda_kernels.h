#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H
#include "utilities.h"



// __global__ void GetRowColNums(DATA_TYPE* B, int dim)
// {
//     int row = blockIdx.y * blockDim.y + threadIdx.y;
//     int col = blockIdx.x * blockDim.x + threadIdx.x;
//     if (row < dim && col < dim)
//     {
//         int idx = row*dim+col;
//         if (row == col)
//         {
//             B[idx]=999.99;
//         }
//         else
//         {
//             B[idx]=row+col/100.;
//         }
        
//     }
// }

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


#define TILE_WIDTH 16

__global__ void matmulTiled(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, int width) {
    __shared__ DATA_TYPE tileA[TILE_WIDTH][TILE_WIDTH];   // fast scratchpad
    __shared__ DATA_TYPE tileB[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE_WIDTH + ty;
    int col = blockIdx.x * TILE_WIDTH + tx;

    DATA_TYPE sum = 0.0f;

    for (int phase = 0; phase < width / TILE_WIDTH; ++phase) {
        // 1. Load: my row fixed, column slides with phase (A walks right)
        tileA[ty][tx] = A[row * width + (phase * TILE_WIDTH + tx)];
        // my col fixed, row slides with phase (B walks down)
        tileB[ty][tx] = B[(phase * TILE_WIDTH + ty) * width + col];

        __syncthreads();   // 2. wait: whole tile loaded before anyone reads

        // 3. Partial dot product over the tile in fast shared memory
        for (int k = 0; k < TILE_WIDTH; ++k)
            sum += tileA[ty][k] * tileB[k][tx];

        __syncthreads();   // 4. wait: everyone done before tile is overwritten
    }

    C[row * width + col] = sum;
}
// __global__ void CUDA_ArraySumKernel(DATA_TYPE* array, DATA_TYPE* result, int dim)
// {
//     // My incoming dimensionality will be BLOCK_SIZE x BLOCK_SIZE threads
//     // So I can have shared memory in this block of BLOCK_SIZE elements.
//     __shared__ DATA_TYPE shared_data[BLOCK_SIZE];
    
//     // Initialize each element to zero first.
//     int thread_id = threadIdx.y;
//     shared_data[thread_id]=0;
//     __syncthreads();
    
//     // All the threads have caught up after initializing the shared memory to zero.
//     int row = blockDim.y * blockIdx.y + threadIdx.y;
//     int col = blockDim.x * blockIdx.x + threadIdx.x;
    
//     // moving across a row in the array, add the value of each column in that row to the shared_memory element of that row
//     if (row < dim && col < dim) 
//     {
//         shared_data[thread_id] += array[row * dim + col];
//     }
//     __syncthreads();
    
//     // Combine all shared_data values into first element.
//     if (thread_id==0)
//     {
//         for (int i=1; i < BLOCK_SIZE;i++)
//         {
//             shared_data[0] += shared_data[i];
//         }
//         atomicAdd(result, shared_data[0]);
//     }
//     __syncthreads();
//     // Add shared_data[0] to result via atomicAdd.
    
// }

#endif

