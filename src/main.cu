// CUDA Matrix Multiplication
#include "cuda_kernels.h"

void algorithm_loop(DATA_TYPE* h_a, DATA_TYPE* h_c, int dim)
{
    size_t sizeA = dim * dim * sizeof(DATA_TYPE);
    
    // Allocate memory on device (GPU)
    DATA_TYPE *d_A, *d_B, *d_C, *host_umatrix;
    cudaMalloc((void**)&d_A, sizeA);
    cudaMalloc((void**)&d_B, sizeA);
    cudaMalloc((void**)&d_C, sizeA);
    host_umatrix = (DATA_TYPE*)malloc(sizeA);
    
    // Copy input matrices from host to device
    printf("Copying data from host to device...\n");
    cudaMemcpy(d_A, h_a, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_a, sizeA, cudaMemcpyHostToDevice); 
        // original matrix copied into d_C to make the loop below easier and more efficient
    

    // Define grid and block dimensions for the kernel launch
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(
        (dim + dimBlock.x - 1) / dimBlock.x, // Grid width (cols of C)
        (dim + dimBlock.y - 1) / dimBlock.y  // Grid height (rows of C)
    );

    // GetRowColNums<<<dimGrid, dimBlock>>>(d_A,d_B,dim);
    // cudaMemcpy(host_umatrix, d_B, sizeA, cudaMemcpyDeviceToHost); // checking the status of UMatrix at each iteration.
    // printf("Current RowCols:\n");
    // printMatrix(host_umatrix, dim, dim);
    // printf("\n");


    for (int i = 0; i < dim; i++)
    {
        // Build UMatrix from d_C (first run, d_C is copy of d_A)
        BuildUMatrixKernel<<<dimGrid, dimBlock>>>(d_C, d_B, dim);

        // if (i%100 == 0)
        // {
            std::cout << "Beginning iteration " << i << ". " << std::endl;
            cudaMemcpy(host_umatrix, d_B, sizeA, cudaMemcpyDeviceToHost); // checking the status of UMatrix at each iteration.
            printf("Current UMatrix:\n");
            printMatrix(host_umatrix, dim, dim);
            printf("\n");
        // }

        // Matrix Multiplication of d_A and d_B into d_C.
        matrixMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, dim); // multiply d_A and d_B, storing result in d_C.
        cudaMemcpy(host_umatrix, d_C, sizeA, cudaMemcpyDeviceToHost); // checking the status of UMatrix at each iteration.
        printf("Current C_Matrix:\n");
        printMatrix(host_umatrix, dim, dim);
        printf("\n");
        cudaDeviceSynchronize();
    }
    
    // Step 7: Copy the result matrix from device to host
    printf("Copying result from device to host...\n");
    cudaMemcpy(h_c, d_C, sizeA, cudaMemcpyDeviceToHost);

    // Step 9: Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    printf("Freed DEVICE memory.\n");
    printf("Result of Matrix Multiplication inside algorithm loop:\n");
    printMatrix(h_c, dim, dim);
    printf("\n");

}

int main(int argc, char** argv) {

    std::string filename = argv[1];
    // Step 1: Allocate memory on host (CPU)
    DATA_TYPE *h_A, *h_C;
    int M = get_matrix_dimensions(filename);
    size_t sizeA = M * M * sizeof(DATA_TYPE);


    h_A = (DATA_TYPE*)malloc(sizeA);
    h_C = (DATA_TYPE*)malloc(sizeA);

    if (h_A == NULL || h_C == NULL) {
        fprintf(stderr, "Failed to allocate host memory!\n");
        exit(EXIT_FAILURE);
    }

    // Step 2: Initialize host matrices
    load_matrix_from_file(filename, h_A, M);
    
    // Optional: Print small matrices
    printf("Matrix A:\n");
    printMatrix(h_A, M, M);
    printf("\n");

    algorithm_loop(h_A, h_C, M);

    // Step 8: Print the result (optional for small matrices)
    printf("Result of Matrix Multiplication:\n");
    printMatrix(h_C, M, M);
    printf("\n");

    // Step 10: Free host memory
    free(h_A);
    free(h_C);

    printf("Matrix multiplication completed and memory freed.\n");
    return 0;
}