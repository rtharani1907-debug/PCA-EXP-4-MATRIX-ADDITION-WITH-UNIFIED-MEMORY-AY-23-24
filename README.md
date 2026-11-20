# PCA-EXP-4-MATRIX-ADDITION-WITH-UNIFIED-MEMORY AY 23-24
<h3>AIM:</h3>
<h3>ENTER YOUR NAME</h3>
<h3>ENTER YOUR REGISTER NO</h3>
<h3>EX. NO</h3>
<h3>DATE</h3>
<h1> <align=center> MATRIX ADDITION WITH UNIFIED MEMORY </h3>
  Refer to the program sumMatrixGPUManaged.cu. Would removing the memsets below affect performance? If you can, check performance with nvprof or nvvp.</h3>

## AIM:
To perform Matrix addition with unified memory and check its performance with nvprof.
## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC
Google Colab with NVCC Compiler
## PROCEDURE:
1.	Setup Device and Properties
Initialize the CUDA device and get device properties.
2.	Set Matrix Size: Define the size of the matrix based on the command-line argument or default value.
Allocate Host Memory
3.	Allocate memory on the host for matrices A, B, hostRef, and gpuRef using cudaMallocManaged.
4.	Initialize Data on Host
5.	Generate random floating-point data for matrices A and B using the initialData function.
6.	Measure the time taken for initialization.
7.	Compute Matrix Sum on Host: Compute the matrix sum on the host using sumMatrixOnHost.
8.	Measure the time taken for matrix addition on the host.
9.	Invoke Kernel
10.	Define grid and block dimensions for the CUDA kernel launch.
11.	Warm-up the kernel with a dummy launch for unified memory page migration.
12.	Measure GPU Execution Time
13.	Launch the CUDA kernel to compute the matrix sum on the GPU.
14.	Measure the execution time on the GPU using cudaDeviceSynchronize and timing functions.
15.	Check for Kernel Errors
16.	Check for any errors that occurred during the kernel launch.
17.	Verify Results
18.	Compare the results obtained from the GPU computation with the results from the host to ensure correctness.
19.	Free Allocated Memory
20.	Free memory allocated on the device using cudaFree.
21.	Reset Device and Exit
22.	Reset the device using cudaDeviceReset and return from the main function.

## PROGRAM:
```

%%writefile unifmem1.cu

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define CHECK(call) \
{ \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s\n", cudaGetErrorString(err)); \
        exit(1); \
    } \
}

void initialData(float *data, int size)
{
    for (int i = 0; i < size; i++)
        data[i] = (float)(rand() % 10);
}

void sumMatrixOnHost(float *A, float *B, float *C, int N)
{
    for (int i = 0; i < N; i++)
        C[i] = A[i] + B[i];
}

__global__ void sumMatrixOnGPU(float *A, float *B, float *C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
        C[idx] = A[idx] + B[idx];
}

int main()
{
    int nx = 1024;
    int ny = 1024;
    int N = nx * ny;
    size_t bytes = N * sizeof(float);

    float *A, *B, *C, *hostRef;

    printf("========================================\n");
    printf(" Matrix Addition Using CUDA Unified Memory\n");
    printf("========================================\n");

    printf("Matrix Size: %d x %d\n", nx, ny);

    CHECK(cudaMallocManaged(&A, bytes));
    CHECK(cudaMallocManaged(&B, bytes));
    CHECK(cudaMallocManaged(&C, bytes));
    CHECK(cudaMallocManaged(&hostRef, bytes));

    printf("Unified Memory allocated successfully.\n");

    srand(123);
    initialData(A, N);
    initialData(B, N);

    sumMatrixOnHost(A, B, hostRef, N);

    printf("CPU matrix addition completed.\n");

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    printf("Grid: %d\n", blocks);
    printf("Block: %d\n", threads);

    printf("Running matrix addition on GPU...\n");

    sumMatrixOnGPU<<<blocks, threads>>>(A, B, C, N);

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    printf("GPU matrix addition completed.\n");

    bool match = true;

    for (int i = 0; i < N; i++)
    {
        if (fabs(hostRef[i] - C[i]) > 1e-5)
        {
            match = false;
            printf("Mismatch at index %d\n", i);
            break;
        }
    }

    if (match)
        printf("Arrays match successfully!\n");
    else
        printf("Arrays do not match!\n");

    printf("\nFirst 10 Results:\n");

    for (int i = 0; i < 10; i++)
        printf("%.2f + %.2f = %.2f\n", A[i], B[i], C[i]);

    CHECK(cudaFree(A));
    CHECK(cudaFree(B));
    CHECK(cudaFree(C));
    CHECK(cudaFree(hostRef));

    CHECK(cudaDeviceReset());

    printf("\nProgram completed successfully!\n");

    return 0;
}

```

## OUTPUT:

<img width="1776" height="647" alt="Screenshot 2026-11-20 182033" src="https://github.com/user-attachments/assets/bb434488-d108-4e0d-b3eb-a66b1b356da5" />

## RESULT:
Thus the program has been executed by using unified memory. It is observed that removing memset function has given less time.
