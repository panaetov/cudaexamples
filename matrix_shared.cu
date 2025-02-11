#include <stdint.h>    /* for uint64 definition */
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#define BLOCK_SIZE 16
#define N 512
#define MATRIX_SIZE (BLOCK_SIZE * N)
#define MATRIX_CAPABILITY (MATRIX_SIZE * MATRIX_SIZE)


__global__ void kernel(float *a, float *b, float *c)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int abegin = N * BLOCK_SIZE * by;
    int aend = abegin + N * BLOCK_SIZE - 1;
    int astep = BLOCK_SIZE;

    int bstep = BLOCK_SIZE * N;
    int bbegin = BLOCK_SIZE *  bx;

    float result = 0.0f;
    for(int ia = abegin, ib = bbegin; ia <= aend; ia+= astep, ib += bstep) {
        __shared__ float as[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float bs[BLOCK_SIZE][BLOCK_SIZE];

        as[ty][tx] = a[ia + N * ty + tx];
        bs[ty][tx] = b[ib + N * ty + tx];

        __syncthreads();

        for(int k = 0; k < BLOCK_SIZE; ++k) {
            // normal
            result += as[ty][k] * bs[k][tx];

            // broadcast
            // result += as[0][0] * bs[0][0];

            // bank conflict
            // result += as[(ty + tx * 2) % BLOCK_SIZE][k] * bs[k][tx];

        }
        __syncthreads();
    }

    int ic = N * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    c[ic + N * BLOCK_SIZE * ty + tx] = result;
}


float generate_random()
{
    return (float)(rand()) / (float)(rand());
}


float run()
{
    int i, j;
    float dt;

    cudaEvent_t event_start, event_stop;
    cudaEventCreate(&event_start);
    cudaEventCreate(&event_stop);

    float *matrix_a = (float *)malloc(sizeof(float) * MATRIX_CAPABILITY);
    float *matrix_b = (float *)malloc(sizeof(float) * MATRIX_CAPABILITY);
    float *matrix_c = (float *)malloc(sizeof(float) * MATRIX_CAPABILITY);

    float *matrix_a_gpu;
    float *matrix_b_gpu;
    float *matrix_c_gpu;

    for(i = 0; i < MATRIX_SIZE; ++i) {
        for(j = 0; j < MATRIX_SIZE; ++j) {
            matrix_a[i * MATRIX_SIZE + j] = generate_random(); 
            matrix_b[i * MATRIX_SIZE + j] = generate_random(); 
        } 
    }

    cudaMalloc(&matrix_a_gpu, sizeof(float) * MATRIX_CAPABILITY);
    cudaMalloc(&matrix_b_gpu, sizeof(float) * MATRIX_CAPABILITY);
    cudaMalloc(&matrix_c_gpu, sizeof(float) * MATRIX_CAPABILITY);

    cudaMemcpy(matrix_a_gpu, matrix_a, sizeof(float) * MATRIX_CAPABILITY, cudaMemcpyHostToDevice);
    cudaMemcpy(matrix_b_gpu, matrix_b, sizeof(float) * MATRIX_CAPABILITY, cudaMemcpyHostToDevice);

    dim3 block_size = dim3(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size = dim3(N, N);

    cudaEventRecord(event_start, 0);
    kernel<<<grid_size, block_size>>>(matrix_a_gpu, matrix_b_gpu, matrix_c_gpu);
    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&dt, event_start, event_stop);

    cudaError_t error = cudaGetLastError();

    printf("Kernel error %d \n", error);

    cudaMemcpy(matrix_c, matrix_c_gpu, sizeof(matrix_c), cudaMemcpyDeviceToHost);
    cudaFree(matrix_a_gpu);
    cudaFree(matrix_b_gpu);
    cudaFree(matrix_c_gpu);

    free(matrix_a);
    free(matrix_b);
    free(matrix_c);

    return dt;
}


int main()
{
    float dt = run();
    printf("Kernel elapsed time:  %3.3f ms \n", dt);
    return 0;
}
