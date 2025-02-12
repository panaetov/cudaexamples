#include <stdint.h>    /* for uint64 definition */
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#define BLOCK_SIZE 256
#define N (2 * 1024000)
#define SIZE (BLOCK_SIZE * N)


__global__ void reduce1(float *a, float *b)
{
    __shared__ int cache[BLOCK_SIZE];

    unsigned int tx = threadIdx.x;
    unsigned int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    cache[tx] = a[i];
    __syncthreads();

    for(int s = 1; s < BLOCK_SIZE; s *= 2) {
        // в каждом варпе будут работающие нити ещё долго.
        if (tx % (2*s) == 0) {
            cache[tx] += cache[tx + s];
        }
        __syncthreads();
    }

    if (tx == 0) {
        b[blockIdx.x] = cache[0];
    }
}


__global__ void reduce2(float *a, float *b)
{
    __shared__ int cache[BLOCK_SIZE];

    unsigned int tx = threadIdx.x;
    unsigned int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    cache[tx] = a[i];
    __syncthreads();

    for(int s = 1; s < BLOCK_SIZE; s *= 2) {
        int index = 2 * s * tx;

        if (index < BLOCK_SIZE) {
            cache[index] += cache[index + s];
        }
        __syncthreads();
    }

    if (tx == 0) {
        b[blockIdx.x] = cache[0];
    }
}


__global__ void reduce3(float *a, float *b)
{
    __shared__ int cache[BLOCK_SIZE];

    unsigned int tx = threadIdx.x;
    unsigned int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    cache[tx] = a[i];
    __syncthreads();

    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tx < s) {
            cache[tx] += cache[tx + s];
        }
        __syncthreads();
    }

    if (tx == 0) {
        b[blockIdx.x] = cache[0];
    }
}


__global__ void reduce4(float *a, float *b)
{
    __shared__ int cache[BLOCK_SIZE];

    unsigned int tx = threadIdx.x;
    unsigned int i = blockIdx.x * BLOCK_SIZE * 2 + threadIdx.x;

    cache[tx] = a[i] + a[i + BLOCK_SIZE];
    __syncthreads();

    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tx < s) {
            cache[tx] += cache[tx + s];
        }
        // продолжаем синхронизироваться, хотя нить уже не работает.
        __syncthreads();
    }

    if (tx == 0) {
        b[blockIdx.x] = cache[0];
    }
}


__global__ void reduce5(float *a, float *b)
{
    __shared__ int cache[BLOCK_SIZE];

    unsigned int tx = threadIdx.x;
    unsigned int i = blockIdx.x * BLOCK_SIZE * 2 + threadIdx.x;

    cache[tx] = a[i] + a[i + BLOCK_SIZE];
    __syncthreads();

    for (int s = BLOCK_SIZE / 2; s > 32; s >>= 1) {
        if (tx < s) {
            cache[tx] += cache[tx + s];
        }
        __syncthreads();
    }

    // Работаю нити одного варпа.
    // Они по умолчанию синхронизированы, поэтому не нужен __syncthreads();
    cache[tx] += cache[tx + 32];
    cache[tx] += cache[tx + 16];
    cache[tx] += cache[tx + 8];
    cache[tx] += cache[tx + 4];
    cache[tx] += cache[tx + 2];
    cache[tx] += cache[tx + 1];

    if (tx == 0) {
        b[blockIdx.x] = cache[0];
    }
}


float generate_random()
{
    return (float)(rand()) / (float)(rand());
}


float run()
{
    int i;
    float dt;

    cudaEvent_t event_start, event_stop;
    cudaEventCreate(&event_start);
    cudaEventCreate(&event_stop);

    float *a = (float *)malloc(sizeof(float) * SIZE);
    float *b = (float *)malloc(sizeof(float) * SIZE);

    float *a_gpu;
    float *b_gpu;

    for(i = 0; i < SIZE; ++i) {
        a[i] = generate_random(); 
    }

    cudaMalloc(&a_gpu, sizeof(float) * SIZE);
    cudaMalloc(&b_gpu, sizeof(float) * SIZE);

    cudaMemcpy(a_gpu, a, sizeof(float) * SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(b_gpu, b, sizeof(float) * SIZE, cudaMemcpyHostToDevice);

    int block_size = BLOCK_SIZE;
    int grid_size = N;

    cudaEventRecord(event_start, 0);
    reduce1<<<grid_size, block_size>>>(a_gpu, b_gpu);
    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&dt, event_start, event_stop);
    printf("reduce1 elapsed time:  %3.3f ms \n", dt);

    cudaEventRecord(event_start, 0);
    reduce2<<<grid_size, block_size>>>(a_gpu, b_gpu);
    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&dt, event_start, event_stop);
    printf("reduce2 elapsed time:  %3.3f ms \n", dt);

    cudaEventRecord(event_start, 0);
    reduce3<<<grid_size, block_size>>>(a_gpu, b_gpu);
    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&dt, event_start, event_stop);
    printf("reduce3 elapsed time:  %3.3f ms \n", dt);

    cudaEventRecord(event_start, 0);
    reduce4<<<grid_size / 2, block_size>>>(a_gpu, b_gpu);
    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&dt, event_start, event_stop);
    printf("reduce4 elapsed time:  %3.3f ms \n", dt);

    cudaEventRecord(event_start, 0);
    reduce5<<<grid_size / 2, block_size>>>(a_gpu, b_gpu);
    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&dt, event_start, event_stop);
    printf("reduce5 elapsed time:  %3.3f ms \n", dt);

    cudaMemcpy(b, b_gpu, sizeof(float) * SIZE, cudaMemcpyDeviceToHost);
    cudaFree(a_gpu);
    cudaFree(b_gpu);

    free(a);
    free(b);

    return dt;
}


int main()
{
    run();
    return 0;
}
