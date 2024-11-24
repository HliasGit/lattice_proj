#include <iostream>
#include <cuda_runtime.h>

__global__ void vec_min(int *a, int *b, int size) {
    extern __shared__ int sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Load data into shared memory
    sdata[tid] = INT_MAX;
    __syncthreads();


    for(int j = i; j<size; j+=stride){
        sdata[tid] = (sdata[tid] < a[j]) ? sdata[tid] : a[j];
    }
    __syncthreads();


    // Block reductionti
    for (unsigned int s = blockDim.x / 2; s > 0; s>>=1) {
        if (tid < s) {
            sdata[tid] = (sdata[tid] < sdata[tid + s]) ? sdata[tid] : sdata[tid + s];
        }
        __syncthreads();
    }

    // Write the result for this block to global memory
    if (tid == 0) {
        b[blockIdx.x] = sdata[0];
    }
}


int main(){

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    // sizes
    int vecSize = 1024*1024*8;
    int n_threads = 1024;
    int n_blocks = 1024;

    // Allocate memory for GPU
    int *A;
    int *B;

    // Allocate for host
    int *data = new int[vecSize];
    int *result = new int[n_blocks];

    std::cout << "Vector size: " << vecSize << std::endl;

    // Initialize data
    for (int i = 0; i < vecSize; i++) {
        data[i] = i + 4;
    }

    // Set a minimum somewhere
    data[1024*1024*8-10] = 1;

    // Allocate CUDA memory
    cudaMalloc(&A, vecSize*sizeof(int));
    cudaMalloc(&B, n_threads*sizeof(int));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Copy to the CUDA mem
    cudaMemcpy(A, data, vecSize*sizeof(int), cudaMemcpyHostToDevice);

    cudaEventRecord(start);
    vec_min<<<n_blocks, n_threads, n_threads * sizeof(int)>>>(A, B, vecSize);
    cudaDeviceSynchronize(),
    cudaEventRecord(stop);

    // Copy back
    cudaMemcpy(result, B, n_blocks*sizeof(int), cudaMemcpyDeviceToHost);

    int min = INT_MAX;
    for (int i = 0; i < n_blocks; i++) {
        if (result[i] < min) {
            min = result[i];
        }
    }

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Minimum value: " << min << std::endl;
    std::cout << "Time: " << milliseconds / 1000<< "s" << std::endl;

    cudaFree(A);
    cudaFree(B);   

}