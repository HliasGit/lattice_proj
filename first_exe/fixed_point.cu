#include <iostream>
#include <cuda_runtime.h>

// kernel definition
__global__ void vec_min(int *a, int *min, int size) {
    
    //extern __shared__ int sdata[];

    //int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Load data into shared memory
    // sdata[tid] = (i < size) ? a[i] : INT_MAX;
    // __syncthreads();

    // for(int j = i; j<size; j+=stride){
    //     sdata[tid] = (sdata[tid] < a[j]) ? sdata[tid] : a[j];
    // }
    // __syncthreads();

    // Block reduction
    // for (unsigned int s = blockDim.x / 2; s > 0; s>>=1) {
    //     if (tid < s) {
    //         sdata[tid] = (sdata[tid] < sdata[tid + s]) ? sdata[tid] : sdata[tid + s];
    //     }
    //     __syncthreads();
    // }

    // Write the result for this block to global memory in atomic way
    // if (tid == 0 && sdata[0] < *min) {
    //     *min = sdata[0];
    // }

    for(int j = i; j<size; j+=stride){
            if (a[j] < *min) {
                *min = a[j];
            }
        }
    }

    

int main(){
    int size = 1024*1024*8;

    int *data_GPU;
    int *min_GPU;;

    int *data_local = new int[size];
    int min_local[1] = {INT_MAX};

    for(int i = 0; i < size; i++){
        data_local[i] = i+3;
    }

    data_local[1289344] = 1;

    // Allocate memory for GPU
    cudaMalloc(&data_GPU, 1024*1024*8*sizeof(int));
    cudaMalloc(&min_GPU, sizeof(int));
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Kernel call
    int old_min = INT_MAX-1;
    
    cudaMemcpy(data_GPU, data_local, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(min_GPU, &min_local, sizeof(int), cudaMemcpyHostToDevice);


    while(old_min != min_local[0]){
        old_min = min_local[0];
        vec_min<<<1024, 1024>>>(data_GPU, min_GPU, size);
        cudaDeviceSynchronize();
        cudaMemcpy(&min_local, min_GPU, sizeof(int), cudaMemcpyDeviceToHost);
        std::cout << old_min << std::endl;
    }

    cudaDeviceSynchronize(),
    cudaEventRecord(stop);

    // print size
    std::cout << "Vector size: " << size << std::endl;

    // print result
    std::cout << "Min: " << min_local[0] << std::endl;

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Time: " << milliseconds / 1000<< "s" << std::endl;


    cudaFree(data_GPU);
    cudaFree(min_GPU);

}