#include <cuda_runtime.h>

__global__ void cube(float * d_out, float * d_in){
    int idx = threadIdx.x;
    float f = d_in[idx];
    d_out[idx] = f + 21;
}

extern "C" void launchCube(float * h_in, int n, float * h_out){
    float * d_in;
    float * d_out;
    

    int byteSize = n * sizeof(float);
    
    cudaMalloc((void**) &d_in, byteSize);
    cudaMalloc((void**) &d_out, byteSize);

    cudaMemcpy(d_in, h_in, byteSize, cudaMemcpyHostToDevice);

    cube<<<1, n>>>(d_out, d_in);

    cudaMemcpy(h_out, d_out, byteSize, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);

}




