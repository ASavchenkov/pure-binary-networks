#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

#include "binutil_cuda_kernel.h"


dim3 cuda_gridsize(int n)
{
    int k = (n - 1) / BLOCK + 1;
    int x = k;
    int y = 1;
    if(x > 65535) {
        x = ceil(sqrt(k));
        y = (n - 1) / (x * BLOCK) + 1;
    }
    dim3 d(x, y, 1);
    return d;
}

__global__ void popc_kernel(unsigned char *in, int * out, int size)
{
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    
    if(i >= size) return;
    out[i] = __popc((unsigned int)(in[i]));
}

void popc_cuda(unsigned char *in, int *out, int size, cudaStream_t stream)
{
    cudaError_t err;

    popc_kernel<<<cuda_gridsize(size), BLOCK, 0, stream>>>(in, out, size);
	
    err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}
