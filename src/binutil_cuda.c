#include <THC/THC.h>
#include "binutil_cuda_kernel.h"

extern THCState *state;

//size should be the same for both input and output, so it's a single variable
int popc(THCudaByteTensor *in_tensor, THCudaIntTensor *out_tensor, int size)
{
    unsigned char *in = THCudaByteTensor_data(state, in_tensor);
    int *out = THCudaIntTensor_data(state, out_tensor);
    cudaStream_t stream = THCState_getCurrentStream(state);

    popc_cuda(in, out, size, stream);

    return 1;
}
