#ifndef _BINUTIL_CUDA_KERNEL
#define _BINUTIL_CUDA_KERNEL


#define BLOCK 512
#define MAX_STREAMS 512

#ifdef __cplusplus
extern "C" {
#endif

void popc_cuda(unsigned char *in, int *out, int size, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif
