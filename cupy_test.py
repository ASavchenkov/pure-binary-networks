import torch
from cupy.cuda import function
from pynvrtc.compiler import Program
from collections import namedtuple

a = torch.ByteTensor([34,5]).cuda()
b = torch.IntTensor(a.size()).cuda()

kernel = '''
extern "C"
__global__ void popc_kernel(unsigned char *in, int * out, int size)
{
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if(i >= size) return;
    out[i] = __popc((unsigned int)(in[i]));
}
'''
program = Program(kernel.encode(), 'popc_kernel.cu'.encode())
ptx = program.compile()

m = function.Module()
m.load(bytes(ptx.encode()))

f = m.get_function('popc_kernel')

Stream = namedtuple('Stream', ['ptr'])
s = Stream(ptr=torch.cuda.current_stream().cuda_stream)

print(a)
f(grid=(1,1,1), block=(1024,1,1), args=[a.data_ptr(), b.data_ptr(), a.numel()],
  stream=s)

print(a)
print(b)
