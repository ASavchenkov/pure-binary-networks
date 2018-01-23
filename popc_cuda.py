import torch
from cupy.cuda import function
from pynvrtc.compiler import Program
from collections import namedtuple
import math

#I'm not entirely sure why I have to do this, but cupy complains
#about initialization if we haven't put anything on the GPU
#(I'm guessing there's a correct way to do this, but this will do.)
sacrifice = torch.ByteTensor().cuda()

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

BLOCK_SIZE = 1024

#a is an n dimensional CudaByteTensor
def popc(bits):
     
    #create a cuda tensor to hold the counts
    counts = torch.IntTensor(bits.size()).cuda()
    
    #cupy pynvrtc boilerplate
    Stream = namedtuple('Stream', ['ptr'])
    s = Stream(ptr=torch.cuda.current_stream().cuda_stream)
    
    blocks_x = bits.numel()//BLOCK_SIZE + 1
    blocks_y = 0
    if(blocks_x >65535):
        tmp = int(math.sqrt(blocks_x)) + 1
        blocks_x = tmp
        blocks_y = tmp
    f(grid=(blocks_x, blocks_y,1), block=(BLOCK_SIZE,1,1), args=[bits.data_ptr(), counts.data_ptr(), bits.numel()],
      stream=s)

    return counts
    
if __name__ == '__main__':
    a = torch.ByteTensor([34,2]).cuda()
    print(a,popc(a)) 


