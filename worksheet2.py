import numpy
import pycuda.autoinit  # noqa
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

a = numpy.random.randn(4, 4)

a = a.astype(numpy.float32)

a_gpu = cuda.mem_alloc(a.size * a.dtype.itemsize)

cuda.memcpy_htod(a_gpu, a)

mod = SourceModule("""
    __global__ void doublify(float *a)
    {
      int idx = threadIdx.x + threadIdx.y*4;
      a[idx] *= 2;
    }
    """)

func = mod.get_function("doublify")
func(a_gpu, block=(4, 4, 1), grid=(1, 1), shared=0)
