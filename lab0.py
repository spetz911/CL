#! /usr/bin/env python
# -*- coding: utf-8 -*-
###-------------------------------------------------------------------
### File    : lab0.py
### Author  : Oleg Baskakov
### Description : two arrays summ
###
### 2011. Written for Moscow Aviation Institute.
###-------------------------------------------------------------------
import pyopencl as cl
import pyopencl.array as cl_array
import numpy as np
import numpy.linalg as la


def cl_init(type = 'GPU'):
	if type == 'GPU':
		my_type = cl.device_type.GPU
	elif type == 'CPU':
		my_type = cl.device_type.CPU
	
	try:
		plt = cl.get_platforms()[0]
		devices = plt.get_devices(device_type=my_type)
		ctx = cl.Context(devices = devices)
	except:
		ctx = cl.create_some_context(interactive=True)
	return ctx

code = """
__kernel void
sum(__global const float *a,
    __global const float *b,
    __global float *c)
{
  int gid = get_global_id(0);
  c[gid] = a[gid] + b[gid];
}
"""

def alter_sum():
	ctx = cl_init()
	queue = cl.CommandQueue(ctx)

	n = 10**6
	a_gpu = cl_array.to_device(
		    queue, np.random.randn(n).astype(np.float32))
	b_gpu = cl_array.to_device(
		    queue, np.random.randn(n).astype(np.float32))

	cl_sum = cl_array.sum(a_gpu).get()
	numpy_sum = np.sum(a_gpu.get())

	print cl_sum, numpy_sum


#=====================================================================
def main():
	global prg, code
	ctx = cl_init()
	prg = cl.Program(ctx, code).build()
	
	queue = cl.CommandQueue(ctx)
	
	a = np.random.rand(50000).astype(np.float32)
	b = np.random.rand(50000).astype(np.float32)
	
	mf = cl.mem_flags
	a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
	b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
	dest_buf = cl.Buffer(ctx, mf.WRITE_ONLY, b.nbytes)

	prg.sum(queue, a.shape, None, a_buf, b_buf, dest_buf)

	a_plus_b = np.empty_like(a)
	cl.enqueue_copy(queue, a_plus_b, dest_buf)

	print(la.norm(a_plus_b - (a+b)), la.norm(a_plus_b))
	print(a_plus_b[:10])


#=====================================================================
if __name__ == "__main__":
	main()

