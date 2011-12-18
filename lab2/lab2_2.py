#! /usr/bin/env python
# -*- coding: utf-8 -*-
###-------------------------------------------------------------------
### File	: multiply.py
### Author  : Oleg Baskakov
### Description : matrix class
###
### 2011. Written for Moscow Aviation Institute.
###-------------------------------------------------------------------

from __future__ import division
import pyopencl as cl
import pyopencl.array as cl_array
import numpy as np
import numpy.linalg as la
from transpose import NaiveTranspose, TransposeWithLocal
from time import time

# example provided by Eilif Muller

## works with NxN matrix when N > block_size

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
	queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
	return ctx, queue

block_size = 4


def test_mul(M1, M2):
	h1 = len(M1)
	w1 = len(M1[0])
	h2 = len(M2)
	w2 = len(M2[0])

	result = [ [0.0]*h2 for i in range(w1)]
	
	assert w1 == h2
	
	for i in range(w2):
		for j in range(h1):
			result[i][j] = 0
			for k in range(h2):
				result[i][j] += M1[i][k] * M2[k][j]
	return result


def matrix_to_array(A, n, m):
	h = ((n-1) // block_size + 1) * block_size
	w = ((m-1) // block_size + 1) * block_size

	# not fills array by zero
	result = np.empty((h, w), dtype=np.float32)

	for i in range(h):
		for j in range(w):
			result[i][j] = np.float32(0.0)

	for i in range(n):
		for j in range(m):
			result[i][j] = A[i][j]
	
	return result

def array_to_matrix(Arr, n, m, k):
	A = [[0.0] * m for i in range(n)]

	for i in range(n):
		for j in range(m):
			A[i][j] = Arr[i][j]
	return A



def multiply(A, B):
## check sizes here!!!
	ctx, queue = cl_init()
	h1 = len(A)
	h2 = len(B)
	w1 = len(A[0])
	w2 = len(B[0])
	
	if "NVIDIA" in queue.device.vendor:
		options = "-cl-mad-enable -cl-fast-relaxed-math"
	else:
		options = ""
	
	a_buf = matrix_to_array(A, h1, w1)
	b_buf = matrix_to_array(B, h2, w2)
	c_buf = np.empty((a_buf.shape[0], b_buf.shape[1])).astype(np.float32)
	
	print("a_buf")
	print(a_buf)
	print("b_buf")
	print(b_buf)
	
	kernel_params = {"block_size": block_size}

	prg = cl.Program(ctx, open("mul.cl").read() % kernel_params, ).build(options=options)

	mf = cl.mem_flags
	d_a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_buf)
	d_b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_buf)
	d_c_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=c_buf.nbytes)

	# actual benchmark ------------------------------------------------------------
	t1 = time()
	
	event = prg.matrixMul(queue, c_buf.shape, (block_size, block_size),
	                      d_c_buf, d_a_buf, d_b_buf,
	                      np.uint32(c_buf.shape[0]), np.uint32(c_buf.shape[1]))

	event.wait()
	gpu_time = (time() - t1)

	cl.enqueue_copy(queue, c_buf, d_c_buf)
	
	print("c_buf")
	print(c_buf)
	
	res = array_to_matrix(c_buf, h1, w2, c_buf.shape[1])
	
	print("result:")
	for row in res:
		print(row)
	print("origin with numpy:")
	print( np.matrix(A) * np.matrix(B))




def benchmark_transpose():
	ctx = cl.create_some_context()

	for dev in ctx.devices:
		assert dev.local_mem_size > 0

	queue = cl.CommandQueue(ctx, 
			properties=cl.command_queue_properties.PROFILING_ENABLE)

	mem_bandwidths = {}

	method = TransposeWithLocal
	
	name = TransposeWithLocal.__name__.replace("Transpose", "")

	mem_bandwidths[TransposeWithLocal] = meth_mem_bws = []

	size = 1024

	source = np.random.rand(size, size).astype(np.float32)

	mf = cl.mem_flags
	a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=source)
	a_t_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=source.nbytes)

#	TransposeWithLocal(ctx)(queue, a_t_buf, a_buf, source.shape)

	event = TransposeWithLocal(ctx)(queue, a_t_buf, a_buf, source.shape)

	event.wait()

	time = event.profile.end - event.profile.start

	mem_bw = 2*source.nbytes*12/(time*1e-9)
	print("benchmarking", name, size, mem_bw/1e9, "GB/s")
	meth_mem_bws.append(mem_bw)

	a_buf.release()
	a_t_buf.release()
	
	methods = [TransposeWithLocal]


	from matplotlib.pyplot import clf, plot, title, xlabel, ylabel, savefig, legend, grid
	for i in range(len(methods)):
		clf()
		for j in range(i+1):
			method = methods[j]
			name = method.__name__.replace("Transpose", "")
			plot(size, np.array(mem_bandwidths[method])/1e9, "o-", label=name)

		xlabel("Matrix width/height $N$")
		ylabel("Memory Bandwidth [GB/s]")
		legend(loc="best")
		grid()

		savefig("transpose-benchmark-%d.pdf" % i)






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
#	check_transpose()
#	benchmark_transpose()

	multiply( [[1,2,3],[3,4,3],[5,6,3]], [[5,6,7],[7,8,9],[7,8,9]] )
#	multiply( [[6,5,3],[7,4,2],[8,3,1]], [[1,2,4,5],[9,8,6,5],[1,8,6,5]] )

#=====================================================================

