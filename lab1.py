#! /usr/bin/env python
# -*- coding: utf-8 -*-
###-------------------------------------------------------------------
### File	: hyperbolic.py
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



block_size = 16

def matrix_to_array(A, n = len(A), m = len(A[0])):
	h = (len(A) // block_size + 1) * block_size
	w = (len(A[0]) // block_size + 1) * block_size
	
	# not fills array by zero
	result = np.empty((h, w), dtype=np.float32)

	for i in range(n):
		for j in range(m):
			result[i][j] = A[i][j]
	
	return result

def array_to_matrix(Arr, n, m):
	A = [[0.0] * m for i in range(n)]
	for i in range(n):
		for j in range(m):
			A[i][j] = Arr[i][j]
	return A

def transponate(A):
	ctx = cl_init()
	queue = cl.CommandQueue(ctx)
	for dev in ctx.devices:
		assert dev.local_mem_size > 0
	
	n = len(A)
	m = len(A[0])
	
#	source = np.random.rand(size, size).astype(np.float32)
	src_array = matrix_to_array(A)
	
	print("src_array = ", src_array)


	mf = cl.mem_flags
	a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=src_array)
	a_t_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=src_array.nbytes)
	
	method = NaiveTranspose
	
#	TransposeWithLocal(ctx)(queue, a_t_buf, a_buf, src_array.shape)
	method(ctx)(queue, a_t_buf, a_buf, src_array.shape)
	
	w, h = src_array.shape
	result = np.empty((h, w), dtype=src_array.dtype)
	cl.enqueue_read_buffer(queue, a_t_buf, result).wait()

	a_buf.release()
	a_t_buf.release()

	err = src_array.T - result
	print("err = ", la.norm(err))
	print("result = ", array_to_matrix(result))


def check_transpose():
	cls = TransposeWithLocal
	print("checking", cls.__name__)
	ctx = cl_init()

	for dev in ctx.devices:
		assert dev.local_mem_size > 0

	queue = cl.CommandQueue(ctx)

	size = 1024


	result = transpose_using_cl(ctx, queue, source, NaiveTranspose)

	err = source.T - result
	err_norm = la.norm(err)

	assert err_norm == 0, (size, err_norm)




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
	transponate([[0,1],[2,3]])

#=====================================================================

