#! /usr/bin/env python
# -*- coding: utf-8 -*-
###-------------------------------------------------------------------
### File	: LUdecomposition.py
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

def array_to_matrix(Arr, n, m):
	A = [[0.0] * m for i in range(n)]

	for i in range(n):
		for j in range(m):
			A[i][j] = Arr[i][j]
	return A

def maximum(v):
	ctx, queue = cl_init()
	size = ((len(v)-1) // block_size + 1) * block_size
	print "size =", size
	v_buf = np.empty(size, dtype=np.float32)
	m_buf = np.empty(size //4 + 1, dtype=np.float32)

	for i in range(len(v)):
		v_buf[i] = v[i]
	for i in range(len(v), size):
		v_buf[i] = np.float32(0.0)
	
	kernel_params = {"block_size": block_size}

#	prg = cl.Program(ctx, open("LU.cl").read() % kernel_params, ).build()
	prg = cl.Program(ctx, open("max.cl").read() % kernel_params ).build()

	mf = cl.mem_flags
	d_v_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=v_buf)
	d_m_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=v_buf.nbytes)

	n = len(v)
	
	gpu_time = 0
	
	tmp = v_buf.shape
	tmp = (tmp[0] // 16 *4 +4, )
	
	while n>2:
		evt = prg.maximum(queue, tmp, (block_size,),
	                     d_v_buf, d_m_buf, np.uint32(n))
		evt.wait()
		gpu_time += evt.profile.end - evt.profile.start
		n //= 4
		tmp = (tmp[0] // 16 *4 +4, )
		
	print('gpu_time', gpu_time*1e-9)

#	cl.enqueue_copy(queue, L_buf, d_L_buf)
	cl.enqueue_copy(queue, m_buf, d_m_buf)
	
	print("m_buf")
	print(m_buf)
	
#	res = array_to_matrix(L_buf, n, m)
	
	
	



def LUP(A):
## check sizes here!!!
	ctx, queue = cl_init()
	n = len(A)
	m = len(A[0])
	
	a_buf = matrix_to_array(A, n, m)
	L_buf = np.empty(a_buf.shape).astype(np.float32)
#	c_buf = np.empty(a_buf.shape).astype(np.float32)
	
	print("a_buf")
	print(a_buf)
	
	print(L_buf.shape)
	

#	print("result:")
#	print(res)
#	print("origin:")
#	print(np.matrix(A))







#=====================================================================
def main():
	pass


#=====================================================================
if __name__ == "__main__":
#	check_transpose()
#	benchmark_transpose()
	
	inp = [1,2,3,4,5,6,7] * 10**6
	
	inp[666] = 9
	inp[0] = 8
	
	maximum( inp )
#	multiply( [[6,5,3],[7,4,2],[8,3,1]], [[1,2,4,5],[9,8,6,5],[1,8,6,5]] )

#=====================================================================

