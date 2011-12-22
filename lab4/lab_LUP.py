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
from time import time

# example provided by Eilif Muller

queue = None
prg = None
ctx = None


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

block_size = 2

def calc_groupsize(n):
	return ((n-1) // block_size + 1) * block_size


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

def LU_one_step(d_a_buf, d_L_buf, sizes, m, k):
	gs = tuple(calc_groupsize(x) for x in sizes)
	event = prg.one_step(queue, gs, (block_size, block_size),
	                              d_a_buf, d_L_buf, np.uint32(m), np.uint32(k))
	event.wait()

def swap_row(d_a_buf, m, i1, i2):
	event = prg.swap_row(queue, (m, ), (block_size, ),
	                              d_a_buf, np.uint32(m), np.uint32(i1), np.uint32(i2))
	event.wait()

def swap_col(d_a_buf, m, i1, i2):
	event = prg.swap_col(queue, (m, ), (block_size, ),
	                              d_a_buf, np.uint32(m), np.uint32(i1), np.uint32(i2))
	event.wait()

def i_max(d_a_buf, m, row):
	m_buf = np.empty(m).astype(np.float32)

	mf = cl.mem_flags
	d_m_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=m_buf.nbytes)
	event = prg.extract_row(queue, m_buf.shape, (block_size, ),
	                              d_a_buf, d_m_buf, np.uint32(m), np.uint32(row)).wait()
	cl.enqueue_copy(queue, m_buf, d_m_buf)
	
	print("m_buf =>")
	print(m_buf)
	
	arr = [x for x in m_buf]
	
	#maybe auto release
	d_m_buf.release()
	
	return arr.index(max(arr))

def max_col(d_a_buf, m, col):
	m_buf = np.empty(m).astype(np.float32)
	res_buf = np.empty(1).astype(np.uint32)
	shift = col
	
	mf = cl.mem_flags
	d_m_buf = cl.Buffer(ctx, mf.READ_WRITE, size=m_buf.nbytes)
	d_r_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=res_buf.nbytes)
	
	event = prg.max_col(queue, m_buf.shape, (block_size, ),
	                    d_a_buf, d_m_buf, np.uint32(m), np.uint32(col), np.uint32(shift))
	event.wait()
	cl.enqueue_copy(queue, m_buf, d_m_buf)

	event = prg.index_col(queue, m_buf.shape, (block_size, ),
	                      d_a_buf, d_m_buf, d_r_buf,
	                      np.uint32(m), np.uint32(col), np.uint32(shift))
	event.wait()
	
	cl.enqueue_copy(queue, res_buf, d_r_buf)
	
	print("max_row =>")
	print(m_buf)

	print("res_row =>")
	print(res_buf)
	return res_buf[0]
		
#	arr = [x for x in m_buf]

	

#	d_m_buf.release()
	
#	return arr.index(max(arr))






def LUP(A):
## check sizes here!!!
	global ctx
	global queue
	global prg
	
	ctx, queue = cl_init()
	kernel_params = {"block_size": block_size}

	prg = cl.Program(ctx, open("my_LUP.cl").read() % kernel_params, ).build()
#	prg = cl.Program(ctx, open("my_LUP.cl").read() ).build()

	n = len(A)
	m = len(A[0])

	# correct size
	rem = m % block_size
	if rem:
		A = [row + [0.0]*(block_size - rem) for row in A]
		m = len(A[0])
	
	a_buf = np.array(A).astype(np.float32)
	L_buf = np.empty_like(a_buf).astype(np.float32)

	print("a_buf input")
	print(a_buf)
	
	print(L_buf.shape)
	
	mf = cl.mem_flags
	d_a_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=a_buf)
	d_L_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, size=L_buf.nbytes, hostbuf=L_buf)

#	s = max_col(d_a_buf, m, 3)
#	print("s =",s)
	
#	LU_one_step(d_a_buf, d_L_buf, a_buf.shape, n, 0)
#	LU_one_step(d_a_buf, d_L_buf, a_buf.shape, n, 1)
#	LU_one_step(d_a_buf, d_L_buf, a_buf.shape, n, 2)
#	LU_one_step(d_a_buf, d_L_buf, a_buf.shape, n, 3)

#	cl.enqueue_copy(queue, a_buf, d_a_buf)
#	cl.enqueue_copy(queue, L_buf, d_L_buf)

#	print("a_buf")
#	print(a_buf)
#	print("L_buf")
#	print(L_buf)

#	return	
	
	# actual benchmark ------------------------------------------------------------
	t1 = time()
	
	p_act = []
	for k in range(n-1):
		s = max_col(d_a_buf, m, k)
		if s != k:
			p_act.append((s,k))
			swap_row(d_a_buf, m, s, k)
			swap_row(d_L_buf, m, s, k)
			swap_col(d_L_buf, m, s, k)
			print('swap', k, s)

#		print("a_buf_tmp")
#		print(a_buf)
		
		# add some spike
#		cl.enqueue_copy(queue, a_buf, d_a_buf)
#		cl.enqueue_copy(queue, L_buf, d_L_buf)
#		d_a_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=a_buf)
#		d_L_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, size=L_buf.nbytes, hostbuf=L_buf)
		# end of spike
		
		print("one_step", k)
		LU_one_step(d_a_buf, d_L_buf, a_buf.shape, n, k)
		
		cl.enqueue_copy(queue, a_buf, d_a_buf)
		cl.enqueue_copy(queue, L_buf, d_L_buf)

#		if k==1: break
	
	# construct permute vector
	p = list(range(n))
	for (s,k) in p_act:
		p[s],p[k] = p[k],p[s]
	

	gpu_time = (time() - t1)

	cl.enqueue_copy(queue, a_buf, d_a_buf)
	cl.enqueue_copy(queue, L_buf, d_L_buf)
	
	print("a_buf")
	print(a_buf)
	print("L_buf")
	print(L_buf)
	
	print("p = " + str(p))

	if rem:
		m -= block_size - rem
	
	res = [[ round(a+l, 5) for (a,l) in zip(row_a[:m], row_l[:m])]
	              for (row_a, row_l) in zip(a_buf, L_buf)]
	
#	res = [[0.0]*m for i in range(n)]
#	for i in range(n):
#		for j in range(m):
#			if i > j: #low sub-matrix
#				res[i][j] = L[i][j]
#			else:
#				res[i][j] = A[i][j]

#	for i in range(n): res[i][i] = 1.0

	return res
		
	
#	res = array_to_matrix(L_buf, n, m)
	
#	print("result:")
#	print(res)
#	print("origin:")
#	print(np.matrix(A))







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

#	res = LUP( [[1,2,3,4],[3,4,3,4],[5,6,8,9],[5,6,3,7]] )
	
	res = LUP([
			[2, 7, -8, 6],
			[4, 4, 0, -7],
			[-1, -3, 6, 3],
			[9, -7, -2, -8]
		  ])
	
#	multiply( [[6,5,3],[7,4,2],[8,3,1]], [[1,2,4,5],[9,8,6,5],[1,8,6,5]] )
	from pprint import pprint
	pprint(res)

#=====================================================================

