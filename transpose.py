#! /usr/bin/env python
# -*- coding: utf-8 -*-
###-------------------------------------------------------------------
### File	: transpose.py
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

# Transposition of a matrix
# originally for PyCUDA by Hendrik Riedmann <riedmann@dam.brown.edu>


block_size = 16

class NaiveTranspose:
	def __init__(self, ctx):
		self.kernel = cl.Program(ctx, """
		__kernel
		void transpose(
		  __global float *a_t, __global float *a,
		  unsigned a_width, unsigned a_height)
		{
		  int read_idx = get_global_id(0) + get_global_id(1) * a_width;
		  int write_idx = get_global_id(1) + get_global_id(0) * a_height;

		  a_t[write_idx] = a[read_idx];
		}
		"""% {"block_size": block_size}).build().transpose

	def __call__(self, queue, tgt, src, shape):
		w, h = shape
		assert w % block_size == 0
		assert h % block_size == 0

		return self.kernel(queue, (w, h), (block_size, block_size),
			tgt, src, np.uint32(w), np.uint32(h))



class TransposeWithLocal:
	def __init__(self, ctx):
		self.kernel = cl.Program(ctx, """
		#define BLOCK_SIZE %(block_size)d
		#define A_BLOCK_STRIDE (BLOCK_SIZE * a_width)
		#define A_T_BLOCK_STRIDE (BLOCK_SIZE * a_height)

		__kernel __attribute__((reqd_work_group_size(BLOCK_SIZE, BLOCK_SIZE, 1)))
		void transpose(
		  __global float *a_t,
		  __global float *a,
		  unsigned a_width,
		  unsigned a_height,
		  __local float *a_local)
		{
		  int base_idx_a   =
			get_group_id(0) * BLOCK_SIZE +
			get_group_id(1) * A_BLOCK_STRIDE;
		  int base_idx_a_t =
			get_group_id(1) * BLOCK_SIZE +
			get_group_id(0) * A_T_BLOCK_STRIDE;

		  int glob_idx_a   = base_idx_a + get_local_id(0) + a_width * get_local_id(1);
		  int glob_idx_a_t = base_idx_a_t + get_local_id(0) + a_height * get_local_id(1);

		  a_local[get_local_id(1)*BLOCK_SIZE+get_local_id(0)] = a[glob_idx_a];

		  barrier(CLK_LOCAL_MEM_FENCE);

		  a_t[glob_idx_a_t] = a_local[get_local_id(0)*BLOCK_SIZE+get_local_id(1)];
		}
		"""% {"block_size": block_size}).build().transpose

	def __call__(self, queue, tgt, src, shape):
		w, h = shape
		
		assert w % block_size == 0
		assert h % block_size == 0

		return self.kernel(queue, (w, h), (block_size, block_size),
			tgt, src, np.uint32(w), np.uint32(h),
			cl.LocalMemory(4*block_size*(block_size+1)))


		savefig("transpose-benchmark-%d.pdf" % i)




#=====================================================================
if __name__ == "__main__":
#	check_transpose()
#	benchmark_transpose()
#	transponate([[0,1],[2,3]])
	pass

#=====================================================================

