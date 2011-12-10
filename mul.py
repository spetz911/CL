#! /usr/bin/env python
# -*- coding: utf-8 -*-
###-------------------------------------------------------------------
### File	: mul.py
### Author  : Oleg Baskakov
### Description : matrix class
###
### 2011. Written for Moscow Aviation Institute.
###-------------------------------------------------------------------

# example provided by Eilif Muller


KERNEL_CODE = """

// Thread block size
#define BLOCK_SIZE %(block_size)d

// Matrix dimensions
// (chosen as multiples of the thread block size for simplicity)
#define WA %(w_a)d // Matrix A width
#define HA %(h_a)d // Matrix A height
#define WB %(w_b)d // Matrix B width
#define HB WA  // Matrix B height
#define WC WB  // Matrix C width
#define HC HA  // Matrix C height


/* Matrix multiplication: C = A * B.
 * Device code.
 */

#define AS(j, i) As[i + j * BLOCK_SIZE]
#define BS(j, i) Bs[i + j * BLOCK_SIZE]

////////////////////////////////////////////////////////////////////////////////
//! Matrix multiplication on the device: C = A * B
//! WA is A's width and WB is B's width
////////////////////////////////////////////////////////////////////////////////
__kernel __attribute__((reqd_work_group_size(BLOCK_SIZE,BLOCK_SIZE,1))) 
void
matrixMul( __global float* C, __global float* A, __global float* B)
{
	__local float As[BLOCK_SIZE*BLOCK_SIZE];
	__local float Bs[BLOCK_SIZE*BLOCK_SIZE];

	// Block index
	int bx = get_group_id(0);
	int by = get_group_id(1);

	// Thread index
	int tx = get_local_id(0);
	int ty = get_local_id(1);

	// Index of the first sub-matrix of A processed by the block
	int aBegin = WA * BLOCK_SIZE * by;

	// Index of the last sub-matrix of A processed by the block
	int aEnd   = aBegin + WA - 1;

	// Step size used to iterate through the sub-matrices of A
	int aStep  = BLOCK_SIZE;

	// Index of the first sub-matrix of B processed by the block
	int bBegin = BLOCK_SIZE * bx;

	// Step size used to iterate through the sub-matrices of B
	int bStep  = BLOCK_SIZE * WB;

	// Csub is used to store the element of the block sub-matrix
	// that is computed by the thread
	float Csub = 0.0f;

	// Loop over all the sub-matrices of A and B
	// required to compute the block sub-matrix
	for (int a = aBegin, b = bBegin;
			 a <= aEnd;
			 a += aStep, b += bStep) {

		// Load the matrices from device memory
		// to shared memory; each thread loads
		// one element of each matrix
		AS(ty, tx) = A[a + WA * ty + tx];
		BS(ty, tx) = B[b + WB * ty + tx];

		// Synchronize to make sure the matrices are loaded
		barrier(CLK_LOCAL_MEM_FENCE);

		// Multiply the two matrices together;
		// each thread computes one element
		// of the block sub-matrix
		for (int k = 0; k < BLOCK_SIZE; ++k)
			Csub += AS(ty, k) * BS(k, tx);

		// Synchronize to make sure that the preceding
		// computation is done before loading two new
		// sub-matrices of A and B in the next iteration
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// Write the block sub-matrix to device memory;
	// each thread writes one element
	C[get_global_id(1) * get_global_size(0) + get_global_id(0)] = Csub;

}

"""

