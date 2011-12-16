/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/* Matrix multiplication: C = A * B.
 * Device code.
 */
// Thread block size
#define BLOCK_SIZE %(block_size)d

#define AS(i, j) As[j + i * BLOCK_SIZE]
#define BS(i, j) Bs[j + i * BLOCK_SIZE]

///////////////////////////////////////////////////////////////////////////////
//! Matrix multiplication on the device: C = A * B
//! uiWA is A's width and uiWB is B's width
////////////////////////////////////////////////////////////////////////////////
__kernel void
maximum(__global float4* A,
        __global float* B,
	    int n)
{
//	__local float4 A_local[BLOCK_SIZE];
//	__local float B_local[BLOCK_SIZE];
	

    // Block index
//    int bx = get_group_id(0);



    // Thread index
    int i = get_global_id(0);

//	A_local[get_local_id(0)] = A[i];
//   	barrier(CLK_LOCAL_MEM_FENCE);
    
//    get_global_id(1) * get_global_size(0) + get_global_id(0)
	
	if (i <= n/4 + 1) {
		float4 z = A[i];
		B[i] = max(max(z.x, z.y), max(z.z, z.w));
	}
	
//	B[i] = B_local[get_local_id(0)];
//   	barrier(CLK_LOCAL_MEM_FENCE);

}

