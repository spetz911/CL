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


// Thread block size
#define BLOCK_SIZE %(block_size)d

#define AS(i, j) As[j + i * BLOCK_SIZE]
#define BS(i, j) Bs[j + i * BLOCK_SIZE]


__kernel void
max_row(__global float4* A,
		__global float* B,
		int m, int row)
{
	__local float a_local[BLOCK_SIZE];
	const int k = BLOCK_SIZE;
	m /= 4;
	// Thread index
	int i_g = get_global_id(0);
	int i_l = get_local_id(0);
	if (i_g >= m) {
		return;
	}
	
	float4 z = A[row*m + i_g];
	B[i_g + 0] = fabs(z.x);
	B[i_g + 1] = fabs(z.y);
	B[i_g + 2] = fabs(z.z);
	B[i_g + 3] = fabs(z.w);
	
	barrier(CLK_GLOBAL_MEM_FENCE);
	
	while (m > 0 && i_g < m) {
		a_local[i_l] = max(max(B[i_g/4+0], B[i_g/4+1]), max(B[i_g/4+2], B[i_g/4+3]));
		barrier(CLK_LOCAL_MEM_FENCE);
		
		B[i_g] = a_local[i_l];
		barrier(CLK_GLOBAL_MEM_FENCE);
		m /= k;
	}
}


__kernel void
index_row(__global float* A,
          __global float* B,
          __global int* RES,
         int m, int i)
{
	float val = B[0];
    int j = get_global_id(0);

	if (j >= m) {
		return;
	}
	
	if (fabs(A[m*i + j]) == val) {
		RES[0] = j;
	}
}

__kernel void
swap_row(__global float* A,
         int n, int i1, int i2)
{
//	__local float4 A_local[BLOCK_SIZE];
//	__local float B_local[BLOCK_SIZE];

    // Thread index
    int j = get_global_id(0);

	if (j >= n) {
		return;
	}
	
    float z1 = A[n*i1 + j];
    float z2 = A[n*i2 + j];

	A[n*i2 + j] = z1;
	A[n*i1 + j] = z2;
}

__kernel void
swap_coll(__global float* A,
         int n, int j1, int j2)
{
//	__local float4 A_local[BLOCK_SIZE];
//	__local float B_local[BLOCK_SIZE];

    // Thread index
    int i = get_global_id(0);

	if (i >= n) {
		return;
	}
	
    float z1 = A[n*i + j1];
    float z2 = A[n*i + j2];

	A[n*i + j2] = z1;
	A[n*i + j1] = z2;
}


__kernel void
one_step(__global float* A,
            int n, int k)
{
//	__local float4 A_local[BLOCK_SIZE];
//	__local float B_local[BLOCK_SIZE];

    float z = A[n*k + k];

    // Thread index
	int i = get_global_id(1) + k+1;
    int j = get_global_id(0) + k;
	
	if (j >= n || i >= n) {
		return;
	}
	
	float koef = A[n*i + k] / z;
	
	A[i*n + j] = A[i*n + j] - koef * A[n*k + j];
}







/*********************************************
 * Legacy code:
 */



__kernel void
extract_row(__global float4* A,
		    __global float4* B,
		    int m, int i)
{
	int j = get_global_id(0);
	m /= 4;

	if (j >= m) {
		return;
	}

	B[j] = fabs(A[i*m + j]);
}

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
		B[i] = max(max(fabs(z.x), fabs(z.y)), max(fabs(z.z), fabs(z.w)));
	}
	
//	B[i] = B_local[get_local_id(0)];
//   	barrier(CLK_LOCAL_MEM_FENCE);
}

__kernel void
mat_vec_product(__global float4* A,
                __global float4* B,
	            int m, int n)
{
	__local float4 B_local[BLOCK_SIZE];
//	__local float B_local[BLOCK_SIZE];
	


    // Thread index
    int j_g = get_global_id(0);
    int j_l = get_local_id(0);
    
    
    if (j >= n) {
    	return;
    }
    
    B_local[j_l] = B[j_g];
    
    float4 z = 0.0f;
    for (int i=0; i<m; ++i) {
    	z += B_local[j_l]
    }
    

//	A_local[get_local_id(0)] = A[i];
//   	barrier(CLK_LOCAL_MEM_FENCE);
    
//    get_global_id(1) * get_global_size(0) + get_global_id(0)
	
	if (i <= n/4 + 1) {
		float4 z = A[i];
		B[i] = max(max(fabs(z.x), fabs(z.y)), max(fabs(z.z), fabs(z.w)));
	}
	
//	B[i] = B_local[get_local_id(0)];
//   	barrier(CLK_LOCAL_MEM_FENCE);
}



