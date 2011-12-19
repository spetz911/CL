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
		platform = cl.get_platforms()[0]
		devices = platform.get_devices(device_type=my_type)
		ctx = cl.Context(devices = devices)
	except:
		ctx = cl.create_some_context(interactive=True)
	
	device = devices[0]
	print("===============================================================")
	print("Platform name: " + platform.name)
	print("Platform vendor: " + platform.vendor)
	print("Platform version: " + platform.version)
	print("---------------------------------------------------------------")
	print("Device name: " + device.name)
	print("Device type: " + cl.device_type.to_string(device.type))
	print("Local memory: " + str(device.local_mem_size//1024) + ' KB')
	print("Device memory: " + str(device.global_mem_size//1024//1024) + ' MB')
	print("Device max clock speed:" + str(device.max_clock_frequency) + ' MHz')
	print("Device compute units:" + str(device.max_compute_units))
	
	return ctx

code = """
__kernel void
sum(__global const float4 *a,
    __global const float4 *b,
    __global float4 *c)
{
  int gid = get_global_id(0);
  c[gid] = a[gid] + b[gid];
}

__kernel void
sum4(__global const float4 *a,
    __global const float4 *b,
    __global float4 *c)
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
	
	# initialize_CL
	ctx = cl_init()
	cp = cl.command_queue_properties
	queue = cl.CommandQueue(ctx, properties=cp.PROFILING_ENABLE)
	prg = cl.Program(ctx, code).build()


	# generate long random vectors
	size = 100500*16
	a = np.random.rand(size).astype(np.float32)
	b = np.random.rand(size).astype(np.float32)
	a_plus_b = np.empty_like(a)
	
	# create buffers
	mf = cl.mem_flags
	a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
	b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
	dest_buf = cl.Buffer(ctx, mf.WRITE_ONLY, b.nbytes)

	print("...Calcuate 16*100500 elem's vector")
	
	# call OpenCL kernel
	exec_evt = prg.sum(queue, a.shape, (16,), a_buf, b_buf, dest_buf).wait()

	elapsed = 1e-9*(exec_evt.profile.end - exec_evt.profile.start)
	print("Execution time of test1: %g second" % elapsed)
	cl.enqueue_copy(queue, a_plus_b, dest_buf)

	a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
	b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
	dest_buf = cl.Buffer(ctx, mf.WRITE_ONLY, b.nbytes)

	print("...Start vectorized kernel")

	# call OpenCL kernel
	exec_evt = prg.sum4(queue, (a.shape[0]//4,), (16,), a_buf, b_buf, dest_buf).wait()

	elapsed = 1e-9*(exec_evt.profile.end - exec_evt.profile.start)
	print("Execution time of test2: %g second" % elapsed)

	cl.enqueue_copy(queue, a_plus_b, dest_buf)

	# compare result with native numpy
	print('error = ' + str(la.norm(a_plus_b - (a+b))))
	
	# part of vectors and their sum:
	print('Testing:')
	print(a[:3])
	print('+')
	print(b[:3])
	print('=')
	print(a_plus_b[:3])


#=====================================================================
if __name__ == "__main__":
	main()


def tmp1():
		print("===============================================================")
		print("Platform name:", platform.name)
		print("Platform profile:", platform.profile)
		print("Platform vendor:", platform.vendor)
		print("Platform version:", platform.version)
		print("---------------------------------------------------------------")
		print("Device name:", device.name)
		print("Device type:", cl.device_type.to_string(device.type))
		print("Device memory: ", device.global_mem_size//1024//1024, 'MB')
		print("Device max clock speed:", device.max_clock_frequency, 'MHz')
		print("Device compute units:", device.max_compute_units)

		# Simnple speed test
		ctx = cl.Context([device])
		queue = cl.CommandQueue(ctx, 
				properties=cl.command_queue_properties.PROFILING_ENABLE)

		mf = cl.mem_flags
		a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
		b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
		dest_buf = cl.Buffer(ctx, mf.WRITE_ONLY, b.nbytes)

		prg = cl.Program(ctx, """
			__kernel void sum(__global const float *a,
			__global const float *b, __global float *c)
			{
				        int loop;
				        int gid = get_global_id(0);
				        for(loop=0; loop<1000;loop++)
				        {
				                c[gid] = a[gid] + b[gid];
				                c[gid] = c[gid] * (a[gid] + b[gid]);
				                c[gid] = c[gid] * (a[gid] / 2.0);
				        }
				}
				""").build()

		exec_evt = prg.sum(queue, a.shape, None, a_buf, b_buf, dest_buf)
		exec_evt.wait()
		elapsed = 1e-9*(exec_evt.profile.end - exec_evt.profile.start)

		print("Execution time of test: %g s" % elapsed)

		c = numpy.empty_like(a)
		cl.enqueue_read_buffer(queue, dest_buf, c).wait()


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



