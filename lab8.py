# I found this example for PyCuda here:
# http://wiki.tiker.net/PyCuda/Examples/Mandelbrot
#
# I adapted it for PyOpenCL. Hopefully it is useful to someone.
# July 2010, HolgerRapp@gmx.net
#
# Original readme below these lines.

# Mandelbrot calculate using GPU, Serial numpy and faster numpy
# Use to show the speed difference between CPU and GPU calculations
# ian@ianozsvald.com March 2010

# Based on vegaseat's TKinter/numpy example code from 2006
# http://www.daniweb.com/code/snippet216851.html#
# with minor changes to move to numpy from the obsolete Numeric

import numpy as np
import time
import sys

import numpy
import numpy.linalg as la

import pyopencl as cl

import Tkinter as tk
import Image          # PIL
import ImageTk        # PIL

# You can choose a calculation routine below (calc_fractal), uncomment
# one of the three lines to test the three variations
# Speed notes are listed in the same place

# set width and height of window, more pixels take longer to calculate
w = 256
h = 256

def calc_fractal_opencl(q, maxiter):
    ctx = cl.Context(cl.get_platforms()[0].get_devices())
    queue = cl.CommandQueue(ctx)

    output = np.empty(q.shape, dtype=np.uint16)

    mf = cl.mem_flags
    q_opencl = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=q)
    output_opencl = cl.Buffer(ctx, mf.WRITE_ONLY, output.nbytes)

    prg = cl.Program(ctx, """
    #pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
    __kernel void mandelbrot(__global float2 *q,
                     __global ushort *output, ushort const maxiter)
    {
        int gid = get_global_id(0);
        float nreal, real = 0;
        float imag = 0;

        output[gid] = 0;

        for(int curiter = 0; curiter < maxiter; curiter++) {
            nreal = real*real - imag*imag + q[gid].x;
            imag = 2* real*imag + q[gid].y;
            real = nreal;

            if (real*real + imag*imag > 4.0f)
                 output[gid] = curiter;
        }
    }
    """).build()

    prg.mandelbrot(queue, output.shape, (64,), q_opencl,
            output_opencl, np.uint16(maxiter))

    cl.enqueue_read_buffer(queue, output_opencl, output).wait()

    return output



def calc_fractal_serial(q, maxiter):
    # calculate z using numpy
    # this routine unrolls calc_fractal_numpy as an intermediate
    # step to the creation of calc_fractal_opencl
    # it runs slower than calc_fractal_numpy
    z = np.zeros(q.shape, np.complex64)
    output = np.resize(np.array(0,), q.shape)
    for i in range(len(q)):
        for iter in range(maxiter):
            z[i] = z[i]*z[i] + q[i]
            if abs(z[i]) > 2.0:
                q[i] = 0+0j
                z[i] = 0+0j
                output[i] = iter
    return output

def calc_fractal_numpy(q, maxiter):
    # calculate z using numpy, this is the original
    # routine from vegaseat's URL
    output = np.resize(np.array(0,), q.shape)
    z = np.zeros(q.shape, np.complex64)

    for iter in range(maxiter):
        z = z*z + q
        done = np.greater(abs(z), 2.0)
        q = np.where(done,0+0j, q)
        z = np.where(done,0+0j, z)
        output = np.where(done, iter, output)
    return output

# choose your calculation routine here by uncommenting one of the options
calc_fractal = calc_fractal_opencl
# calc_fractal = calc_fractal_serial
# calc_fractal = calc_fractal_numpy

def main2():



    class Mandelbrot(object):
        def __init__(self):
            # create window
            self.root = tk.Tk()
            self.root.title("Mandelbrot Set")
            self.create_image()
            self.create_label()
            # start event loop
            self.root.mainloop()


        def draw(self, x1, x2, y1, y2, maxiter=30):
            # draw the Mandelbrot set, from numpy example
            xx = np.arange(x1, x2, (x2-x1)/w)
            yy = np.arange(y2, y1, (y1-y2)/h) * 1j
            q = np.ravel(xx+yy[:, np.newaxis]).astype(np.complex64)

            start_main = time.time()
            output = calc_fractal(q, maxiter)
            end_main = time.time()

            secs = end_main - start_main
            print("Main took", secs)

            self.mandel = (output.reshape((h,w)) /
                    float(output.max()) * 255.).astype(np.uint8)

        def create_image(self):
            """"
            create the image from the draw() string
            """
            # you can experiment with these x and y ranges
            self.draw(-2.13, 0.77, -1.3, 1.3)
            self.im = Image.fromarray(self.mandel)
            self.im.putpalette(reduce(
                lambda a,b: a+b, ((i,0,0) for i in range(255))
            ))


        def create_label(self):
            # put the image on a label widget
            self.image = ImageTk.PhotoImage(self.im)
            self.label = tk.Label(self.root, image=self.image)
            self.label.pack()

    # test the class
    test = Mandelbrot()


def LoadImage(context, fileName):
	im = Image.open(fileName)
	# Make sure the image is RGBA formatted
	if im.mode != "RGBA":
		im = im.convert("RGBA")
	# Convert to uint8 buffer
	buffer = im.tostring()
	clImageFormat = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNORM_INT8)
	clImage = cl.Image(context,
		cl.mem_flags.READ_ONLY |
		cl.mem_flags.COPY_HOST_PTR,
		clImageFormat,
		im.size,
		None,
		buffer
	)
	return clImage, im.size

def SaveImage(fileName, buffer, imgSize):
	im = Image.fromstring("RGBA", imgSize, buffer.tostring())
	im.save(fileName)

def RoundUp(groupSize, globalSize):
	r = globalSize % groupSize;
	if r == 0:
		return globalSize;
	else:
		return globalSize + groupSize - r;

def CreateContext():
	platforms = cl.get_platforms();
	if len(platforms) == 0:
		print "Failed to find any OpenCL platforms."
		return None
	devices = platforms[0].get_devices(cl.device_type.GPU)
	if len(devices) == 0:
		print "Could not find GPU device, trying CPU..."
		devices = platforms[0].get_devices(cl.device_type.CPU)
		if len(devices) == 0:
			print "Could not find OpenCL GPU or CPU device."
			return None
	# Create a context using the first device
	context = cl.Context([devices[0]])
	return context, devices[0]

def CreateProgram(context, device, fileName):
	kernelFile = open(fileName, 'r')
	kernelStr = kernelFile.read()
	# Load the program source
	program = cl.Program(context, kernelStr)
	# Build the program and check for errors
	program.build(devices=[device])
	return program






def main():
	imageObjects = [ 0, 0 ]
	# Main
	if len(sys.argv) != 3:
		print "USAGE: " + sys.argv[0] + " <inputImageFile> <outputImageFile>"
		return 1
	# Create an OpenCL context on first available platform
	context, device = CreateContext();
	if context == None:
		print "Failed to create OpenCL context."
		return 1
	
	# Create a command-queue on the first device available
	commandQueue = cl.CommandQueue(context, device)

	# Make sure the device supports images, otherwise exit
	if not device.get_info(cl.device_info.IMAGE_SUPPORT):
		print "OpenCL device does not support images."
		return 1

	# Load input image from file and load it into
	# an OpenCL image object
	imageObjects[0], imgSize = LoadImage(context, sys.argv[1])
	# Create ouput image object
	clImageFormat = cl.ImageFormat(cl.channel_order.RGBA,
		cl.channel_type.UNORM_INT8)
	imageObjects[1] = cl.Image(context,
		cl.mem_flags.WRITE_ONLY,
		clImageFormat,
		imgSize)
	# Create sampler for sampling image object
	sampler = cl.Sampler(context,
		False, # Non-normalized coordinates
		cl.addressing_mode.CLAMP_TO_EDGE,
		cl.filter_mode.NEAREST)

	# Create OpenCL program
	program = CreateProgram(context, device, "ImageFilter2D.cl")

	# Call the kernel directly
	localWorkSize = ( 16, 16 )
	globalWorkSize = ( RoundUp(localWorkSize[0], imgSize[0]),
			           RoundUp(localWorkSize[1], imgSize[1]) )
	program.gaussian_filter(commandQueue,
		globalWorkSize,
		localWorkSize,
		imageObjects[0],
		imageObjects[1],
		sampler,
		numpy.int32(imgSize[0]),
		numpy.int32(imgSize[1]))

	# Read the output buffer back to the Host
	buffer = numpy.zeros(imgSize[0] * imgSize[1] * 4, numpy.uint8)
	origin = ( 0, 0, 0 )
	region = ( imgSize[0], imgSize[1], 1 )

	cl.enqueue_read_image(commandQueue, imageObjects[1],
		origin, region, buffer).wait()
	print "Executed program successfully."
	# Save the image to disk
	SaveImage(sys.argv[2], buffer, imgSize)





if __name__ == '__main__':
	main()
