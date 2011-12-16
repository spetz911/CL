//OpenCL C Functions for Working with Images


/* We have now explained how the ImageFilter2D example creates image
objects and a sampler object. We can now explain the Gaussian filter
kernel itself, shown in Listing 8.5. A Gaussian filter is a kernel that is typi-
cally used to smooth or blur an image. It does so by reducing the high-
frequency noise in the image.
Listing 8.5 Gaussian Filter Kernel
*/

__kernel void
gaussian_filter(__read_only  image2d_t srcImg,
                __write_only image2d_t dstImg,
                sampler_t sampler,
                int width, int height)
{
	// Gaussian Kernel is:
	// 1 2 1
	// 2 4 2
	// 1 2 1
	float kernelWeights[9] = { 1.0f, 2.0f, 1.0f,
	2.0f, 4.0f, 2.0f,
	1.0f, 2.0f, 1.0f };
	int2 startImageCoord = (int2) (get_global_id(0) - 1,
		                           get_global_id(1) - 1);
	int2 endImageCoord = (int2) (get_global_id(0) + 1,
		                         get_global_id(1) + 1);
	int2 outImageCoord = (int2) (get_global_id(0),
		                         get_global_id(1));

	if (outImageCoord.x < width && outImageCoord.y < height) {
		int weight = 0;
		float4 outColor = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
		for(int y = startImageCoord.y; y <= endImageCoord.y; y++) {
			for(int x= startImageCoord.x; x <= endImageCoord.x; x++) {
				outColor +=
				(read_imagef(srcImg, sampler, (int2)(x, y)) *
				(kernelWeights[weight] / 16.0f));
				weight += 1;
			}
		}
		// Write the output value to image
		write_imagef(dstImg, outImageCoord, outColor);
	}
}


