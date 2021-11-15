// Copyright (c) 2014 Intel Corporation 
// All rights reserved. 
// 
// WARRANTY DISCLAIMER 
// 
// THESE MATERIALS ARE PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT 
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR 
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL INTEL OR ITS 
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, 
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR 
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY 
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY OR TORT (INCLUDING 
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THESE 
// MATERIALS, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. 
// 
// Intel Corporation is the author of the Materials, and requests that all 
// problem reports or change requests be submitted to it directly 

#include <clu.h>
#include <string>
#include <iostream>

// Utility functions to convert and write .ppm files
void convertU8Gray_to_U8RGB (int w, int h, cl_uchar* pSrcU8_G,  cl_uchar* pDstU8_RGB)
{
	unsigned int strideU8_RGB	= w * 3;	// 3 bytes for each RGB
	unsigned int strideU8_G		= w;		// 1 byte for each grayscale

	for (int y=0; y < h; y++)
	{
		for (int x=0; x < w; x++)
		{
			unsigned int locU8_RGB	= y*strideU8_RGB	+ x*3;
			unsigned int locU8_G	= y*strideU8_G		+ x;

			pDstU8_RGB[locU8_RGB]		= (cl_uchar) pSrcU8_G[locU8_G]; // Write gray value as Red
			pDstU8_RGB[locU8_RGB + 1]	= (cl_uchar) pSrcU8_G[locU8_G]; // Write gray value as Green
			pDstU8_RGB[locU8_RGB + 2]	= (cl_uchar) pSrcU8_G[locU8_G]; // Write gray value as Blue
		}
	}
}

void saveppm_fromU8RGB(const char *fname, unsigned int w, unsigned int h, cl_uchar *img)
{
	FILE *fp;
	errno_t err;

	err = fopen_s(&fp, fname, "wb");
	if (err != 0) {
		std::cerr
			<< __FILE__ << "(" << __LINE__ << ") fopen_s returned error!"
		    << std::endl;
		exit(-1);
	}
	err = fprintf(fp, "P6\n");
	if (err < 0) {
		std::cerr
			<< __FILE__ << "(" << __LINE__ << ") fprintf returned error!"
		    << std::endl;
		exit(-1);
	}
	err = fprintf(fp, "%d %d\n", w, h);
	if (err < 0) {
		std::cerr
			<< __FILE__ << "(" << __LINE__ << ") fprintf returned error!"
		    << std::endl;
		exit(-1);
	}
	err = fprintf(fp, "255\n");
	if (err < 0) {
		std::cerr
			<< __FILE__ << "(" << __LINE__ << ") fprintf returned error!"
		    << std::endl;
		exit(-1);
	}
	size_t num_recs_written = fwrite(img, w * h * 3, 1, fp);
	if (num_recs_written != 1) {
		std::cerr
			<< __FILE__ << "(" << __LINE__ << ") fwrite failed to write a file!"
		    << std::endl;
		exit(-1);
	}
	err = fclose(fp);
	if (err != 0) {
		std::cerr
			<< __FILE__ << "(" << __LINE__ << ") fclose returned error!"
		    << std::endl;
		exit(-1);
	}

}

#define STRINGIFY(...) #__VA_ARGS__

// Here is the source code for Sierpinski kernel
// For more on Sierpinski carpet, see Wikipedia 
// http://en.wikipedia.org/wiki/Sierpinski_carpet
//
// The algorithm here is to paint all the pixels in the center sub-square black
// and all the pixels in the other eight sub-squares white, then enqueue
// eight kernels that repeat the procedure on the eight white subsquares.
// We stop enqueueing when the size of square is 3x3 pixels.
// Note that we could switch to iterative formula described in the Wikipedia 
// article for squares larger than 3x3 to reduce the number of enqueues.

const char* sierpinski = STRINGIFY(

const unsigned int BLACK = 0;
const unsigned int WHITE = 255;

__kernel void sierpinski(__global char* src, int width, int offsetx, int offsety) 
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	queue_t q = get_default_queue();

	int one_third = get_global_size(0) / 3;
	int two_thirds = 2 * one_third;

	if (x >= one_third && x < two_thirds && y >= one_third && y < two_thirds) 
	{
		src[(y+offsety)*width+(x+offsetx)] = BLACK;
	} 
	else 
	{
		src[(y+offsety)*width+(x+offsetx)] = WHITE;

		if (one_third > 1 && x % one_third == 0 && y % one_third == 0) 
		{
			const size_t  grid[2] = {one_third, one_third};
			enqueue_kernel(q, 0, ndrange_2D(grid), ^{ sierpinski(src, width, x+offsetx, y+offsety); });
		}
	}
}

);
//=============================================================================

void OCL_CHECK(cl_int in_status, bool exit_on_error = true)
{
    if (in_status != CL_SUCCESS)
    {
		std::cerr
			<< __FILE__ << "(" << __LINE__ << ") OpenCL returned error: "
			<< in_status << " (" << cluPrintError(in_status) << ")" << std::endl;
		if (exit_on_error) 
			exit(-1);
    }
}

int main(int argc, char** argv)
{
    printf("\n>> Sierpinski <<\n");    

    clu_initialize_params ip = { "Intel", 0, "-cl-std=CL2.0", 0, 0, CL_DEVICE_TYPE_GPU };
    cl_int status = cluInitialize(&ip);
	OCL_CHECK(status);

	const unsigned int IMAGE_WIDTH  = 2187;
	const unsigned int IMAGE_HEIGHT = 2187;

	cl_mem dst = cluCreateAlignedBuffer(CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, IMAGE_WIDTH*IMAGE_HEIGHT, 0, &status);
	OCL_CHECK(status);

    cl_program prog = cluBuildSource(sierpinski, 0, &status);
    if (CL_SUCCESS != status)
    {
		OCL_CHECK(status, false);
        printf(cluGetBuildErrors(prog));
		exit(-1);
    }

    cl_kernel kern = clCreateKernel(prog, "sierpinski", &status);
	OCL_CHECK(status);

    cl_uint work_dim = 2;
	size_t global_work_size[2] = { IMAGE_WIDTH, IMAGE_HEIGHT };

	// You need to create device side queue for enqueue_kernel to work
	// We set the device side queue to 16MB, since we are going to have a large number of enqueues
	cl_queue_properties qprop[] = {CL_QUEUE_SIZE, 16*1024*1024, CL_QUEUE_PROPERTIES, 
		                          (cl_command_queue_properties)CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_ON_DEVICE | CL_QUEUE_ON_DEVICE_DEFAULT, 0}; 
   
	cl_command_queue my_device_q = clCreateCommandQueueWithProperties(CLU_CONTEXT, cluGetDevice(CL_DEVICE_TYPE_GPU), qprop, &status);
	OCL_CHECK(status);
	// Verify device side queue size
	int default_queue_size = 0;
	status = clGetCommandQueueInfo(my_device_q, CL_QUEUE_SIZE, sizeof(int), &default_queue_size, 0);
	OCL_CHECK(status);
	printf("CL_QUEUE_SIZE is %d\n", default_queue_size);

	int width = IMAGE_WIDTH, offsetx = 0, offsety = 0;
	status |= clSetKernelArg(kern, 0, sizeof(cl_mem), (void*)&dst);
	status |= clSetKernelArg(kern, 1, sizeof(int), (void*)&width);
	status |= clSetKernelArg(kern, 2, sizeof(int), (void*)&offsetx);
	status |= clSetKernelArg(kern, 3, sizeof(int), (void*)&offsety);
	OCL_CHECK(status);

    status = clEnqueueNDRangeKernel(CLU_DEFAULT_Q, kern, work_dim, 0, global_work_size,  0, 0, 0, 0);
	OCL_CHECK(status);

	cl_uchar* pbuff = (cl_uchar*)clEnqueueMapBuffer(CLU_DEFAULT_Q, dst, CL_TRUE, CL_MAP_READ, 0, IMAGE_WIDTH*IMAGE_HEIGHT, 0, 0, 0, &status);
	OCL_CHECK(status);

	cl_uchar* RGB_data	= (cl_uchar*) malloc(IMAGE_WIDTH*IMAGE_HEIGHT*3);
	if (RGB_data == 0) {
		std::cerr
			<< __FILE__ << "(" << __LINE__ << ") malloc failed to allocate array!"
            << std::endl;
		exit(-1);
	}

	memset (RGB_data, 0, IMAGE_WIDTH*IMAGE_HEIGHT*3);	
	convertU8Gray_to_U8RGB (IMAGE_WIDTH, IMAGE_HEIGHT, pbuff, RGB_data);
	saveppm_fromU8RGB("sierpinski.ppm", IMAGE_WIDTH, IMAGE_HEIGHT, RGB_data);
	free (RGB_data);

	status = clEnqueueUnmapMemObject(CLU_DEFAULT_Q, dst, pbuff, 0, 0, 0);
	OCL_CHECK(status);

	status = clReleaseCommandQueue(my_device_q);
	OCL_CHECK(status);
    status = clReleaseKernel(kern);
	OCL_CHECK(status);
    status = clReleaseProgram(prog);
	OCL_CHECK(status);
	status = clReleaseMemObject(dst);
	OCL_CHECK(status);

    cluRelease();

    printf(">> Done <<\n");

	return 0;
}
