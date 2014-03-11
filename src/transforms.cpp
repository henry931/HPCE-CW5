//
//  utilities.cpp
//  Process
//
//  Created by Richard Worrall on 28/02/2014.
//
//

// Header files for windows compilation
#ifdef _WIN32
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#endif

// Shared Headers
#define __CL_ENABLE_EXCEPTIONS

#include <stdexcept>
#include <cmath>
#include <stdint.h>
#include <memory>
#include <cstdio>
#include <fstream>
#include <streambuf>
#include <iostream>

#include "transforms.h"
#include "utilities.h"

#include "tbb/parallel_for.h"
#include "tbb/task_group.h"

#include "CL/cl.hpp"

//#include "tmmintrin.h"

// SSE3
#include "pmmintrin.h"

void erode(unsigned w, unsigned h, const std::vector<uint32_t> &input, std::vector<uint32_t> &output)
{
	auto in=[&](int x, int y) -> uint32_t { return input[y*w+x]; };
	auto out=[&](int x, int y) -> uint32_t & {return output[y*w+x]; };

	for(unsigned x=0;x<w;x++){
		if(x==0){
			out(0,0)=vmin(in(0,0), in(0,1), in(1,0));
			for(unsigned y=1;y<h-1;y++){
				out(0,y)=vmin(in(0,y), in(0,y-1), in(1,y), in(0,y+1));
			}
			out(0,h-1)=vmin(in(0,h-1), in(0,h-2), in(1,h-1));
		}else if(x<w-1){
			out(x,0)=vmin(in(x,0), in(x-1,0), in(x,1), in(x+1,0));
			for(unsigned y=1;y<h-1;y++){
				out(x,y)=vmin(in(x,y), in(x-1,y), in(x,y-1), in(x,y+1), in(x+1,y));
			}
			out(x,h-1)=vmin(in(x,h-1), in(x-1,h-1), in(x,h-2), in(x+1,h-1));
		}else{
			out(w-1,0)=vmin(in(w-1,0), in(w-1,1), in(w-2,0));
			for(unsigned y=1;y<h-1;y++){
				out(w-1,y)=vmin(in(w-1,y), in(w-1,y-1), in(w-2,y), in(w-1,y+1));
			}
			out(w-1,h-1)=vmin(in(w-1,h-1), in(w-1,h-2), in(w-2,h-1));
		}
	}
}

void dilate(unsigned w, unsigned h, const std::vector<uint32_t> &input, std::vector<uint32_t> &output)
{
	auto in=[&](int x, int y) -> uint32_t { return input[y*w+x]; };
	auto out=[&](int x, int y) -> uint32_t & {return output[y*w+x]; };

	for(unsigned x=0;x<w;x++){
		if(x==0){
			out(0,0)=vmax(in(0,0), in(0,1), in(1,0));
			for(unsigned y=1;y<h-1;y++){
				out(0,y)=vmax(in(0,y), in(0,y-1), in(1,y), in(0,y+1));
			}
			out(0,h-1)=vmax(in(0,h-1), in(0,h-2), in(1,h-1));
		}else if(x<w-1){
			out(x,0)=vmax(in(x,0), in(x-1,0), in(x,1), in(x+1,0));
			for(unsigned y=1;y<h-1;y++){
				out(x,y)=vmax(in(x,y), in(x-1,y), in(x,y-1), in(x,y+1), in(x+1,y));
			}
			out(x,h-1)=vmax(in(x,h-1), in(x-1,h-1), in(x,h-2), in(x+1,h-1));
		}else{
			out(w-1,0)=vmax(in(w-1,0), in(w-1,1), in(w-2,0));
			for(unsigned y=1;y<h-1;y++){
				out(w-1,y)=vmax(in(w-1,y), in(w-1,y-1), in(w-2,y), in(w-1,y+1));
			}
			out(w-1,h-1)=vmax(in(w-1,h-1), in(w-1,h-2), in(w-2,h-1));
		}
	}
}

void process(int levels, unsigned w, unsigned h, unsigned /*bits*/, std::vector<uint32_t> &pixels)
{
	std::vector<uint32_t> buffer(w*h);

	// Depending on whether levels is positive or negative,
	// we flip the order round.
	auto fwd=levels < 0 ? erode : dilate;
	auto rev=levels < 0 ? dilate : erode;

	for(int i=0;i<std::abs(levels);i++){
		fwd(w, h, pixels, buffer);
		std::swap(pixels, buffer);
	}
	for(int i=0;i<std::abs(levels);i++){
		rev(w,h,pixels, buffer);
		std::swap(pixels, buffer);
	}
}

void process_opencl(int levels, unsigned w, unsigned h, unsigned /*bits*/, std::vector<uint32_t> &pixels)
{
	std::vector<cl::Platform> platforms;

	cl::Platform::get(&platforms);

	if(platforms.size()==0) throw std::runtime_error("No OpenCL platforms found.");

	std::cerr<<"Found "<<platforms.size()<<" platforms\n";
	for(unsigned i=0;i<platforms.size();i++){
		std::string vendor=platforms[0].getInfo<CL_PLATFORM_VENDOR>();
		std::cerr<<"  Platform "<<i<<" : "<<vendor<<"\n";
	}

	int selectedPlatform=0;
	if(getenv("HPCE_SELECT_PLATFORM")){
		selectedPlatform=atoi(getenv("HPCE_SELECT_PLATFORM"));
	}
	std::cerr<<"Choosing platform "<<selectedPlatform<<"\n";
	cl::Platform platform=platforms.at(selectedPlatform);

	std::vector<cl::Device> devices;
	platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
	if(devices.size()==0){
		throw std::runtime_error("No opencl devices found.\n");
	}

	std::cerr<<"Found "<<devices.size()<<" devices\n";
	for(unsigned i=0;i<devices.size();i++){
		std::string name=devices[i].getInfo<CL_DEVICE_NAME>();
		std::cerr<<"  Device "<<i<<" : "<<name<<"\n";
	}

	//TODO: Automatic device selection based on startup perfomance test

	int selectedDevice=0;
	if(getenv("HPCE_SELECT_DEVICE")){
		selectedDevice=atoi(getenv("HPCE_SELECT_DEVICE"));
	}
	std::cerr<<"Choosing device "<<selectedDevice<<"\n";
	cl::Device device=devices.at(selectedDevice);

	cl::Context context(devices);

	std::string kernelSource=LoadSource("kernels.cl");

	cl::Program::Sources sources;	// A vector of (data,length) pairs
	sources.push_back(std::make_pair(kernelSource.c_str(), kernelSource.size()+1));	// push on our single string

	cl::Program program(context, sources);
	try{
		program.build(devices);
	}catch(...){
		for(unsigned i=0;i<devices.size();i++){
			std::cerr<<"Log for device "<<devices[i].getInfo<CL_DEVICE_NAME>()<<":\n\n";
			std::cerr<<program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[i])<<"\n\n";
		}
		throw;
	}

	size_t cbBuffer=4*w*h;

	cl::Buffer buffInput(context, CL_MEM_READ_WRITE, cbBuffer);
	cl::Buffer buffOutput(context, CL_MEM_READ_WRITE, cbBuffer);

	cl::Kernel erodeKernel(program, "erode_kernel");
	cl::Kernel dilateKernel(program, "dilate_kernel");

	cl::CommandQueue queue(context, device);

	queue.enqueueWriteBuffer(buffInput, CL_TRUE, 0, cbBuffer, &pixels[0]);

	cl::NDRange offset(0, 0);				// Always start iterations at x=0, y=0
	cl::NDRange globalSize(w, h);           // Global size must match the original loops
	cl::NDRange localSize=cl::NullRange;	// We don't care about local size

	if (levels < 0)
	{
		for(unsigned t=0;t<abs(levels);t++){

			erodeKernel.setArg(0, buffInput);
			erodeKernel.setArg(1, buffOutput);

			queue.enqueueNDRangeKernel(erodeKernel, offset, globalSize, localSize);

			queue.enqueueBarrier();

			std::swap(buffInput, buffOutput);
		}
		for(unsigned t=0;t<abs(levels);t++){

			dilateKernel.setArg(0, buffInput);
			dilateKernel.setArg(1, buffOutput);

			queue.enqueueNDRangeKernel(dilateKernel, offset, globalSize, localSize);

			queue.enqueueBarrier();

			std::swap(buffInput, buffOutput);
		}
	}
	else
	{
		for(unsigned t=0;t<abs(levels);t++){

			dilateKernel.setArg(0, buffInput);
			dilateKernel.setArg(1, buffOutput);

			queue.enqueueNDRangeKernel(dilateKernel, offset, globalSize, localSize);

			queue.enqueueBarrier();

			std::swap(buffInput, buffOutput);
		}
		for(unsigned t=0;t<abs(levels);t++){

			erodeKernel.setArg(0, buffInput);
			erodeKernel.setArg(1, buffOutput);

			queue.enqueueNDRangeKernel(erodeKernel, offset, globalSize, localSize);

			queue.enqueueBarrier();

			std::swap(buffInput, buffOutput);
		}
	}

	queue.enqueueReadBuffer(buffInput, CL_TRUE, 0, cbBuffer, &pixels[0]);
}

void process_tbb(int levels, unsigned w, unsigned h, unsigned /*bits*/, std::vector<uint32_t> &pixels)
{
	std::vector<uint32_t> buffer(w*h);

	// Depending on whether levels is positive or negative,
	// we flip the order round.
	auto fwd=levels < 0 ? erode_parfor : dilate_parfor;
	auto rev=levels < 0 ? dilate_parfor : erode_parfor;

	for(int i=0;i<std::abs(levels);i++){
		fwd(w, h, pixels, buffer);
		std::swap(pixels, buffer);
	}
	for(int i=0;i<std::abs(levels);i++){
		rev(w,h,pixels, buffer);
		std::swap(pixels, buffer);
	}
}

void erode_parfor(unsigned w, unsigned h, const std::vector<uint32_t> &input, std::vector<uint32_t> &output)
{
	auto in=[&](int x, int y) -> uint32_t { return input[y*w+x]; };
	auto out=[&](int x, int y) -> uint32_t & {return output[y*w+x]; };

	tbb::parallel_for<unsigned>(0, w, 1, [=](unsigned x)	{
		if(x==0){
			out(0,0)=vmin(in(0,0), in(0,1), in(1,0));
			for(unsigned y=1;y<h-1;y++){
				out(0,y)=vmin(in(0,y), in(0,y-1), in(1,y), in(0,y+1));
			}
			out(0,h-1)=vmin(in(0,h-1), in(0,h-2), in(1,h-1));
		}else if(x<w-1){
			out(x,0)=vmin(in(x,0), in(x-1,0), in(x,1), in(x+1,0));
			for(unsigned y=1;y<h-1;y++){
				out(x,y)=vmin(in(x,y), in(x-1,y), in(x,y-1), in(x,y+1), in(x+1,y));
			}
			out(x,h-1)=vmin(in(x,h-1), in(x-1,h-1), in(x,h-2), in(x+1,h-1));
		}else{
			out(w-1,0)=vmin(in(w-1,0), in(w-1,1), in(w-2,0));
			for(unsigned y=1;y<h-1;y++){
				out(w-1,y)=vmin(in(w-1,y), in(w-1,y-1), in(w-2,y), in(w-1,y+1));
			}
			out(w-1,h-1)=vmin(in(w-1,h-1), in(w-1,h-2), in(w-2,h-1));
		}
	});
}

void dilate_parfor(unsigned w, unsigned h, const std::vector<uint32_t> &input, std::vector<uint32_t> &output)
{
	auto in=[&](int x, int y) -> uint32_t { return input[y*w+x]; };
	auto out=[&](int x, int y) -> uint32_t & {return output[y*w+x]; };

	tbb::parallel_for<unsigned>(0, w, 1, [=](unsigned x)	{
		if(x==0){
			out(0,0)=vmax(in(0,0), in(0,1), in(1,0));
			for(unsigned y=1;y<h-1;y++){
				out(0,y)=vmax(in(0,y), in(0,y-1), in(1,y), in(0,y+1));
			}
			out(0,h-1)=vmax(in(0,h-1), in(0,h-2), in(1,h-1));
		}else if(x<w-1){
			out(x,0)=vmax(in(x,0), in(x-1,0), in(x,1), in(x+1,0));
			for(unsigned y=1;y<h-1;y++){
				out(x,y)=vmax(in(x,y), in(x-1,y), in(x,y-1), in(x,y+1), in(x+1,y));
			}
			out(x,h-1)=vmax(in(x,h-1), in(x-1,h-1), in(x,h-2), in(x+1,h-1));
		}else{
			out(w-1,0)=vmax(in(w-1,0), in(w-1,1), in(w-2,0));
			for(unsigned y=1;y<h-1;y++){
				out(w-1,y)=vmax(in(w-1,y), in(w-1,y-1), in(w-2,y), in(w-1,y+1));
			}
			out(w-1,h-1)=vmax(in(w-1,h-1), in(w-1,h-2), in(w-2,h-1));
		}
	});
}

void erode_line(unsigned w, const std::vector<uint32_t> &inputA, const std::vector<uint32_t> &inputB, const std::vector<uint32_t> &inputC, std::vector<uint32_t> &output)
{

	output[0] = vmin(inputB[0], inputA[0], inputB[1], inputC[0]);

	output[w-1]=vmin(inputB[w-1], inputA[w-1], inputB[w-2], inputC[w-1]);

	tbb::parallel_for<unsigned>(1, w-1, 1, [&](unsigned x)	{

		output[x] = vmin(inputB[x], inputB[x-1], inputA[x], inputC[x], inputB[x+1]);
	});
}

void dilate_line(unsigned w, const std::vector<uint32_t> &inputA, const std::vector<uint32_t> &inputB, const std::vector<uint32_t> &inputC, std::vector<uint32_t> &output)
{
	output[0] = vmax(inputB[0], inputA[0], inputB[1], inputC[0]);

	output[w-1]=vmax(inputB[w-1], inputA[w-1], inputB[w-2], inputC[w-1]);

	// Inputs are ordered vertically LineA,LineB,LineC and the ouput corresponds to LineB as the origin
	tbb::parallel_for<unsigned>(1, w-1, 1, [&](unsigned x)	{

		output[x] = vmax(inputB[x], inputB[x-1], inputA[x], inputC[x], inputB[x+1]);	

	});
}

void erode_line_top(unsigned w, const std::vector<uint32_t> &inputB, const std::vector<uint32_t> &inputC, std::vector<uint32_t> &output)
{
	output[0] = vmin(inputB[0], inputB[1], inputC[0]);

	output[w-1]=vmin(inputB[w-1], inputB[w-2], inputC[w-1]);

	// Inputs are ordered vertically LineB,LineC and the ouput corresponds to LineB as the origin
	tbb::parallel_for<unsigned>(1, w-1, 1, [&](unsigned x)	{

		output[x] = vmin(inputB[x], inputB[x-1], inputC[x], inputB[x+1]);	

	});
}

void dilate_line_top(unsigned w, const std::vector<uint32_t> &inputB, const std::vector<uint32_t> &inputC, std::vector<uint32_t> &output)
{
	output[0] = vmax(inputB[0], inputB[1], inputC[0]);

	output[w-1]=vmax(inputB[w-1], inputB[w-2], inputC[w-1]);

	// Inputs are ordered vertically LineB,LineC and the ouput corresponds to LineB as the origin
	tbb::parallel_for<unsigned>(1, w-1, 1, [&](unsigned x)	{

		output[x] = vmax(inputB[x], inputB[x-1], inputC[x], inputB[x+1]);	

	});
}

void erode_line_sse_8(unsigned w, const std::vector<__m128i> &inputA, const std::vector<__m128i> &inputB, const std::vector<__m128i> &inputC, std::vector<__m128i> &output)
{
	// Each vector element now contains 16 unsigned packed ints
	unsigned num_elements = w/16;
	if(w%16 == 8) num_elements++;

	//Temporary buffers
	__m128i shiftedright, shiftedleft;
	uint8_t *shiftedright_ptr = (uint8_t*) &shiftedright;
	uint8_t *shiftedleft_ptr = (uint8_t*) &shiftedleft;
	uint8_t *next_container, *previous_container;

	// Masks
	const __m128i maskright = _mm_set_epi8(14,13,12,11,10,9,8,7,6,5,4,3,2,1,0,50);
	const __m128i maskleft = _mm_set_epi8(50,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1);

	for (unsigned i = 1; i < num_elements - 1; i++)
	{
		previous_container = (uint8_t*) &inputB[i-1];
		next_container = (uint8_t*) &inputB[i+1];
		// Shift self so we can find minimum
		shiftedright = _mm_shuffle_epi8(inputB[i], maskright);
		shiftedleft = _mm_shuffle_epi8(inputB[i], maskleft);
		// Need to bring in int from next element in vector
		shiftedright_ptr[0] = previous_container[15];
		shiftedleft_ptr[15] = next_container[0];
		// Get minimum
		output[i] = _mm_min_epu8(_mm_min_epu8(inputB[i], _mm_min_epu8(shiftedleft, shiftedright)), _mm_min_epu8(inputA[i], inputC[i]));
	}

	// When i = 0
	next_container = (uint8_t*) &inputB[1];
	// Shift self so we can find minimum
	shiftedright = _mm_shuffle_epi8(inputB[0], maskright);
	shiftedleft = _mm_shuffle_epi8(inputB[0], maskleft);
	// Need to bring in int from next element in vector
	shiftedright_ptr[0] = 255; // Nothing to bring in so assign maxmium possible value
	shiftedleft_ptr[15] = next_container[0];
	// Get minimum
	output[0] = _mm_min_epu8(_mm_min_epu8(inputB[0], _mm_min_epu8(shiftedleft, shiftedright)), _mm_min_epu8(inputA[0], inputC[0]));

	// When i = num_elements - 1
	previous_container = (uint8_t*) &inputB[num_elements-2];
	// Shift self so we can find minimum
	shiftedright = _mm_shuffle_epi8(inputB[num_elements - 1], maskright);
	shiftedleft = _mm_shuffle_epi8(inputB[num_elements - 1], maskleft);
	// Need to bring in int from next element in vector
	shiftedright_ptr[0] = previous_container[15];
	shiftedleft_ptr[15-w%16] = 255; // Accounts for w%16 == 8 case
	// Get minimum
	output[num_elements-1] = _mm_min_epu8(_mm_min_epu8(inputB[num_elements-1], _mm_min_epu8(shiftedleft, shiftedright)), _mm_min_epu8(inputA[num_elements-1], inputC[num_elements-1]));
}

void dilate_line_sse_8(unsigned w, const std::vector<__m128i> &inputA, const std::vector<__m128i> &inputB, const std::vector<__m128i> &inputC, std::vector<__m128i> &output)
{
	// Each vector element now contains 16 unsigned packed ints
	unsigned num_elements = w/16;
	if(w%16 == 8) num_elements++;

	//Temporary buffers
	__m128i shiftedright, shiftedleft;
	uint8_t *shiftedright_ptr = (uint8_t*) &shiftedright;
	uint8_t *shiftedleft_ptr = (uint8_t*) &shiftedleft;
	uint8_t *next_container, *previous_container;

	// Masks
	const __m128i maskright = _mm_set_epi8(14,13,12,11,10,9,8,7,6,5,4,3,2,1,0,50);
	const __m128i maskleft = _mm_set_epi8(50,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1);

	for (unsigned i = 1; i < num_elements - 1; i++)
	{
		previous_container = (uint8_t*) &inputB[i-1];
		next_container = (uint8_t*) &inputB[i+1];
		// Shift self so we can find maximum
		shiftedright = _mm_shuffle_epi8(inputB[i], maskright);
		shiftedleft = _mm_shuffle_epi8(inputB[i], maskleft);
		// Need to bring in int from next element in vector
		shiftedright_ptr[0] = previous_container[15];
		shiftedleft_ptr[15] = next_container[0];
		// Get maximum
		output[i] = _mm_max_epu8(_mm_max_epu8(inputB[i], _mm_max_epu8(shiftedleft, shiftedright)), _mm_max_epu8(inputA[i], inputC[i]));
	}

	// When i = 0
	next_container = (uint8_t*) &inputB[1];
	// Shift self so we can find maximum
	shiftedright = _mm_shuffle_epi8(inputB[0], maskright);
	shiftedleft = _mm_shuffle_epi8(inputB[0], maskleft);
	// Need to bring in int from next element in vector
	shiftedright_ptr[0] = 0; // Nothing to bring in so assign minmium possible value
	shiftedleft_ptr[15] = next_container[0];
	// Get maximum
	output[0] = _mm_max_epu8(_mm_max_epu8(inputB[0], _mm_max_epu8(shiftedleft, shiftedright)), _mm_max_epu8(inputA[0], inputC[0]));

	// When i = num_elements - 1
	previous_container = (uint8_t*) &inputB[num_elements-2];
	// Shift self so we can find maximum
	shiftedright = _mm_shuffle_epi8(inputB[num_elements - 1], maskright);
	shiftedleft = _mm_shuffle_epi8(inputB[num_elements - 1], maskleft);
	// Need to bring in int from next element in vector
	shiftedright_ptr[0] = previous_container[15];
	shiftedleft_ptr[15-w%16] = 0; // Accounts for w%16 == 8 case
	// Get maximum
	output[num_elements-1] = _mm_max_epu8(_mm_max_epu8(inputB[num_elements-1], _mm_max_epu8(shiftedleft, shiftedright)), _mm_max_epu8(inputA[num_elements-1], inputC[num_elements-1]));
}

void erode_line_top_sse_8(unsigned w, const std::vector<__m128i> &inputB, const std::vector<__m128i> &inputC, std::vector<__m128i> &output)
{
	// Each vector element now contains 16 unsigned packed ints
	unsigned num_elements = w/16;
	if(w%16 == 8) num_elements++;

	//Temporary buffers
	__m128i shiftedright, shiftedleft;
	uint8_t *shiftedright_ptr = (uint8_t*) &shiftedright;
	uint8_t *shiftedleft_ptr = (uint8_t*) &shiftedleft;
	uint8_t *next_container, *previous_container;

	// Masks
	const __m128i maskright = _mm_set_epi8(14,13,12,11,10,9,8,7,6,5,4,3,2,1,0,50);
	const __m128i maskleft = _mm_set_epi8(50,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1);

	for (unsigned i = 1; i < num_elements - 1; i++)
	{
		previous_container = (uint8_t*) &inputB[i-1];
		next_container = (uint8_t*) &inputB[i+1];
		// Shift self so we can find minimum
		shiftedright = _mm_shuffle_epi8(inputB[i], maskright);
		shiftedleft = _mm_shuffle_epi8(inputB[i], maskleft);
		// Need to bring in int from next element in vector
		shiftedright_ptr[0] = previous_container[15];
		shiftedleft_ptr[15] = next_container[0];
		// Get minimum
		output[i] = _mm_min_epu8(_mm_min_epu8(shiftedleft, shiftedright), _mm_min_epu8(inputB[i], inputC[i]));
	}

	// When i = 0
	next_container = (uint8_t*) &inputB[1];
	// Shift self so we can find minimum
	shiftedright = _mm_shuffle_epi8(inputB[0], maskright);
	shiftedleft = _mm_shuffle_epi8(inputB[0], maskleft);
	// Need to bring in int from next element in vector
	shiftedright_ptr[0] = 255; // Nothing to bring in so assign maxmium possible value
	shiftedleft_ptr[15] = next_container[0];
	// Get minimum
	output[0] = _mm_min_epu8(_mm_min_epu8(shiftedleft, shiftedright), _mm_min_epu8(inputB[0], inputC[0]));

	// When i = num_elements - 1
	previous_container = (uint8_t*) &inputB[num_elements-2];
	// Shift self so we can find minimum
	shiftedright = _mm_shuffle_epi8(inputB[num_elements - 1], maskright);
	shiftedleft = _mm_shuffle_epi8(inputB[num_elements - 1], maskleft);
	// Need to bring in int from next element in vector
	shiftedright_ptr[0] = previous_container[15];
	shiftedleft_ptr[15-w%16] = 255; // Accounts for w%16 == 8 case
	// Get minimum
	output[num_elements-1] = _mm_min_epu8(_mm_min_epu8(shiftedleft, shiftedright), _mm_min_epu8(inputB[num_elements-1], inputC[num_elements-1]));
}

void dilate_line_top_sse_8(unsigned w, const std::vector<__m128i> &inputB, const std::vector<__m128i> &inputC, std::vector<__m128i> &output)
{
	// Each vector element now contains 16 unsigned packed ints
	unsigned num_elements = w/16;
	if(w%16 == 8) num_elements++;

	//Temporary buffers
	__m128i shiftedright, shiftedleft;
	uint8_t *shiftedright_ptr = (uint8_t*) &shiftedright;
	uint8_t *shiftedleft_ptr = (uint8_t*) &shiftedleft;
	uint8_t *next_container, *previous_container;

	// Masks
	const __m128i maskright = _mm_set_epi8(14,13,12,11,10,9,8,7,6,5,4,3,2,1,0,50);
	const __m128i maskleft = _mm_set_epi8(50,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1);

	for (unsigned i = 1; i < num_elements - 1; i++)
	{
		previous_container = (uint8_t*) &inputB[i-1];
		next_container = (uint8_t*) &inputB[i+1];
		// Shift self so we can find maximum
		shiftedright = _mm_shuffle_epi8(inputB[i], maskright);
		shiftedleft = _mm_shuffle_epi8(inputB[i], maskleft);
		// Need to bring in int from next element in vector
		shiftedright_ptr[0] = previous_container[15];
		shiftedleft_ptr[15] = next_container[0];
		// Get maximum
		output[i] = _mm_max_epu8(_mm_max_epu8(shiftedleft, shiftedright), _mm_max_epu8(inputB[i], inputC[i]));
	}

	// When i = 0
	next_container = (uint8_t*) &inputB[1];
	// Shift self so we can find maximum
	shiftedright = _mm_shuffle_epi8(inputB[0], maskright);
	shiftedleft = _mm_shuffle_epi8(inputB[0], maskleft);
	// Need to bring in int from next element in vector
	shiftedright_ptr[0] = 0; // Nothing to bring in so assign minmium possible value
	shiftedleft_ptr[15] = next_container[0];
	// Get maximum
	output[0] = _mm_max_epu8(_mm_max_epu8(shiftedleft, shiftedright), _mm_max_epu8(inputB[0], inputC[0]));

	// When i = num_elements - 1
	previous_container = (uint8_t*) &inputB[num_elements-2];
	// Shift self so we can find maximum
	shiftedright = _mm_shuffle_epi8(inputB[num_elements - 1], maskright);
	shiftedleft = _mm_shuffle_epi8(inputB[num_elements - 1], maskleft);
	// Need to bring in int from next element in vector
	shiftedright_ptr[0] = previous_container[15];
	shiftedleft_ptr[15-w%16] = 0; // Accounts for w%16 == 8 case
	// Get maximum
	output[num_elements-1] = _mm_max_epu8( _mm_max_epu8(shiftedleft, shiftedright), _mm_max_epu8(inputB[num_elements-1], inputC[num_elements-1]));
}