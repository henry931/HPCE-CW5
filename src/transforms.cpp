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

#include "smmintrin.h"

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
	/*
	output[0] = vmin(inputB[0], inputA[0], inputB[1], inputC[0]);

	output[w-1]=vmin(inputB[w-1], inputA[w-1], inputB[w-2], inputC[w-1]);

	output[1] = vmin(inputB[1], inputB[0], inputA[1], inputC[1], inputB[2]);

	output[w-2] = vmin(inputB[w-2], inputB[w-3], inputA[w-2], inputC[w-2], inputB[w-1]);

	__m128i a, b, c, d, e, res;

	//tbb::parallel_for<unsigned>(0, w/4-1, 1, [&](unsigned j)	{
	for (unsigned j = 0; j < w/4-1; j++){
	a.m128i_u32[0] = inputB[4*j+2];
	a.m128i_u32[1] = inputB[4*j+3];
	a.m128i_u32[2] = inputB[4*j+4];
	a.m128i_u32[3] = inputB[4*j+5];

	b.m128i_u32[0] = inputB[4*j-1+2];
	b.m128i_u32[1] = inputB[4*j-1+3];
	b.m128i_u32[2] = inputB[4*j-1+4];
	b.m128i_u32[3] = inputB[4*j-1+5];

	c.m128i_u32[0] = inputA[4*j+2];
	c.m128i_u32[1] = inputA[4*j+3];
	c.m128i_u32[2] = inputA[4*j+4];
	c.m128i_u32[3] = inputA[4*j+5];

	d.m128i_u32[0] = inputC[4*j+2];
	d.m128i_u32[1] = inputC[4*j+3];
	d.m128i_u32[2] = inputC[4*j+4];
	d.m128i_u32[3] = inputC[4*j+5];

	e.m128i_u32[0] = inputB[4*j+1+2];
	e.m128i_u32[1] = inputB[4*j+1+3];
	e.m128i_u32[2] = inputB[4*j+1+4];
	e.m128i_u32[3] = inputB[4*j+1+5];

	a = _mm_min_epu32(a, b);
	c = _mm_min_epu32(c, d);
	res = _mm_min_epu32(a, c);
	res = _mm_min_epu32(res, e);

	output[4*j+2] = res.m128i_u32[0];
	output[4*j+3] = res.m128i_u32[1];
	output[4*j+4] = res.m128i_u32[2];
	output[4*j+5] = res.m128i_u32[3];
	//});

	}*/
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