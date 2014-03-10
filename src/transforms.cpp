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
#include <tr1/tuple>


#include "transforms.h"
#include "utilities.h"

#include "tbb/parallel_for.h"
#include "tbb/task_group.h"

#include "CL/cl.hpp"

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

std::tr1::tuple<cl::Device,cl::Context,cl::Program> init_cl(std::string source)
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
    
    std::string kernelSource=LoadSource(source.c_str());
	
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
    
    return std::tr1::make_tuple(device,context,program);

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

void process_opencl_packed(int levels, unsigned w, unsigned h, unsigned bits, std::vector<uint32_t> &pixels)
{
    auto cl_instance = init_cl("kernels.cl");
    
    cl::Device device = std::tr1::get<0>(cl_instance);
    cl::Context context = std::tr1::get<1>(cl_instance);
    cl::Program program = std::tr1::get<2>(cl_instance);
    
    size_t cbBuffer=(w*h*bits)/8;
    
    cl::Buffer buffInput(context, CL_MEM_READ_WRITE, cbBuffer);
	cl::Buffer buffOutput(context, CL_MEM_READ_WRITE, cbBuffer);
    
    std::string erodeKernelName;
    std::string dilateKernelName;
    
    switch (bits) {
        case 1:
            erodeKernelName = "erode_kernel_1";
            dilateKernelName = "dilate_kernel_1";
            break;
        case 2:
            erodeKernelName = "erode_kernel_2";
            dilateKernelName = "dilate_kernel_2";
            break;
        case 4:
            erodeKernelName = "erode_kernel_4";
            dilateKernelName = "dilate_kernel_4";
            break;
        case 8:
            erodeKernelName = "erode_kernel_8";
            dilateKernelName = "dilate_kernel_8";
            break;
        case 16:
            erodeKernelName = "erode_kernel_16";
            dilateKernelName = "dilate_kernel_16";
            break;
        case 32:
            erodeKernelName = "erode_kernel_32";
            dilateKernelName = "dilate_kernel_32";
            break;
        default:
            break;
    }
    
    cl::Kernel erodeKernel(program, erodeKernelName.c_str());
    cl::Kernel dilateKernel(program, dilateKernelName.c_str());
    
    cl::CommandQueue queue(context, device);
    
    queue.enqueueWriteBuffer(buffInput, CL_TRUE, 0, cbBuffer, &pixels[0]);
    
    cl::NDRange offset(0, 0);				 // Always start iterations at x=0, y=0
    cl::NDRange globalSize((w*bits)/32, h);  // Global size must match the original loops
    cl::NDRange localSize=cl::NullRange;	 // We don't care about local size
    
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

void transform(int levels, unsigned w, unsigned h, unsigned bits)
{
    size_t cbBuffer=3*4*w;
    
    auto cl_instance = init_cl("kernels.cl");
    
    cl::Device device = std::tr1::get<0>(cl_instance);
    cl::Context context = std::tr1::get<1>(cl_instance);
    cl::Program program = std::tr1::get<2>(cl_instance);
    
    //cl::Buffer buffInput(context, CL_MEM_READ_WRITE, cbBuffer);
	//cl::Buffer buffOutput(context, CL_MEM_READ_WRITE, cbBuffer);
    
    //cl::Kernel erodeKernel(program, "erode_kernel");
    //cl::Kernel dilateKernel(program, "dilate_kernel");
    
    //cl::CommandQueue queue(context, device);
    
    //queue.enqueueWriteBuffer(buffInput, CL_TRUE, 0, cbBuffer, &pixels[0]);
    
    cl::NDRange offset(0, 0);				// Always start iterations at x=0, y=0
    cl::NDRange globalSize(w, 1);           // Global size must match the original loops
    cl::NDRange localSize=cl::NullRange;	// We don't care about local size
    
    std::vector<cl::Buffer*> erodeBuffers(levels);
    std::vector<cl::Buffer*> dilateBuffers(levels);
    std::vector<uint32_t> erodeBufferOffsets;
    std::vector<uint32_t> dilateBufferOffsets;
    
    for (int i=0; i<levels; i++)
    {
        erodeBuffers.push_back(new cl::Buffer(context, CL_MEM_READ_WRITE, cbBuffer));
        dilateBuffers.push_back(new cl::Buffer(context, CL_MEM_READ_WRITE, cbBuffer));
        erodeBufferOffsets.push_back(0);
        dilateBufferOffsets.push_back(0);
    }
    
    cl::Kernel erodeKernelPipelined(program, "erode_kernel_pipelined");
    cl::Kernel dilateKernelPipelined(program, "dilate_kernel_pipelined");
    
    uint64_t cbRaw=uint64_t(w)*bits/8;
    
    std::vector<uint64_t> raw(cbRaw/8);
    std::vector<uint32_t> pixels(w);

    auto fwd = levels > 0 ? dilateKernelPipelined : erodeKernelPipelined;
    auto reverse = levels < 0 ? erodeKernelPipelined : dilateKernelPipelined;
    
    cl::CommandQueue queue(context, device);
    
    while(1){
        
        if(!read_blob(STDIN_FILENO, cbRaw, &raw[0]))
            break;	// No more images
        unpack_blob(w, h, bits, &raw[0], &pixels[0]);
        /*
        queue.enqueueWriteBuffer(buffInput, CL_TRUE, erodeBufferOffsets[0], cbBuffer, &pixels[0]);
        erodeBufferOffsets[0] += 4*w;
        if (erodeBufferOffsets[0] == cbBuffer) erodeBufferOffsets[0] = 0;
        
        for(int i=0;i<levels;i++)
        {
            
        }
        for(int i=0;i<levels;i++)
        {
            
        }
        
        queue.enqueueReadBuffer(buffInput, CL_TRUE, 0, cbBuffer, &pixels[0]);
        */
        //invert(levels, w, h, bits, pixels);
        
        pack_blob(w, h, bits, &pixels[0], &raw[0]);
        write_blob(STDOUT_FILENO, cbRaw, &raw[0]);
    }

}
