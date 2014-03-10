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

std::tr1::tuple<cl::Kernel,cl::Kernel,cl::Buffer,cl::Buffer,cl::CommandQueue,cl::NDRange,cl::NDRange,cl::NDRange> init_cl(unsigned w, unsigned h, unsigned bits, std::string source)
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
    
    cl::NDRange offset(0, 0);				 // Always start iterations at x=0, y=0
    cl::NDRange globalSize((w*bits)/32, h);  // Global size must match the original loops
    cl::NDRange localSize=cl::NullRange;	 // We don't care about local size
    
    return std::tr1::make_tuple(erodeKernel,dilateKernel,buffInput,buffOutput,queue,offset,globalSize,localSize);
}

void process_opencl_packed(int levels, unsigned w, unsigned h, unsigned bits, std::vector<uint32_t> &pixels, std::tr1::tuple<cl::Kernel,cl::Kernel,cl::Buffer,cl::Buffer,cl::CommandQueue,cl::NDRange,cl::NDRange,cl::NDRange> cl_instance)
{
    size_t cbBuffer=(w*h*bits)/8;
    
    cl::Kernel erodeKernel = std::tr1::get<0>(cl_instance);
    cl::Kernel dilateKernel = std::tr1::get<1>(cl_instance);
    cl::Buffer buffInput = std::tr1::get<2>(cl_instance);
    cl::Buffer buffOutput = std::tr1::get<3>(cl_instance);
    cl::CommandQueue queue = std::tr1::get<4>(cl_instance);
    cl::NDRange offset = std::tr1::get<5>(cl_instance);
    cl::NDRange globalSize = std::tr1::get<6>(cl_instance);
    cl::NDRange localSize = std::tr1::get<7>(cl_instance);
    
    queue.enqueueWriteBuffer(buffInput, CL_TRUE, 0, cbBuffer, &pixels[0]);
    
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
    uint64_t cbRaw=uint64_t(w)*h*bits/8;
    std::vector<uint64_t> raw(cbRaw/8);
    
    std::vector<uint32_t> pixels(cbRaw/4);
    
    auto cl_instance = init_cl(w,h,bits,"kernels.cl");
    
    while(1){
        if(!read_blob(STDIN_FILENO, cbRaw, &raw[0]))
            break;	// No more images
        
        unpack_blob_32(cbRaw, &raw[0], &pixels[0]);
        
        process_opencl_packed(levels, w, h, bits, pixels, cl_instance);
        
        pack_blob_32(cbRaw, &pixels[0], &raw[0]);
        
        write_blob(STDOUT_FILENO, cbRaw, &raw[0]);
    }
}