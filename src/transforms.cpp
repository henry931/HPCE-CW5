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
#include <sys/time.h>

#include "transforms.h"
#include "utilities.h"

#include "tbb/parallel_for.h"
#include "tbb/task_group.h"

#include "CL/cl.hpp"

int enumerate_cl_devices()
{
    std::vector<cl::Platform> platforms;
    
	cl::Platform::get(&platforms);
	
    if(platforms.size()==0) return 0;
    
    int selectedPlatform=0;
	if(getenv("HPCE_SELECT_PLATFORM")){
		selectedPlatform=atoi(getenv("HPCE_SELECT_PLATFORM"));
	}
    
	cl::Platform platform=platforms.at(selectedPlatform);
    
    std::vector<cl::Device> devices;
	platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
	
    return devices.size();
}

std::tr1::tuple<cl::Kernel,cl::Kernel,std::vector<cl::Buffer*>,cl::CommandQueue,cl::NDRange,cl::NDRange,cl::NDRange> init_cl(int levels, unsigned w, unsigned h, unsigned bits, std::string source, int deviceNumber)
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
    
    int selectedDevice=0;
    
    if (deviceNumber != -1) selectedDevice = deviceNumber;
    
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
    
    size_t cbBuffer= (w*bits)/2;
    
    std::vector<cl::Buffer*> gpuBuffers;
    
    for (int i=0; i<abs(levels); i++)
    {
        gpuBuffers.push_back(new cl::Buffer(context, CL_MEM_READ_WRITE, cbBuffer));
        gpuBuffers.push_back(new cl::Buffer(context, CL_MEM_READ_WRITE, cbBuffer));
    }
    gpuBuffers.push_back(new cl::Buffer(context, CL_MEM_READ_WRITE, cbBuffer)); // ... and one for luck.
    
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
    cl::NDRange globalSize((w*bits)/32, 1);  // Global size must match the original loops
    cl::NDRange localSize=cl::NullRange;	 // We don't care about local size
    
    return std::tr1::make_tuple(erodeKernel,dilateKernel,gpuBuffers,queue,offset,globalSize,localSize);
}

int test_cl_devices(int levels, unsigned w, unsigned h, unsigned bits, std::string source)
{
    std::vector<cl::Platform> platforms;
    
	cl::Platform::get(&platforms);
	
    if(platforms.size()==0) throw std::runtime_error("No OpenCL platforms found.");
    
    int selectedPlatform=0;
	if(getenv("HPCE_SELECT_PLATFORM")){
		selectedPlatform=atoi(getenv("HPCE_SELECT_PLATFORM"));
	}

	cl::Platform platform=platforms.at(selectedPlatform);
    
    std::vector<cl::Device> devices;
	platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
	if(devices.size()==0){
		throw std::runtime_error("No opencl devices found.\n");
	}
    
    uint64_t best = 0xFFFFFFFFFFFFFFFFul;
    int best_device = -1;
    
	for(unsigned i=0;i<devices.size();i++){
        uint64_t microseconds = 0xFFFFFFFFFFFFFFFFul;
        bool failedAttempt = false;
        
        try {
            uint64_t cbinput=uint64_t(w)*bits/8;
            
            std::vector<uint32_t> gpuReadOffsets(2*levels+1,0);
            std::vector<uint32_t> gpuWriteOffsets(2*levels+1,0);
            
            std::vector<uint32_t> aboveOverrides(2*levels,0);
            std::vector<uint32_t> belowOverrides(2*levels,0);
            
            std::vector<uint32_t> unpackedInput(2*(cbinput/4));
            std::vector<uint32_t> unpackedOutput(2*(cbinput/4));
            uint32_t* unpackedInputReadptr = &unpackedInput[cbinput/4];
            uint32_t* unpackedOutputWriteptr = &unpackedOutput[0];
            
            auto cl_instance = init_cl(levels,w,h,bits,"pipeline_kernels.cl",i);
            
            timeval before, after;
            
            gettimeofday(&before, NULL);
            
            for(int i=0; i<100; i++)
            {
                process_opencl_packed_line(levels, w, bits, gpuReadOffsets, gpuWriteOffsets, unpackedInputReadptr, unpackedOutputWriteptr, aboveOverrides, belowOverrides, cl_instance);
            }
            
            gettimeofday(&after, NULL);
            
            microseconds =(after.tv_sec - before.tv_sec)*1000000L + after.tv_usec - before.tv_usec;
            
        } catch (std::exception) {
            failedAttempt = true;
        }
        if (failedAttempt) continue;
        if (microseconds < best) {
            best_device = i;
            best = microseconds;
        }
	}
    
    return best_device;
}

void process_opencl_packed_line(int levels, unsigned w, unsigned bits,std::vector<uint32_t>& gpuReadOffsets, std::vector<uint32_t>& gpuWriteOffsets, uint32_t* pixelsIn, uint32_t* pixelsOut,std::vector<uint32_t> aboveOverrides,std::vector<uint32_t> belowOverrides,std::tr1::tuple<cl::Kernel,cl::Kernel,std::vector<cl::Buffer*>,cl::CommandQueue,cl::NDRange,cl::NDRange,cl::NDRange> cl_instance)
{
    size_t cbBuffer=(w*bits)/8;
    
    cl::Kernel erodeKernel = std::tr1::get<0>(cl_instance);
    cl::Kernel dilateKernel = std::tr1::get<1>(cl_instance);
    std::vector<cl::Buffer*> gpuBuffers = std::tr1::get<2>(cl_instance);
    cl::CommandQueue queue = std::tr1::get<3>(cl_instance);
    cl::NDRange offset = std::tr1::get<4>(cl_instance);
    cl::NDRange globalSize = std::tr1::get<5>(cl_instance);
    cl::NDRange localSize = std::tr1::get<6>(cl_instance);
    
    queue.enqueueWriteBuffer(*gpuBuffers[0], CL_FALSE, cbBuffer*gpuWriteOffsets[0], cbBuffer, pixelsIn);
    
    if (levels < 0)
    {
        for(unsigned i=0;i<abs(levels);i++){
            
            erodeKernel.setArg(0, aboveOverrides[i]);
            erodeKernel.setArg(1, belowOverrides[i]);
            erodeKernel.setArg(2, gpuReadOffsets[i]);
            erodeKernel.setArg(3, gpuWriteOffsets[i+1]);
            erodeKernel.setArg(4, *gpuBuffers[i]);
            erodeKernel.setArg(5, *gpuBuffers[i+1]);
            
            queue.enqueueNDRangeKernel(erodeKernel, offset, globalSize, localSize);
        }
        for(unsigned i=abs(levels);i<2*abs(levels);i++){
            
            dilateKernel.setArg(0, aboveOverrides[i]);
            dilateKernel.setArg(1, belowOverrides[i]);
            dilateKernel.setArg(2, gpuReadOffsets[i]);
            dilateKernel.setArg(3, gpuWriteOffsets[i+1]);
            dilateKernel.setArg(4, *gpuBuffers[i]);
            dilateKernel.setArg(5, *gpuBuffers[i+1]);
            
            queue.enqueueNDRangeKernel(dilateKernel, offset, globalSize, localSize);
        }
    }
    else
    {
        for(unsigned i=0;i<abs(levels);i++){
            
            dilateKernel.setArg(0, aboveOverrides[i]);
            dilateKernel.setArg(1, belowOverrides[i]);
            dilateKernel.setArg(2, gpuReadOffsets[i]);
            dilateKernel.setArg(3, gpuWriteOffsets[i+1]);
            dilateKernel.setArg(4, *gpuBuffers[i]);
            dilateKernel.setArg(5, *gpuBuffers[i+1]);
            
            queue.enqueueNDRangeKernel(dilateKernel, offset, globalSize, localSize);
        }
        for(unsigned i=abs(levels);i<2*abs(levels);i++){
            
            erodeKernel.setArg(0, aboveOverrides[i]);
            erodeKernel.setArg(1, belowOverrides[i]);
            erodeKernel.setArg(2, gpuReadOffsets[i]);
            erodeKernel.setArg(3, gpuWriteOffsets[i+1]);
            erodeKernel.setArg(4, *gpuBuffers[i]);
            erodeKernel.setArg(5, *gpuBuffers[i+1]);
            
            queue.enqueueNDRangeKernel(erodeKernel, offset, globalSize, localSize);
        }
    }
    
    queue.enqueueReadBuffer(*gpuBuffers[2*abs(levels)], CL_TRUE, cbBuffer*gpuReadOffsets[2*abs(levels)], cbBuffer, pixelsOut);

    queue.enqueueBarrier();
}

void transform(int deviceNumber, int levels, unsigned w, unsigned h, unsigned bits)
{
    uint64_t cbinput=uint64_t(w)*bits/8;
    
    
    std::vector<uint64_t> input(2*(cbinput/8));
    std::vector<uint64_t> output(2*(cbinput/8));
    
    std::vector<uint32_t> unpackedInput(2*(cbinput/4));
    std::vector<uint32_t> unpackedOutput(2*(cbinput/4));
    
    uint64_t* inputWriteptr = &input[0];
    uint64_t* inputReadptr = &input[cbinput/8];
    
    uint32_t* unpackedInputReadptr = &unpackedInput[cbinput/4];
    uint32_t* unpackedInputWriteptr = &unpackedInput[0];

    uint32_t* unpackedOutputReadptr = &unpackedOutput[cbinput/4];
    uint32_t* unpackedOutputWriteptr = &unpackedOutput[0];
    
    uint64_t* outputWriteptr = &output[0];
    uint64_t* outputReadptr = &output[cbinput/8];
    
    std::vector<uint32_t> gpuReadOffsets(2*levels+1,0);
    std::vector<uint32_t> gpuWriteOffsets(2*levels+1,0);
    
    std::vector<uint32_t> aboveOverrides(2*levels,0);
    std::vector<uint32_t> belowOverrides(2*levels,0);
    
    int j=0;
    
    for (int i=0; i<2*levels+1; i++)
    {
        gpuWriteOffsets[i] = j;
        
        j+= 2;
        j = j - (4 & -(j >= 4));
        j = j + (4 & -(j < 0));
        
        gpuReadOffsets[i] = j;
    }
    
    auto cl_instance = init_cl(levels,w,h,bits,"pipeline_kernels.cl",deviceNumber);
    
    tbb::task_group group;
    
    bool finished = false;
    
    int fullness = 0;
    
    bool full = false;
    
    int tailEnd = 4*levels+5;
    
    int image_line = 0;
    
    while(1){
        
        for (int i=0; i<levels; i++) {
            if (image_line == 4 + 2*i) aboveOverrides[i] = 0x0 ^ -(levels < 0);
            else aboveOverrides[i] = 0xFFFFFFFF ^ -(levels < 0);
            
            if (image_line == 3 + 2*i) belowOverrides[i] = 0x0 ^ -(levels < 0);
            else belowOverrides[i] = 0xFFFFFFFF ^ -(levels < 0);
        }
        for (int i=levels; i<2*levels; i++) {
            if (image_line == 4 + 2*i) aboveOverrides[i] = 0xFFFFFFFF ^ -(levels < 0);
            else aboveOverrides[i] = 0x0 ^ -(levels < 0);
            
            if (image_line == 3 + 2*i) belowOverrides[i] = 0xFFFFFFFF ^ -(levels < 0);
            else belowOverrides[i] = 0x0 ^ -(levels < 0);
        }
        
        group.run([&](){
            if(!finished && !read_blob(STDIN_FILENO, cbinput, inputWriteptr)) finished = true;
            
            unpack_blob_32(cbinput, inputReadptr, unpackedInputWriteptr);
            
            pack_blob_32(cbinput, unpackedOutputReadptr, outputWriteptr);
            
            if (fullness >= 4*levels+6 || full)
            {
                full = true;
                write_blob(STDOUT_FILENO, cbinput, outputReadptr);
            }
            else
            {
                fullness++;
            }
        });
        
        group.run([&](){
            process_opencl_packed_line(levels, w, bits, gpuReadOffsets, gpuWriteOffsets, unpackedInputReadptr, unpackedOutputWriteptr, aboveOverrides, belowOverrides, cl_instance);
            
            for (int i=0; i<2*levels+1; i++)
            {
                int j = gpuWriteOffsets[i];
                
                j++;
                j = j - (4 & -(j >= 4));
                
                gpuWriteOffsets[i] = j;
                
                j = gpuReadOffsets[i];
                
                j++;
                j = j - (4 & -(j >= 4));
                
                gpuReadOffsets[i] = j;
            }
        });
        
        group.wait();
        
        if (tailEnd == 0) {
            break;
        }
        if (finished) tailEnd--;
        
        std::swap(inputReadptr, inputWriteptr);
        std::swap(unpackedInputReadptr, unpackedInputWriteptr);
        std::swap(unpackedOutputReadptr, unpackedOutputWriteptr);
        std::swap(outputReadptr, outputWriteptr);
        
        image_line++;
        if (image_line == h) image_line = 0;
    }
}