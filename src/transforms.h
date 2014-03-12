
#ifdef _WIN32
// Header files for windows compilation
#include <io.h>
#include <stdint.h>
#include <tuple>

#else
// Header files for OSX compilation
#include <unistd.h>
#include <tr1/tuple>
#endif

// Shared Headers
#include <vector>

#include "CL/cl.hpp"

int enumerate_cl_devices();

int test_cl_devices(int levels, unsigned w, unsigned h, unsigned bits, std::string source);

void transform(int deviceNumber, int levels, unsigned w, unsigned h, unsigned bits);

void process_opencl_packed_line(int levels, unsigned w, unsigned bits,std::vector<uint32_t>& gpuReadOffsets, std::vector<uint32_t>& gpuWriteOffsets, uint32_t* pixelsIn, uint32_t* pixelsOut,std::vector<uint32_t> aboveOverrides,std::vector<uint32_t> belowOverrides, std::tr1::tuple<cl::Kernel,cl::Kernel,std::vector<cl::Buffer*>,cl::CommandQueue,cl::NDRange,cl::NDRange,cl::NDRange> cl_instance);

std::tr1::tuple<cl::Kernel,cl::Kernel,std::vector<cl::Buffer*>,cl::CommandQueue,cl::NDRange,cl::NDRange,cl::NDRange> init_cl(int levels, unsigned w, unsigned h, unsigned bits, std::string source, int deviceNumber = -1);