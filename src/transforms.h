// Header files for windows compilation
#ifdef _WIN32
#include <io.h>
#include <stdint.h>

// Header files for OSX compilation
#else
#include <unistd.h>
#endif

// Shared Headers
#include <vector>
#include <tr1/tuple>
#include "CL/cl.hpp"

void transform(int levels, unsigned w, unsigned h, unsigned bits);

void process_opencl_packed_line(int levels, unsigned w, unsigned bits,std::vector<uint32_t>& gpuReadOffsets, std::vector<uint32_t>& gpuWriteOffsets, uint32_t* pixelsIn, uint32_t* pixelsOut,std::vector<uint32_t> aboveOverrides,std::vector<uint32_t> belowOverrides, std::tr1::tuple<cl::Device,cl::Context,cl::Program> cl_instance);

std::tr1::tuple<cl::Kernel,cl::Kernel,std::vector<cl::Buffer*>,cl::CommandQueue,cl::NDRange,cl::NDRange,cl::NDRange> init_cl(int levels, unsigned w, unsigned h, unsigned bits, std::string source);