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

void process_opencl_packed(int levels, unsigned w, unsigned h, unsigned bits, std::vector<uint32_t> &pixels, std::tr1::tuple<cl::Device,cl::Context,cl::Program> cl_instance);

std::tr1::tuple<cl::Device,cl::Context,cl::Program> init_cl(std::string source);