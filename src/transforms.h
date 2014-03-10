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

void erode(unsigned w, unsigned h, const std::vector<uint32_t> &input, std::vector<uint32_t> &output);
void dilate(unsigned w, unsigned h, const std::vector<uint32_t> &input, std::vector<uint32_t> &output);

void process(int levels, unsigned w, unsigned h, unsigned /*bits*/, std::vector<uint32_t> &pixels);

void process_opencl(int levels, unsigned w, unsigned h, unsigned /*bits*/, std::vector<uint32_t> &pixels);

void process_opencl_packed(int levels, unsigned w, unsigned h, unsigned bits, std::vector<uint32_t> &pixels);

void transform(int levels, unsigned w, unsigned h, unsigned bits);