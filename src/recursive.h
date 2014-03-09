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

//int process_recursive_function(unsigned recursionlevel,const unsigned w,const unsigned h,const unsigned bits,const int inputhandle,const int outputhandle, std::vector<std::vector<uint32_t>> &pixelsA, std::vector<std::vector<uint32_t>> &pixelsB, std::vector<std::vector<uint32_t>> &pixelsC, std::vector<uint32_t> &line, std::vector<uint32_t> &output, std::vector<uint32_t> &status, const uint32_t toplevel);

//int process_recursive_function_8(unsigned recursionlevel,const unsigned w,const unsigned h,const unsigned bits,const int inputhandle,const int outputhandle, std::vector<std::vector<uint32_t>> &pixelsA, std::vector<std::vector<uint32_t>> &pixelsB, std::vector<std::vector<uint32_t>> &pixelsC, std::vector<uint32_t> &line, std::vector<uint32_t> &output, std::vector<uint32_t> &status, const uint32_t toplevel);

void process_recursive_8(const int levels, const unsigned w, const unsigned h,const int inputhandle,const int outputhandle);

void process_recursive(const int levels, const unsigned w, const unsigned h,const unsigned bits,const int inputhandle,const int outputhandle);

void process_recursive_sse_8(const int levels, const unsigned w, const unsigned h,const int inputhandle,const int outputhandle);