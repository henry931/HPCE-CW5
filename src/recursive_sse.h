// Header files for Windows compilation
#ifdef _WIN32
//#include <io.h>
//#include <stdint.h>

// Header files for OSX compilation
#else
#include <unistd.h>
#endif

// Shared Headers
#include <vector>

// SSE3
#include "pmmintrin.h"

void process_recursive_sse_8(const int levels, const unsigned w, const unsigned h,const int inputhandle,const int outputhandle);
void process_recursive_sse_4(const int levels, const unsigned w, const unsigned h,const int inputhandle,const int outputhandle);
void process_recursive_sse_2(const int levels, const unsigned w, const unsigned h,const int inputhandle,const int outputhandle);
void process_recursive_sse_1(const int levels, const unsigned w, const unsigned h,const int inputhandle,const int outputhandle);
void process_recursive_sse_16(const int levels, const unsigned w, const unsigned h,const int inputhandle,const int outputhandle);
void process_recursive_sse_32(const int levels, const unsigned w, const unsigned h,const int inputhandle,const int outputhandle);

void process_recursive_sse(const unsigned bits, const int levels, const unsigned w, const unsigned h,const int inputhandle,const int outputhandle);
