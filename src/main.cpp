// Shared Headers
#include <algorithm>
#include <stdexcept>
#include <vector>
#include <cstdio>
#include <iostream>
#include <string>

// Header files for windows compilation
#ifdef _WIN32
#include <io.h>
#include <stdint.h>
#include <fcntl.h> 
#include <sys/stat.h>

#define read _read
#define write _write
#define STDIN_FILENO 0
#define STDOUT_FILENO 1

void set_binary_io()
{
	_setmode(_fileno(stdin), _O_BINARY);
	_setmode(_fileno(stdout), _O_BINARY);
}

// Header files for OSX compilation
#else
#include <unistd.h>

void set_binary_io()
{}

#endif

#include "utilities.h"
#include "transforms.h"
#include "recursive_sse.h"

#define WIDE_MODE_THRESHOLD 1048576

int main(int argc, char *argv[])
{
	// Windows needs to be set for binary mode IO
	set_binary_io();
    
    try{
		if(argc<3){
			fprintf(stderr, "Usage: process width height [bits] [levels]\n");
			fprintf(stderr, "   bits=8 by default\n");
			fprintf(stderr, "   levels=1 by default\n");
			exit(1);
		}
		
		unsigned w=atoi(argv[1]);
		unsigned h=atoi(argv[2]);
		
		unsigned bits=8;
		if(argc>3){
			bits=atoi(argv[3]);
		}
		
		if(bits>32)
			throw std::invalid_argument("Bits must be <= 32.");
		
		unsigned tmp=bits;
		while(tmp!=1){
			tmp>>=1;
			if(tmp==0)
				throw std::invalid_argument("Bits must be a binary power.");
		}
		
		if( ((w*bits)%64) != 0){
			throw std::invalid_argument(" width*bits must be divisible by 64.");
		}
		
		int levels=1;
		if(argc>4){
			levels=atoi(argv[4]);
		}
		
		fprintf(stderr, "Processing %d x %d image with %d bits per pixel.\n", w, h, bits);
        
        int cl_device_count = enumerate_cl_devices();
        
        if (w >= WIDE_MODE_THRESHOLD && cl_device_count > 0 && h > 4*levels+6)
        {
            fprintf(stderr, "Attempting to use OpenCL for wide image.\n");
            
            int deviceNumber = -1;
            
            deviceNumber = test_cl_devices(levels, w, h, bits, "pipeline_kernels.cl");
            
            if (deviceNumber != -1)
            {
                transform(deviceNumber,levels,w,h,bits);
            }
            else
            {
                process_recursive_sse(bits, levels, w, h, STDIN_FILENO, STDOUT_FILENO);
            }
            
        }
        else
        {
            process_recursive_sse(bits, levels, w, h, STDIN_FILENO, STDOUT_FILENO);
        }
        
		return 0;
        
	}catch(std::exception &e){
		std::cerr<<"Caught exception : "<<e.what()<<"\n";
		return 1;
	}
}

