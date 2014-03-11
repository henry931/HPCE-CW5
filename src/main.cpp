// Header files for Windows compilation
#ifdef _WIN32
#include <io.h>
#include <stdint.h>
#include <fcntl.h> 
#include <sys/stat.h>

// Header files for OSX compilation
#else
#include <unistd.h>
#endif

// Shared Headers
#include <algorithm>
#include <stdexcept>
#include <vector>
#include <cstdio>
#include <iostream>
#include <string>

// Recursive SSE Processes
#include "recursive_sse.h"

int main(int argc, char *argv[])
{
	// REMOVE BEFORE SUBMIT
#ifdef _WIN32
	int STDIN_FILENO, STDOUT_FILENO;
	_sopen_s(&STDIN_FILENO, "input.raw", _O_BINARY | _O_RDONLY, _SH_DENYWR, _S_IREAD);
	_sopen_s(&STDOUT_FILENO , "output.raw", _O_BINARY | _O_WRONLY | _O_TRUNC | _O_CREAT, _SH_DENYRD, _S_IWRITE);
#else
	freopen("input.raw", "r", stdin);
	freopen("output.raw", "w", stdout);
#endif
	// REMOVE BEFORE SUBMIT

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

		// Do processing
		if(bits == 8)		process_recursive_sse_8(levels, w, h,STDIN_FILENO,STDOUT_FILENO);
		else if(bits == 4)	process_recursive_sse_4(levels, w, h,STDIN_FILENO,STDOUT_FILENO);
		else if(bits == 2)	process_recursive_sse_2(levels, w, h,STDIN_FILENO,STDOUT_FILENO);
		else if(bits == 1)	process_recursive_sse_1(levels, w, h,STDIN_FILENO,STDOUT_FILENO);
		else if(bits == 16)	process_recursive_sse_16(levels, w, h,STDIN_FILENO,STDOUT_FILENO);
		else if(bits == 32)	process_recursive_sse_32(levels, w, h,STDIN_FILENO,STDOUT_FILENO);

		return 0;

	}catch(std::exception &e){
		std::cerr<<"Caught exception : "<<e.what()<<"\n";
		return 1;
	}

}