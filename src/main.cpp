#include <unistd.h>
#include <algorithm>
#include <stdexcept>
#include <vector>
#include <cstdio>
#include <iostream>
#include <string>

#include "utilities.h"
#include "transforms.h"


int main(int argc, char *argv[])
{
	// REMOVE BEFORE SUBMIT
    freopen("input.raw", "r", stdin);
    freopen("output.raw", "w", stdout);
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
		
		uint64_t cbRaw=uint64_t(w)*h*bits/8;
		std::vector<uint64_t> raw(cbRaw/8);
		
		std::vector<uint32_t> pixels(w*h);
		
		while(1){
			if(!read_blob(STDIN_FILENO, cbRaw, &raw[0]))
				break;	// No more images
			unpack_blob(w, h, bits, &raw[0], &pixels[0]);		
			
			process_opencl(levels, w, h, bits, pixels);
			//invert(levels, w, h, bits, pixels);
			
			pack_blob(w, h, bits, &pixels[0], &raw[0]);
			write_blob(STDOUT_FILENO, cbRaw, &raw[0]);
		}
		
		return 0;
	}catch(std::exception &e){
		std::cerr<<"Caught exception : "<<e.what()<<"\n";
		return 1;
	}
}

