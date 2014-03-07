// Header files for windows compilation
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

#include "utilities.h"
#include "transforms.h"



int process_recursive(unsigned recursionlevel,const unsigned w,const unsigned h,const unsigned bits,const int inputhandle,const int outputhandle, std::vector<std::vector<uint32_t>> &pixelsA, std::vector<std::vector<uint32_t>> &pixelsB, std::vector<std::vector<uint32_t>> &pixelsC, std::vector<uint32_t> &line, std::vector<uint32_t> &output, std::vector<uint32_t> &status, const uint32_t toplevel){

	uint32_t *thisline = &line[recursionlevel];

	// Size of one line
	uint64_t cbLine=uint64_t(w)*bits/8;

	// Raw buffer for conversion
	std::vector<uint64_t> raw(cbLine/8);

	// Pointers for circular addressing
	uint32_t *pixelptr[3] = {&pixelsA[recursionlevel][0],&pixelsB[recursionlevel][0],&pixelsC[recursionlevel][0]};
	std::vector<uint32_t> *pixelptr_vec[3] = {&pixelsA[recursionlevel],&pixelsB[recursionlevel],&pixelsC[recursionlevel]};

	// Output buffer

	///////////////////////// Try to read first line into A /////////////////////////////
	if (*thisline == 0) {

		if (recursionlevel == 0){
			if( !read_blob(inputhandle, cbLine, &raw[0]) )
				return 0;	// No image
			unpack_blob(w, 1, bits, &raw[0], pixelptr[0]);
		}
		else{
			process_recursive(recursionlevel-1, w, h, bits, inputhandle, outputhandle, pixelsA, pixelsB, pixelsC, line, *pixelptr_vec[0], status, toplevel);
		}

		*thisline = *thisline + 1;
	}

	///////////////////////// Read second line into B and unpack ////////////////////////////
	if (*thisline == 1) {

		if (recursionlevel == 0){
			read_blob(inputhandle, cbLine,&raw[0]);
			unpack_blob(w, 1, bits, &raw[0], pixelptr[1]);
		}
		else{
			process_recursive(recursionlevel-1, w, h, bits, inputhandle, outputhandle, pixelsA, pixelsB, pixelsC, line, *pixelptr_vec[1], status, toplevel);
		}

		// Process first line
		if( status[recursionlevel] == 1){
			dilate_line_top(w, *pixelptr_vec[0], *pixelptr_vec[1], output);
		}
		else if( status[recursionlevel] == 0){
			erode_line_top(w, *pixelptr_vec[0], *pixelptr_vec[1], output);
		}

		if (recursionlevel == toplevel){
			// Pack and write first line
			pack_blob(w, 1, bits, &output[0], &raw[0]);
			write_blob(outputhandle, cbLine, &raw[0]);
		}

		*thisline = *thisline + 1;

		if (recursionlevel != toplevel){
		return 0;
		}
	}
	/////////////////////////////////////////////

	while(*thisline < h) {

		if (recursionlevel == 0){
		read_blob(inputhandle, cbLine,&raw[0]);
		unpack_blob(w, 1, bits, &raw[0], pixelptr[*thisline%3]);
		}
		else{
			process_recursive(recursionlevel-1, w, h, bits, inputhandle, outputhandle, pixelsA, pixelsB, pixelsC, line, *pixelptr_vec[*thisline%3], status, toplevel);
		}

		if( status[recursionlevel] == 1 ){
			dilate_line(w, *pixelptr_vec[(*thisline+1)%3], *pixelptr_vec[(*thisline+2)%3], *pixelptr_vec[*thisline%3], output);
		}
		else if( status[recursionlevel] == 0 ){
			erode_line(w, *pixelptr_vec[(*thisline+1)%3], *pixelptr_vec[(*thisline+2)%3], *pixelptr_vec[*thisline%3], output);
		}

		if (recursionlevel == toplevel){
		pack_blob(w, 1, bits, &output[0], &raw[0]);
		write_blob(outputhandle, cbLine, &raw[0]);
		}

		*thisline = *thisline + 1;

		if (recursionlevel != toplevel){
		return 0;
		}

	}

	////////////////////////////////////////////////////

	if( status[recursionlevel] == 1){
		dilate_line_bottom(w, *pixelptr_vec[(*thisline+1)%3], *pixelptr_vec[(*thisline+2)%3], output);
	}
	else if( status[recursionlevel] == 0 ){
		erode_line_bottom(w, *pixelptr_vec[(*thisline+1)%3], *pixelptr_vec[(*thisline+2)%3], output);
	}

	if (recursionlevel == toplevel){
	pack_blob(w, 1, bits, &output[0], &raw[0]);
	write_blob(outputhandle, cbLine, &raw[0]);
	}

}










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

		//uint64_t cbRaw=uint64_t(w)*h*bits/8;
		//std::vector<uint64_t> raw(cbRaw/8);

		//std::vector<uint32_t> pixels(w*h);

		/////////////////////////////////////
		/*
		while(1){
		if(!read_blob(STDIN_FILENO, cbRaw, &raw[0]))
		break;	// No more images
		unpack_blob(w, h, bits, &raw[0], &pixels[0]);		

		process(levels, w, h, bits, pixels);
		//invert(levels, w, h, bits, pixels);

		pack_blob(w, h, bits, &pixels[0], &raw[0]);
		write_blob(STDOUT_FILENO, cbRaw, &raw[0]);
		}
		*/

		/////////////////////////////////////////

		// Create buffers
		// The interior vector corresponds to a row of pixels
		// Exterior index corresponds to the recursion level
		std::vector<std::vector<uint32_t> > pixelsA(2*std::abs(levels), std::vector<uint32_t>(w));
		std::vector<std::vector<uint32_t> > pixelsB(2*std::abs(levels), std::vector<uint32_t>(w));
		std::vector<std::vector<uint32_t> > pixelsC(2*std::abs(levels), std::vector<uint32_t>(w));

		// Size of one line
		//uint64_t cbRaw=uint64_t(w)*bits/8;

		// A raw buffer for temporary storage during conversion
		//std::vector<uint64_t> raw(cbRaw/8);

		// An integer for the start level of recursion
		const uint32_t reclevel = 2*std::abs(levels) - 1;

		// Line count (record of circular addressing status)
		std::vector<uint32_t> line(2*std::abs(levels));

		// Vector where each element signifies if process should erode or dilate
		std::vector<uint32_t> status(2*std::abs(levels));
		if (levels >= 1){
			for(int i = 0; i < std::abs(levels); i++){
				status[i] = 1;
			};
			for(int i = std::abs(levels); i < 2*std::abs(levels); i++){
				status[i] = 0;
			};
		}
		else if (levels <= -1){
			for(int i = 0; i < std::abs(levels); i++){
				status[i] = 0;
			};
			for(int i = std::abs(levels); i < 2*std::abs(levels); i++){
				status[i] = 1;
			};
		}

		// Top level
		//const int toplevel = std::abs(levels) - 1;

		// An output buffer
		std::vector<uint32_t> outbuff(w);



		process_recursive(reclevel, w, h, bits, STDIN_FILENO, STDOUT_FILENO, pixelsA, pixelsB, pixelsC, line, outbuff, status, reclevel);

		return 0;
	}catch(std::exception &e){
		std::cerr<<"Caught exception : "<<e.what()<<"\n";
		return 1;
	}

}