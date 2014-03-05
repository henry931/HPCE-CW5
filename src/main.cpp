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

		uint64_t cbRaw=uint64_t(w)*h*bits/8;
		std::vector<uint64_t> raw(cbRaw/8);

		std::vector<uint32_t> pixels(w*h);

		/////////// Process by line by line ///////////////////////////////////////////

		// Size of one line
		uint64_t cbLine=uint64_t(w)*bits/8;

		// Three input buffers for each line
		std::vector<uint64_t> rawA(cbLine/8), rawB(cbLine/8), rawC(cbLine/8);

		// Buffers for lines of pixels and output
		std::vector<uint32_t> pixelsA(w), pixelsB(w), pixelsC(w), output(w);

		// Pointers for circular addressing
		//uint64_t *rawptr[3] = {&rawA[0],&rawB[0],&rawC[0]};
		uint32_t *pixelptr[3] = {&pixelsA[0],&pixelsB[0],&pixelsC[0]};
		std::vector<uint32_t> *pixelptr_vec[3] = {&pixelsA,&pixelsB,&pixelsC};

		while(1){

			// Try to read first line into A and unpack
			int line = 0;	// A vertical reference
			if( !read_blob(STDIN_FILENO, cbLine, &rawA[0]) )
				return 0;	// No image
			unpack_blob(w, 1, bits, &rawA[0], pixelptr[0]);

			// Read second line into B and unpack
			line = 1;
			read_blob(STDIN_FILENO, cbLine,&rawA[0]);
			unpack_blob(w, 1, bits, &rawA[0], pixelptr[1]);

			// Process first line
			if( levels >= 1){
				dilate_line_top(w, *pixelptr_vec[0], *pixelptr_vec[1], output);
			}
			else if( levels <= -1){
				erode_line_top(w, *pixelptr_vec[0], *pixelptr_vec[1], output);
			}

			// Pack and write first line
			pack_blob(w, 1, bits, &output[0], &rawA[0]);
			write_blob(STDOUT_FILENO, cbLine, &rawA[0]);

			/////////////////////////////////////////////

			for(line = 2; line < h; line++){

				read_blob(STDIN_FILENO, cbLine,&rawA[0]);
				unpack_blob(w, 1, bits, &rawA[0], pixelptr[line%3]);

				if( levels >= 1){
					dilate_line(w, *pixelptr_vec[(line+1)%3], *pixelptr_vec[(line+2)%3], *pixelptr_vec[line%3], output);
				}
				else if( levels <= -1){
					erode_line(w, *pixelptr_vec[(line+1)%3], *pixelptr_vec[(line+2)%3], *pixelptr_vec[line%3], output);
				}

				pack_blob(w, 1, bits, &output[0], &rawA[0]);
				write_blob(STDOUT_FILENO, cbLine, &rawA[0]);

			}

			////////////////////////////////////////////////////

			if( levels >= 1){
				dilate_line_bottom(w, *pixelptr_vec[(line+1)%3], *pixelptr_vec[(line+2)%3], output);
			}
			else if( levels <= -1){
				erode_line_bottom(w, *pixelptr_vec[(line+1)%3], *pixelptr_vec[(line+2)%3], output);
			}

			pack_blob(w, 1, bits, &output[0], &rawA[0]);
			write_blob(STDOUT_FILENO, cbLine, &rawA[0]);	

		}

		return 0;
	}catch(std::exception &e){
		std::cerr<<"Caught exception : "<<e.what()<<"\n";
		return 1;
	}
}