// Header files for Windows compilation
#ifdef _WIN32
#include <io.h>
#include <stdint.h>
//#include <fcntl.h> 
//#include <sys/stat.h>

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

// SSE3
#include "pmmintrin.h"

// SSE instructions used - http://software.intel.com/sites/landingpage/IntrinsicsGuide/
/*
_mm_setr_epi8		SSE2	Set packed values
_mm_slli_si128		SSE2	Shift left by n bytes
_mm_srli_si128		SSE2	Shift right by n bytes
_mm_min_epu8		SSE2	Find minimum
_mm_max_epu8		SSE2	Find maximum
*/

//////////////////////////// 8-Bit ////////////////////////////////////////

void packandwriteline_sse_8(unsigned w, __m128i *input, int fd)
{
	// Minimum width is 8 so this should be fine

	// Size of one line
	uint64_t cbLine=uint64_t(w);

	// Raw buffer for conversion
	std::vector<uint64_t> raw(cbLine/8);

	const uint64_t MASK=0x00000000000000FFULL;

	uint64_t done=0;

	for(unsigned i=0;i<w/16;i++){

		uint8_t *input_ptr = (uint8_t*) &input[i];

		raw[2*i]=raw[2*i] | (uint64_t(input_ptr[0]&MASK)<< 0);
		raw[2*i]=raw[2*i] | (uint64_t(input_ptr[1]&MASK)<< 8);
		raw[2*i]=raw[2*i] | (uint64_t(input_ptr[2]&MASK)<< 16);
		raw[2*i]=raw[2*i] | (uint64_t(input_ptr[3]&MASK)<< 24);
		raw[2*i]=raw[2*i] | (uint64_t(input_ptr[4]&MASK)<< 32);
		raw[2*i]=raw[2*i] | (uint64_t(input_ptr[5]&MASK)<< 40);
		raw[2*i]=raw[2*i] | (uint64_t(input_ptr[6]&MASK)<< 48);
		raw[2*i]=raw[2*i] | (uint64_t(input_ptr[7]&MASK)<< 56);

		raw[2*i+1]=raw[2*i+1] | (uint64_t(input_ptr[8]&MASK)<< 0);
		raw[2*i+1]=raw[2*i+1] | (uint64_t(input_ptr[9]&MASK)<< 8);
		raw[2*i+1]=raw[2*i+1] | (uint64_t(input_ptr[10]&MASK)<< 16);
		raw[2*i+1]=raw[2*i+1] | (uint64_t(input_ptr[11]&MASK)<< 24);
		raw[2*i+1]=raw[2*i+1] | (uint64_t(input_ptr[12]&MASK)<< 32);
		raw[2*i+1]=raw[2*i+1] | (uint64_t(input_ptr[13]&MASK)<< 40);
		raw[2*i+1]=raw[2*i+1] | (uint64_t(input_ptr[14]&MASK)<< 48);
		raw[2*i+1]=raw[2*i+1] | (uint64_t(input_ptr[15]&MASK)<< 56);
	}

	if(w%16 == 8) // This is a possible case by the specifications
	{
		unsigned index = w/16;
		unsigned index_raw = 2*index; // We need to use index so it rounds down 

		uint8_t *input_ptr = (uint8_t*) &input[index];

		raw[index_raw]=raw[index_raw] | (uint64_t(input_ptr[0]&MASK)<< 0);
		raw[index_raw]=raw[index_raw] | (uint64_t(input_ptr[1]&MASK)<< 8);
		raw[index_raw]=raw[index_raw] | (uint64_t(input_ptr[2]&MASK)<< 16);
		raw[index_raw]=raw[index_raw] | (uint64_t(input_ptr[3]&MASK)<< 24);
		raw[index_raw]=raw[index_raw] | (uint64_t(input_ptr[4]&MASK)<< 32);
		raw[index_raw]=raw[index_raw] | (uint64_t(input_ptr[5]&MASK)<< 40);
		raw[index_raw]=raw[index_raw] | (uint64_t(input_ptr[6]&MASK)<< 48);
		raw[index_raw]=raw[index_raw] | (uint64_t(input_ptr[7]&MASK)<< 56);
	}

	while(done<cbLine){
		int todo=(int)std::min(uint64_t(1)<<30, cbLine-done);

		int got=write(fd, &raw[0]+done, todo);
		if(got<=0)
			throw std::invalid_argument("Write failure.");
		done+=got;
	}
}

int readandunpack_sse_8(unsigned w, int fd, __m128i *output)
{
	// Size of one line
	uint64_t cbLine=uint64_t(w);

	// Raw buffer for conversion
	std::vector<uint64_t> raw(cbLine/8);

	uint64_t done=0;
	while(done<cbLine){
		int todo=(int)std::min(uint64_t(1)<<30, cbLine-done);

		int got=read(fd, &raw[0]+done, todo);
		if(got==0 && done==0)
			return 5;	// end of file
		if(got<=0)
			throw std::invalid_argument("Read failure.");
		done+=got;
	}

	for(unsigned i=0;i<w/16;i++){

		output[i] = _mm_setr_epi8(
			raw[2*i] & 0x00000000000000FFULL,
			(raw[2*i] & 0x000000000000FF00ULL) >> 8,
			(raw[2*i] & 0x0000000000FF0000ULL) >> 16,
			(raw[2*i] & 0x00000000FF000000ULL) >> 24,
			(raw[2*i] & 0x000000FF00000000ULL) >> 32,
			(raw[2*i] & 0x0000FF0000000000ULL) >> 40,
			(raw[2*i] & 0x00FF000000000000ULL) >> 48,
			(raw[2*i] & 0xFF00000000000000ULL) >> 56,
			raw[2*i+1] & 0x00000000000000FFULL,
			(raw[2*i+1] & 0x000000000000FF00ULL) >> 8,
			(raw[2*i+1] & 0x0000000000FF0000ULL) >> 16,
			(raw[2*i+1] & 0x00000000FF000000ULL) >> 24,
			(raw[2*i+1] & 0x000000FF00000000ULL) >> 32,
			(raw[2*i+1] & 0x0000FF0000000000ULL) >> 40,
			(raw[2*i+1] & 0x00FF000000000000ULL) >> 48,
			(raw[2*i+1] & 0xFF00000000000000ULL) >> 56
			);
	}

	if(w%16 == 8) // This is a possible case by the specifications
	{
		unsigned index = w/16;
		unsigned index_raw = 2*index; // We need to use index so it rounds down 

		output[index] = _mm_setr_epi8(
			raw[index_raw] & 0x00000000000000FFULL,
			(raw[index_raw] & 0x000000000000FF00ULL) >> 8,
			(raw[index_raw] & 0x0000000000FF0000ULL) >> 16,
			(raw[index_raw] & 0x00000000FF000000ULL) >> 24,
			(raw[index_raw] & 0x000000FF00000000ULL) >> 32,
			(raw[index_raw] & 0x0000FF0000000000ULL) >> 40,
			(raw[index_raw] & 0x00FF000000000000ULL) >> 48,
			(raw[index_raw] & 0xFF00000000000000ULL) >> 56,
			0,0,0,0,0,0,0,0
			);
	}

	return 0;
}

void erode_line_sse_8(unsigned w, const std::vector<__m128i> &inputA, const std::vector<__m128i> &inputB, const std::vector<__m128i> &inputC, std::vector<__m128i> &output)
{
	// Each vector element now contains 16 unsigned packed ints
	unsigned num_elements = w/16;
	if(w%16 == 8) num_elements++;

	//Temporary buffers
	__m128i shiftedright, shiftedleft;
	uint8_t *shiftedright_ptr = (uint8_t*) &shiftedright;
	uint8_t *shiftedleft_ptr = (uint8_t*) &shiftedleft;
	uint8_t *next_container, *previous_container;

	for (unsigned i = 1; i < num_elements - 1; i++)
	{
		previous_container = (uint8_t*) &inputB[i-1];
		next_container = (uint8_t*) &inputB[i+1];
		// Shift self so we can find minimum
		shiftedright = _mm_slli_si128(inputB[i], 1);
		shiftedleft = _mm_srli_si128(inputB[i], 1);
		// Need to bring in int from next element in vector
		shiftedright_ptr[0] = previous_container[15];
		shiftedleft_ptr[15] = next_container[0];
		// Get minimum
		output[i] = _mm_min_epu8(_mm_min_epu8(inputB[i], _mm_min_epu8(shiftedleft, shiftedright)), _mm_min_epu8(inputA[i], inputC[i]));
	}

	// When i = 0
	next_container = (uint8_t*) &inputB[1];
	// Shift self so we can find minimum
	shiftedright = _mm_slli_si128(inputB[0], 1);
	shiftedleft = _mm_srli_si128(inputB[0], 1);
	// Need to bring in int from next element in vector
	shiftedright_ptr[0] = 255; // Nothing to bring in so assign maxmium possible value
	shiftedleft_ptr[15] = next_container[0];
	// Get minimum
	output[0] = _mm_min_epu8(_mm_min_epu8(inputB[0], _mm_min_epu8(shiftedleft, shiftedright)), _mm_min_epu8(inputA[0], inputC[0]));

	// When i = num_elements - 1
	previous_container = (uint8_t*) &inputB[num_elements-2];
	// Shift self so we can find minimum
	shiftedright = _mm_slli_si128(inputB[num_elements-1], 1);
	shiftedleft = _mm_srli_si128(inputB[num_elements-1], 1);
	// Need to bring in int from next element in vector
	shiftedright_ptr[0] = previous_container[15];
	shiftedleft_ptr[15-w%16] = 255; // Accounts for w%16 == 8 case
	// Get minimum
	output[num_elements-1] = _mm_min_epu8(_mm_min_epu8(inputB[num_elements-1], _mm_min_epu8(shiftedleft, shiftedright)), _mm_min_epu8(inputA[num_elements-1], inputC[num_elements-1]));
}

void dilate_line_sse_8(unsigned w, const std::vector<__m128i> &inputA, const std::vector<__m128i> &inputB, const std::vector<__m128i> &inputC, std::vector<__m128i> &output)
{
	// Each vector element now contains 16 unsigned packed ints
	unsigned num_elements = w/16;
	if(w%16 == 8) num_elements++;

	//Temporary buffers
	__m128i shiftedright, shiftedleft;
	uint8_t *shiftedright_ptr = (uint8_t*) &shiftedright;
	uint8_t *shiftedleft_ptr = (uint8_t*) &shiftedleft;
	uint8_t *next_container, *previous_container;

	for (unsigned i = 1; i < num_elements - 1; i++)
	{
		previous_container = (uint8_t*) &inputB[i-1];
		next_container = (uint8_t*) &inputB[i+1];
		// Shift self so we can find maximum
		shiftedright = _mm_slli_si128(inputB[i], 1);
		shiftedleft = _mm_srli_si128(inputB[i], 1);
		// Need to bring in int from next element in vector
		shiftedright_ptr[0] = previous_container[15];
		shiftedleft_ptr[15] = next_container[0];
		// Get maximum
		output[i] = _mm_max_epu8(_mm_max_epu8(inputB[i], _mm_max_epu8(shiftedleft, shiftedright)), _mm_max_epu8(inputA[i], inputC[i]));
	}

	// When i = 0
	next_container = (uint8_t*) &inputB[1];
	// Shift self so we can find maximum
	shiftedright = _mm_slli_si128(inputB[0], 1);
	shiftedleft = _mm_srli_si128(inputB[0], 1);
	// Need to bring in int from next element in vector
	shiftedright_ptr[0] = 0; // Nothing to bring in so assign minmium possible value
	shiftedleft_ptr[15] = next_container[0];
	// Get maximum
	output[0] = _mm_max_epu8(_mm_max_epu8(inputB[0], _mm_max_epu8(shiftedleft, shiftedright)), _mm_max_epu8(inputA[0], inputC[0]));

	// When i = num_elements - 1
	previous_container = (uint8_t*) &inputB[num_elements-2];
	// Shift self so we can find maximum
	shiftedright = _mm_slli_si128(inputB[num_elements-1], 1);
	shiftedleft = _mm_srli_si128(inputB[num_elements-1], 1);
	// Need to bring in int from next element in vector
	shiftedright_ptr[0] = previous_container[15];
	shiftedleft_ptr[15-w%16] = 0; // Accounts for w%16 == 8 case
	// Get maximum
	output[num_elements-1] = _mm_max_epu8(_mm_max_epu8(inputB[num_elements-1], _mm_max_epu8(shiftedleft, shiftedright)), _mm_max_epu8(inputA[num_elements-1], inputC[num_elements-1]));
}

void erode_line_top_sse_8(unsigned w, const std::vector<__m128i> &inputB, const std::vector<__m128i> &inputC, std::vector<__m128i> &output)
{
	// Each vector element now contains 16 unsigned packed ints
	unsigned num_elements = w/16;
	if(w%16 == 8) num_elements++;

	//Temporary buffers
	__m128i shiftedright, shiftedleft;
	uint8_t *shiftedright_ptr = (uint8_t*) &shiftedright;
	uint8_t *shiftedleft_ptr = (uint8_t*) &shiftedleft;
	uint8_t *next_container, *previous_container;

	for (unsigned i = 1; i < num_elements - 1; i++)
	{
		previous_container = (uint8_t*) &inputB[i-1];
		next_container = (uint8_t*) &inputB[i+1];
		// Shift self so we can find minimum
		shiftedright = _mm_slli_si128(inputB[i], 1);
		shiftedleft = _mm_srli_si128(inputB[i], 1);
		// Need to bring in int from next element in vector
		shiftedright_ptr[0] = previous_container[15];
		shiftedleft_ptr[15] = next_container[0];
		// Get minimum
		output[i] = _mm_min_epu8(_mm_min_epu8(shiftedleft, shiftedright), _mm_min_epu8(inputB[i], inputC[i]));
	}

	// When i = 0
	next_container = (uint8_t*) &inputB[1];
	// Shift self so we can find minimum
	shiftedright = _mm_slli_si128(inputB[0], 1);
	shiftedleft = _mm_srli_si128(inputB[0], 1);
	// Need to bring in int from next element in vector
	shiftedright_ptr[0] = 255; // Nothing to bring in so assign maxmium possible value
	shiftedleft_ptr[15] = next_container[0];
	// Get minimum
	output[0] = _mm_min_epu8(_mm_min_epu8(shiftedleft, shiftedright), _mm_min_epu8(inputB[0], inputC[0]));

	// When i = num_elements - 1
	previous_container = (uint8_t*) &inputB[num_elements-2];
	// Shift self so we can find minimum
	shiftedright = _mm_slli_si128(inputB[num_elements-1], 1);
	shiftedleft = _mm_srli_si128(inputB[num_elements-1], 1);
	// Need to bring in int from next element in vector
	shiftedright_ptr[0] = previous_container[15];
	shiftedleft_ptr[15-w%16] = 255; // Accounts for w%16 == 8 case
	// Get minimum
	output[num_elements-1] = _mm_min_epu8(_mm_min_epu8(shiftedleft, shiftedright), _mm_min_epu8(inputB[num_elements-1], inputC[num_elements-1]));
}

void dilate_line_top_sse_8(unsigned w, const std::vector<__m128i> &inputB, const std::vector<__m128i> &inputC, std::vector<__m128i> &output)
{
	// Each vector element now contains 16 unsigned packed ints
	unsigned num_elements = w/16;
	if(w%16 == 8) num_elements++;

	//Temporary buffers
	__m128i shiftedright, shiftedleft;
	uint8_t *shiftedright_ptr = (uint8_t*) &shiftedright;
	uint8_t *shiftedleft_ptr = (uint8_t*) &shiftedleft;
	uint8_t *next_container, *previous_container;

	for (unsigned i = 1; i < num_elements - 1; i++)
	{
		previous_container = (uint8_t*) &inputB[i-1];
		next_container = (uint8_t*) &inputB[i+1];
		// Shift self so we can find maximum
		shiftedright = _mm_slli_si128(inputB[i], 1);
		shiftedleft = _mm_srli_si128(inputB[i], 1);
		// Need to bring in int from next element in vector
		shiftedright_ptr[0] = previous_container[15];
		shiftedleft_ptr[15] = next_container[0];
		// Get maximum
		output[i] = _mm_max_epu8(_mm_max_epu8(shiftedleft, shiftedright), _mm_max_epu8(inputB[i], inputC[i]));
	}

	// When i = 0
	next_container = (uint8_t*) &inputB[1];
	// Shift self so we can find maximum
	shiftedright = _mm_slli_si128(inputB[0], 1);
	shiftedleft = _mm_srli_si128(inputB[0], 1);
	// Need to bring in int from next element in vector
	shiftedright_ptr[0] = 0; // Nothing to bring in so assign minmium possible value
	shiftedleft_ptr[15] = next_container[0];
	// Get maximum
	output[0] = _mm_max_epu8(_mm_max_epu8(shiftedleft, shiftedright), _mm_max_epu8(inputB[0], inputC[0]));

	// When i = num_elements - 1
	previous_container = (uint8_t*) &inputB[num_elements-2];
	// Shift self so we can find maximum
	shiftedright = _mm_slli_si128(inputB[num_elements-1], 1);
	shiftedleft = _mm_srli_si128(inputB[num_elements-1], 1);
	// Need to bring in int from next element in vector
	shiftedright_ptr[0] = previous_container[15];
	shiftedleft_ptr[15-w%16] = 0; // Accounts for w%16 == 8 case
	// Get maximum
	output[num_elements-1] = _mm_max_epu8(_mm_max_epu8(shiftedleft, shiftedright), _mm_max_epu8(inputB[num_elements-1], inputC[num_elements-1]));
}

int process_recursive_function_sse_8(unsigned recursionlevel,const unsigned w,const unsigned h,const int inputhandle,const int outputhandle, std::vector<std::vector<__m128i>> &pixelsA, std::vector<std::vector<__m128i>> &pixelsB, std::vector<std::vector<__m128i>> &pixelsC, std::vector<uint32_t> &line, std::vector<__m128i> &output, std::vector<uint32_t> &status, const uint32_t toplevel){

	// Minimum width is 8
	// Bits is 8

	// Get pointers for this level to inrease readability
	uint32_t *thisline = &line[recursionlevel];

	// Pointers for circular addressing
	__m128i *pixelptr[3] = {&pixelsA[recursionlevel][0],&pixelsB[recursionlevel][0],&pixelsC[recursionlevel][0]};
	std::vector<__m128i> *pixelptr_vec[3] = {&pixelsA[recursionlevel],&pixelsB[recursionlevel],&pixelsC[recursionlevel]};

	///////////////////////// Try to read first line into A /////////////////////////////
	if (*thisline == 0) {

		if (recursionlevel == 0){
			if( readandunpack_sse_8 (w ,inputhandle , pixelptr[0]) != 0 )
				return 5;	// No image
		}
		else{
			if(process_recursive_function_sse_8(recursionlevel-1, w, h, inputhandle, outputhandle, pixelsA, pixelsB, pixelsC, line, *pixelptr_vec[0], status, toplevel))
				return 5;
		}

		*thisline = *thisline + 1;
	}

	///////////////////////// Read second line into B and unpack ////////////////////////////
	if (*thisline == 1) {

		if (recursionlevel == 0){
			readandunpack_sse_8 (w ,inputhandle , pixelptr[1]);
		}
		else{
			if(process_recursive_function_sse_8(recursionlevel-1, w, h, inputhandle, outputhandle, pixelsA, pixelsB, pixelsC, line, *pixelptr_vec[1], status, toplevel))
				return 5;
		}

		// Process first line
		if( status[recursionlevel] == 1){
			dilate_line_top_sse_8(w, *pixelptr_vec[0], *pixelptr_vec[1], output);
		}
		else if( status[recursionlevel] == 0){
			erode_line_top_sse_8(w, *pixelptr_vec[0], *pixelptr_vec[1], output);
		}

		if (recursionlevel == toplevel){
			// Pack and write first line
			packandwriteline_sse_8(w, &output[0],outputhandle);
		}

		*thisline = *thisline + 1;

		if (recursionlevel != toplevel){
			return 0;
		}
	}
	//////////////////////////// Steady State Section /////////////////////////////

	while(*thisline < h) {

		if (recursionlevel == 0){
			readandunpack_sse_8 (w ,inputhandle , pixelptr[*thisline%3]);
		}
		else{
			if(process_recursive_function_sse_8(recursionlevel-1, w, h, inputhandle, outputhandle, pixelsA, pixelsB, pixelsC, line, *pixelptr_vec[*thisline%3], status, toplevel))
				return 5;
		}

		if( status[recursionlevel] == 1 ){
			dilate_line_sse_8(w, *pixelptr_vec[(*thisline+1)%3], *pixelptr_vec[(*thisline+2)%3], *pixelptr_vec[*thisline%3], output);
		}
		else if( status[recursionlevel] == 0 ){
			erode_line_sse_8(w, *pixelptr_vec[(*thisline+1)%3], *pixelptr_vec[(*thisline+2)%3], *pixelptr_vec[*thisline%3], output);
		}

		if (recursionlevel == toplevel){
			packandwriteline_sse_8(w, &output[0],outputhandle);
		}

		*thisline = *thisline + 1;

		if (recursionlevel != toplevel){
			return 0;
		}

	}

	/////////////////////////// Last row of pixels /////////////////////////

	if( status[recursionlevel] == 1){
		dilate_line_top_sse_8(w, *pixelptr_vec[(*thisline+2)%3], *pixelptr_vec[(*thisline+1)%3], output);
	}
	else if( status[recursionlevel] == 0 ){
		erode_line_top_sse_8(w, *pixelptr_vec[(*thisline+2)%3], *pixelptr_vec[(*thisline+1)%3], output);
	}

	if (recursionlevel == toplevel){
		packandwriteline_sse_8(w, &output[0],outputhandle);
	}

	return 0;

}

void process_recursive_sse_8(const int levels, const unsigned w, const unsigned h,const int inputhandle,const int outputhandle){

	// Each vector element now contains 16 unsigned packed ints
	unsigned num_elements = w/16;
	if(w%16 == 8) num_elements++;

	// Create buffers
	// The interior vector corresponds to a row of pixels
	// Exterior index corresponds to the recursion level
	std::vector<std::vector<__m128i> > pixelsA(2*std::abs(levels), std::vector<__m128i>(num_elements));
	std::vector<std::vector<__m128i> > pixelsB(2*std::abs(levels), std::vector<__m128i>(num_elements));
	std::vector<std::vector<__m128i> > pixelsC(2*std::abs(levels), std::vector<__m128i>(num_elements));

	// An integer for the start level of recursion
	const uint32_t reclevel = 2*std::abs(levels) - 1;

	// Line count (record of circular addressing status)
	std::vector<uint32_t> line(2*std::abs(levels));

	// Vector where each element signifies if process should erode or dilate
	std::vector<uint32_t> status(2*std::abs(levels));

	// 1 signifies dilate, 0 is erode
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

	// An output buffer
	std::vector<__m128i> outbuff(num_elements);

	// While for video
	while(process_recursive_function_sse_8(reclevel, w, h, inputhandle, outputhandle, pixelsA, pixelsB, pixelsC, line, outbuff, status, reclevel) == 0)
	{
		// Reset input status
		for(int i = 0; i < 2*std::abs(levels); i++){
			line[i] = 0;
		};
	}

	return;

}

//////////////////////////// 4-Bit ////////////////////////////////////////

void packandwriteline_sse_4(unsigned w, __m128i *input, int fd)
{
	// Minimum width is 8 so this should be fine

	// Size of one line
	uint64_t cbLine=uint64_t(w);

	// Raw buffer for conversion
	std::vector<uint64_t> raw(cbLine/8);

	const uint64_t MASK=0x00000000000000FFULL;

	uint64_t done=0;

	for(unsigned i=0;i<w/16;i++){

		uint8_t *input_ptr = (uint8_t*) &input[i];

		raw[2*i]=raw[2*i] | (uint64_t(input_ptr[0]&MASK)<< 0);
		raw[2*i]=raw[2*i] | (uint64_t(input_ptr[1]&MASK)<< 8);
		raw[2*i]=raw[2*i] | (uint64_t(input_ptr[2]&MASK)<< 16);
		raw[2*i]=raw[2*i] | (uint64_t(input_ptr[3]&MASK)<< 24);
		raw[2*i]=raw[2*i] | (uint64_t(input_ptr[4]&MASK)<< 32);
		raw[2*i]=raw[2*i] | (uint64_t(input_ptr[5]&MASK)<< 40);
		raw[2*i]=raw[2*i] | (uint64_t(input_ptr[6]&MASK)<< 48);
		raw[2*i]=raw[2*i] | (uint64_t(input_ptr[7]&MASK)<< 56);

		raw[2*i+1]=raw[2*i+1] | (uint64_t(input_ptr[8]&MASK)<< 0);
		raw[2*i+1]=raw[2*i+1] | (uint64_t(input_ptr[9]&MASK)<< 8);
		raw[2*i+1]=raw[2*i+1] | (uint64_t(input_ptr[10]&MASK)<< 16);
		raw[2*i+1]=raw[2*i+1] | (uint64_t(input_ptr[11]&MASK)<< 24);
		raw[2*i+1]=raw[2*i+1] | (uint64_t(input_ptr[12]&MASK)<< 32);
		raw[2*i+1]=raw[2*i+1] | (uint64_t(input_ptr[13]&MASK)<< 40);
		raw[2*i+1]=raw[2*i+1] | (uint64_t(input_ptr[14]&MASK)<< 48);
		raw[2*i+1]=raw[2*i+1] | (uint64_t(input_ptr[15]&MASK)<< 56);
	}

	if(w%16 == 8) // This is a possible case by the specifications
	{
		unsigned index = w/16;
		unsigned index_raw = 2*index; // We need to use index so it rounds down 

		uint8_t *input_ptr = (uint8_t*) &input[index];

		raw[index_raw]=raw[index_raw] | (uint64_t(input_ptr[0]&MASK)<< 0);
		raw[index_raw]=raw[index_raw] | (uint64_t(input_ptr[1]&MASK)<< 8);
		raw[index_raw]=raw[index_raw] | (uint64_t(input_ptr[2]&MASK)<< 16);
		raw[index_raw]=raw[index_raw] | (uint64_t(input_ptr[3]&MASK)<< 24);
		raw[index_raw]=raw[index_raw] | (uint64_t(input_ptr[4]&MASK)<< 32);
		raw[index_raw]=raw[index_raw] | (uint64_t(input_ptr[5]&MASK)<< 40);
		raw[index_raw]=raw[index_raw] | (uint64_t(input_ptr[6]&MASK)<< 48);
		raw[index_raw]=raw[index_raw] | (uint64_t(input_ptr[7]&MASK)<< 56);
	}

	while(done<cbLine){
		int todo=(int)std::min(uint64_t(1)<<30, cbLine-done);

		int got=write(fd, &raw[0]+done, todo);
		if(got<=0)
			throw std::invalid_argument("Write failure.");
		done+=got;
	}
}

int readandunpack_sse_4(unsigned w, int fd, __m128i *output)
{
	// Size of one line
	uint64_t cbLine=uint64_t(w);

	// Raw buffer for conversion
	std::vector<uint64_t> raw(cbLine/8);

	uint64_t done=0;
	while(done<cbLine){
		int todo=(int)std::min(uint64_t(1)<<30, cbLine-done);

		int got=read(fd, &raw[0]+done, todo);
		if(got==0 && done==0)
			return 5;	// end of file
		if(got<=0)
			throw std::invalid_argument("Read failure.");
		done+=got;
	}

	for(unsigned i=0;i<w/16;i++){

		output[i] = _mm_setr_epi8(
			raw[2*i] & 0x00000000000000FFULL,
			(raw[2*i] & 0x000000000000FF00ULL) >> 8,
			(raw[2*i] & 0x0000000000FF0000ULL) >> 16,
			(raw[2*i] & 0x00000000FF000000ULL) >> 24,
			(raw[2*i] & 0x000000FF00000000ULL) >> 32,
			(raw[2*i] & 0x0000FF0000000000ULL) >> 40,
			(raw[2*i] & 0x00FF000000000000ULL) >> 48,
			(raw[2*i] & 0xFF00000000000000ULL) >> 56,
			raw[2*i+1] & 0x00000000000000FFULL,
			(raw[2*i+1] & 0x000000000000FF00ULL) >> 8,
			(raw[2*i+1] & 0x0000000000FF0000ULL) >> 16,
			(raw[2*i+1] & 0x00000000FF000000ULL) >> 24,
			(raw[2*i+1] & 0x000000FF00000000ULL) >> 32,
			(raw[2*i+1] & 0x0000FF0000000000ULL) >> 40,
			(raw[2*i+1] & 0x00FF000000000000ULL) >> 48,
			(raw[2*i+1] & 0xFF00000000000000ULL) >> 56
			);
	}

	if(w%16 == 8) // This is a possible case by the specifications
	{
		unsigned index = w/16;
		unsigned index_raw = 2*index; // We need to use index so it rounds down 

		output[index] = _mm_setr_epi8(
			raw[index_raw] & 0x00000000000000FFULL,
			(raw[index_raw] & 0x000000000000FF00ULL) >> 8,
			(raw[index_raw] & 0x0000000000FF0000ULL) >> 16,
			(raw[index_raw] & 0x00000000FF000000ULL) >> 24,
			(raw[index_raw] & 0x000000FF00000000ULL) >> 32,
			(raw[index_raw] & 0x0000FF0000000000ULL) >> 40,
			(raw[index_raw] & 0x00FF000000000000ULL) >> 48,
			(raw[index_raw] & 0xFF00000000000000ULL) >> 56,
			0,0,0,0,0,0,0,0
			);
	}

	return 0;
}

void erode_line_sse_4(unsigned w, const std::vector<__m128i> &inputA, const std::vector<__m128i> &inputB, const std::vector<__m128i> &inputC, std::vector<__m128i> &output)
{
	// Each vector element now contains 16 unsigned packed ints
	unsigned num_elements = w/16;
	if(w%16 == 8) num_elements++;

	//Temporary buffers
	__m128i shiftedright, shiftedleft;
	uint8_t *shiftedright_ptr = (uint8_t*) &shiftedright;
	uint8_t *shiftedleft_ptr = (uint8_t*) &shiftedleft;
	uint8_t *next_container, *previous_container;

	for (unsigned i = 1; i < num_elements - 1; i++)
	{
		previous_container = (uint8_t*) &inputB[i-1];
		next_container = (uint8_t*) &inputB[i+1];
		// Shift self so we can find minimum
		shiftedright = _mm_slli_si128(inputB[i], 1);
		shiftedleft = _mm_srli_si128(inputB[i], 1);
		// Need to bring in int from next element in vector
		shiftedright_ptr[0] = previous_container[15];
		shiftedleft_ptr[15] = next_container[0];
		// Get minimum
		output[i] = _mm_min_epu8(_mm_min_epu8(inputB[i], _mm_min_epu8(shiftedleft, shiftedright)), _mm_min_epu8(inputA[i], inputC[i]));
	}

	// When i = 0
	next_container = (uint8_t*) &inputB[1];
	// Shift self so we can find minimum
	shiftedright = _mm_slli_si128(inputB[0], 1);
	shiftedleft = _mm_srli_si128(inputB[0], 1);
	// Need to bring in int from next element in vector
	shiftedright_ptr[0] = 255; // Nothing to bring in so assign maxmium possible value
	shiftedleft_ptr[15] = next_container[0];
	// Get minimum
	output[0] = _mm_min_epu8(_mm_min_epu8(inputB[0], _mm_min_epu8(shiftedleft, shiftedright)), _mm_min_epu8(inputA[0], inputC[0]));

	// When i = num_elements - 1
	previous_container = (uint8_t*) &inputB[num_elements-2];
	// Shift self so we can find minimum
	shiftedright = _mm_slli_si128(inputB[num_elements-1], 1);
	shiftedleft = _mm_srli_si128(inputB[num_elements-1], 1);
	// Need to bring in int from next element in vector
	shiftedright_ptr[0] = previous_container[15];
	shiftedleft_ptr[15-w%16] = 255; // Accounts for w%16 == 8 case
	// Get minimum
	output[num_elements-1] = _mm_min_epu8(_mm_min_epu8(inputB[num_elements-1], _mm_min_epu8(shiftedleft, shiftedright)), _mm_min_epu8(inputA[num_elements-1], inputC[num_elements-1]));
}

void dilate_line_sse_4(unsigned w, const std::vector<__m128i> &inputA, const std::vector<__m128i> &inputB, const std::vector<__m128i> &inputC, std::vector<__m128i> &output)
{
	// Each vector element now contains 16 unsigned packed ints
	unsigned num_elements = w/16;
	if(w%16 == 8) num_elements++;

	//Temporary buffers
	__m128i shiftedright, shiftedleft;
	uint8_t *shiftedright_ptr = (uint8_t*) &shiftedright;
	uint8_t *shiftedleft_ptr = (uint8_t*) &shiftedleft;
	uint8_t *next_container, *previous_container;

	for (unsigned i = 1; i < num_elements - 1; i++)
	{
		previous_container = (uint8_t*) &inputB[i-1];
		next_container = (uint8_t*) &inputB[i+1];
		// Shift self so we can find maximum
		shiftedright = _mm_slli_si128(inputB[i], 1);
		shiftedleft = _mm_srli_si128(inputB[i], 1);
		// Need to bring in int from next element in vector
		shiftedright_ptr[0] = previous_container[15];
		shiftedleft_ptr[15] = next_container[0];
		// Get maximum
		output[i] = _mm_max_epu8(_mm_max_epu8(inputB[i], _mm_max_epu8(shiftedleft, shiftedright)), _mm_max_epu8(inputA[i], inputC[i]));
	}

	// When i = 0
	next_container = (uint8_t*) &inputB[1];
	// Shift self so we can find maximum
	shiftedright = _mm_slli_si128(inputB[0], 1);
	shiftedleft = _mm_srli_si128(inputB[0], 1);
	// Need to bring in int from next element in vector
	shiftedright_ptr[0] = 0; // Nothing to bring in so assign minmium possible value
	shiftedleft_ptr[15] = next_container[0];
	// Get maximum
	output[0] = _mm_max_epu8(_mm_max_epu8(inputB[0], _mm_max_epu8(shiftedleft, shiftedright)), _mm_max_epu8(inputA[0], inputC[0]));

	// When i = num_elements - 1
	previous_container = (uint8_t*) &inputB[num_elements-2];
	// Shift self so we can find maximum
	shiftedright = _mm_slli_si128(inputB[num_elements-1], 1);
	shiftedleft = _mm_srli_si128(inputB[num_elements-1], 1);
	// Need to bring in int from next element in vector
	shiftedright_ptr[0] = previous_container[15];
	shiftedleft_ptr[15-w%16] = 0; // Accounts for w%16 == 8 case
	// Get maximum
	output[num_elements-1] = _mm_max_epu8(_mm_max_epu8(inputB[num_elements-1], _mm_max_epu8(shiftedleft, shiftedright)), _mm_max_epu8(inputA[num_elements-1], inputC[num_elements-1]));
}

void erode_line_top_sse_4(unsigned w, const std::vector<__m128i> &inputB, const std::vector<__m128i> &inputC, std::vector<__m128i> &output)
{
	// Each vector element now contains 16 unsigned packed ints
	unsigned num_elements = w/16;
	if(w%16 == 8) num_elements++;

	//Temporary buffers
	__m128i shiftedright, shiftedleft;
	uint8_t *shiftedright_ptr = (uint8_t*) &shiftedright;
	uint8_t *shiftedleft_ptr = (uint8_t*) &shiftedleft;
	uint8_t *next_container, *previous_container;

	for (unsigned i = 1; i < num_elements - 1; i++)
	{
		previous_container = (uint8_t*) &inputB[i-1];
		next_container = (uint8_t*) &inputB[i+1];
		// Shift self so we can find minimum
		shiftedright = _mm_slli_si128(inputB[i], 1);
		shiftedleft = _mm_srli_si128(inputB[i], 1);
		// Need to bring in int from next element in vector
		shiftedright_ptr[0] = previous_container[15];
		shiftedleft_ptr[15] = next_container[0];
		// Get minimum
		output[i] = _mm_min_epu8(_mm_min_epu8(shiftedleft, shiftedright), _mm_min_epu8(inputB[i], inputC[i]));
	}

	// When i = 0
	next_container = (uint8_t*) &inputB[1];
	// Shift self so we can find minimum
	shiftedright = _mm_slli_si128(inputB[0], 1);
	shiftedleft = _mm_srli_si128(inputB[0], 1);
	// Need to bring in int from next element in vector
	shiftedright_ptr[0] = 255; // Nothing to bring in so assign maxmium possible value
	shiftedleft_ptr[15] = next_container[0];
	// Get minimum
	output[0] = _mm_min_epu8(_mm_min_epu8(shiftedleft, shiftedright), _mm_min_epu8(inputB[0], inputC[0]));

	// When i = num_elements - 1
	previous_container = (uint8_t*) &inputB[num_elements-2];
	// Shift self so we can find minimum
	shiftedright = _mm_slli_si128(inputB[num_elements-1], 1);
	shiftedleft = _mm_srli_si128(inputB[num_elements-1], 1);
	// Need to bring in int from next element in vector
	shiftedright_ptr[0] = previous_container[15];
	shiftedleft_ptr[15-w%16] = 255; // Accounts for w%16 == 8 case
	// Get minimum
	output[num_elements-1] = _mm_min_epu8(_mm_min_epu8(shiftedleft, shiftedright), _mm_min_epu8(inputB[num_elements-1], inputC[num_elements-1]));
}

void dilate_line_top_sse_4(unsigned w, const std::vector<__m128i> &inputB, const std::vector<__m128i> &inputC, std::vector<__m128i> &output)
{
	// Each vector element now contains 16 unsigned packed ints
	unsigned num_elements = w/16;
	if(w%16 == 8) num_elements++;

	//Temporary buffers
	__m128i shiftedright, shiftedleft;
	uint8_t *shiftedright_ptr = (uint8_t*) &shiftedright;
	uint8_t *shiftedleft_ptr = (uint8_t*) &shiftedleft;
	uint8_t *next_container, *previous_container;

	for (unsigned i = 1; i < num_elements - 1; i++)
	{
		previous_container = (uint8_t*) &inputB[i-1];
		next_container = (uint8_t*) &inputB[i+1];
		// Shift self so we can find maximum
		shiftedright = _mm_slli_si128(inputB[i], 1);
		shiftedleft = _mm_srli_si128(inputB[i], 1);
		// Need to bring in int from next element in vector
		shiftedright_ptr[0] = previous_container[15];
		shiftedleft_ptr[15] = next_container[0];
		// Get maximum
		output[i] = _mm_max_epu8(_mm_max_epu8(shiftedleft, shiftedright), _mm_max_epu8(inputB[i], inputC[i]));
	}

	// When i = 0
	next_container = (uint8_t*) &inputB[1];
	// Shift self so we can find maximum
	shiftedright = _mm_slli_si128(inputB[0], 1);
	shiftedleft = _mm_srli_si128(inputB[0], 1);
	// Need to bring in int from next element in vector
	shiftedright_ptr[0] = 0; // Nothing to bring in so assign minmium possible value
	shiftedleft_ptr[15] = next_container[0];
	// Get maximum
	output[0] = _mm_max_epu8(_mm_max_epu8(shiftedleft, shiftedright), _mm_max_epu8(inputB[0], inputC[0]));

	// When i = num_elements - 1
	previous_container = (uint8_t*) &inputB[num_elements-2];
	// Shift self so we can find maximum
	shiftedright = _mm_slli_si128(inputB[num_elements-1], 1);
	shiftedleft = _mm_srli_si128(inputB[num_elements-1], 1);
	// Need to bring in int from next element in vector
	shiftedright_ptr[0] = previous_container[15];
	shiftedleft_ptr[15-w%16] = 0; // Accounts for w%16 == 8 case
	// Get maximum
	output[num_elements-1] = _mm_max_epu8(_mm_max_epu8(shiftedleft, shiftedright), _mm_max_epu8(inputB[num_elements-1], inputC[num_elements-1]));
}

int process_recursive_function_sse_4(unsigned recursionlevel,const unsigned w,const unsigned h,const int inputhandle,const int outputhandle, std::vector<std::vector<__m128i>> &pixelsA, std::vector<std::vector<__m128i>> &pixelsB, std::vector<std::vector<__m128i>> &pixelsC, std::vector<uint32_t> &line, std::vector<__m128i> &output, std::vector<uint32_t> &status, const uint32_t toplevel){

	// Minimum width is 8
	// Bits is 8

	// Get pointers for this level to inrease readability
	uint32_t *thisline = &line[recursionlevel];

	// Pointers for circular addressing
	__m128i *pixelptr[3] = {&pixelsA[recursionlevel][0],&pixelsB[recursionlevel][0],&pixelsC[recursionlevel][0]};
	std::vector<__m128i> *pixelptr_vec[3] = {&pixelsA[recursionlevel],&pixelsB[recursionlevel],&pixelsC[recursionlevel]};

	///////////////////////// Try to read first line into A /////////////////////////////
	if (*thisline == 0) {

		if (recursionlevel == 0){
			if( readandunpack_sse_8 (w ,inputhandle , pixelptr[0]) != 0 )
				return 5;	// No image
		}
		else{
			if(process_recursive_function_sse_8(recursionlevel-1, w, h, inputhandle, outputhandle, pixelsA, pixelsB, pixelsC, line, *pixelptr_vec[0], status, toplevel))
				return 5;
		}

		*thisline = *thisline + 1;
	}

	///////////////////////// Read second line into B and unpack ////////////////////////////
	if (*thisline == 1) {

		if (recursionlevel == 0){
			readandunpack_sse_8 (w ,inputhandle , pixelptr[1]);
		}
		else{
			if(process_recursive_function_sse_8(recursionlevel-1, w, h, inputhandle, outputhandle, pixelsA, pixelsB, pixelsC, line, *pixelptr_vec[1], status, toplevel))
				return 5;
		}

		// Process first line
		if( status[recursionlevel] == 1){
			dilate_line_top_sse_8(w, *pixelptr_vec[0], *pixelptr_vec[1], output);
		}
		else if( status[recursionlevel] == 0){
			erode_line_top_sse_8(w, *pixelptr_vec[0], *pixelptr_vec[1], output);
		}

		if (recursionlevel == toplevel){
			// Pack and write first line
			packandwriteline_sse_8(w, &output[0],outputhandle);
		}

		*thisline = *thisline + 1;

		if (recursionlevel != toplevel){
			return 0;
		}
	}
	//////////////////////////// Steady State Section /////////////////////////////

	while(*thisline < h) {

		if (recursionlevel == 0){
			readandunpack_sse_8 (w ,inputhandle , pixelptr[*thisline%3]);
		}
		else{
			if(process_recursive_function_sse_8(recursionlevel-1, w, h, inputhandle, outputhandle, pixelsA, pixelsB, pixelsC, line, *pixelptr_vec[*thisline%3], status, toplevel))
				return 5;
		}

		if( status[recursionlevel] == 1 ){
			dilate_line_sse_8(w, *pixelptr_vec[(*thisline+1)%3], *pixelptr_vec[(*thisline+2)%3], *pixelptr_vec[*thisline%3], output);
		}
		else if( status[recursionlevel] == 0 ){
			erode_line_sse_8(w, *pixelptr_vec[(*thisline+1)%3], *pixelptr_vec[(*thisline+2)%3], *pixelptr_vec[*thisline%3], output);
		}

		if (recursionlevel == toplevel){
			packandwriteline_sse_8(w, &output[0],outputhandle);
		}

		*thisline = *thisline + 1;

		if (recursionlevel != toplevel){
			return 0;
		}

	}

	/////////////////////////// Last row of pixels /////////////////////////

	if( status[recursionlevel] == 1){
		dilate_line_top_sse_8(w, *pixelptr_vec[(*thisline+2)%3], *pixelptr_vec[(*thisline+1)%3], output);
	}
	else if( status[recursionlevel] == 0 ){
		erode_line_top_sse_8(w, *pixelptr_vec[(*thisline+2)%3], *pixelptr_vec[(*thisline+1)%3], output);
	}

	if (recursionlevel == toplevel){
		packandwriteline_sse_8(w, &output[0],outputhandle);
	}

	return 0;

}

void process_recursive_sse_4(const int levels, const unsigned w, const unsigned h,const int inputhandle,const int outputhandle){

	// Each vector element now contains 16 unsigned packed ints
	unsigned num_elements = w/16;
	if(w%16 == 8) num_elements++;

	// Create buffers
	// The interior vector corresponds to a row of pixels
	// Exterior index corresponds to the recursion level
	std::vector<std::vector<__m128i> > pixelsA(2*std::abs(levels), std::vector<__m128i>(num_elements));
	std::vector<std::vector<__m128i> > pixelsB(2*std::abs(levels), std::vector<__m128i>(num_elements));
	std::vector<std::vector<__m128i> > pixelsC(2*std::abs(levels), std::vector<__m128i>(num_elements));

	// An integer for the start level of recursion
	const uint32_t reclevel = 2*std::abs(levels) - 1;

	// Line count (record of circular addressing status)
	std::vector<uint32_t> line(2*std::abs(levels));

	// Vector where each element signifies if process should erode or dilate
	std::vector<uint32_t> status(2*std::abs(levels));

	// 1 signifies dilate, 0 is erode
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

	// An output buffer
	std::vector<__m128i> outbuff(num_elements);

	// While for video
	while(process_recursive_function_sse_8(reclevel, w, h, inputhandle, outputhandle, pixelsA, pixelsB, pixelsC, line, outbuff, status, reclevel) == 0)
	{
		// Reset input status
		for(int i = 0; i < 2*std::abs(levels); i++){
			line[i] = 0;
		};
	}

	return;

}