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
_mm_setr_epi8		SSE2	Set packed values (8-bit)
_mm_slli_si128		SSE2	Shift left by n bytes
_mm_srli_si128		SSE2	Shift right by n bytes
_mm_min_epu8		SSE2	Find minimum
_mm_max_epu8		SSE2	Find maximum
_mm_set_epi16		SSE2	Set packed values (16-bit)
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
	// Minimum width is 16 so this should be fine

	// Size of one line
	uint64_t cbLine=uint64_t(w/2);

	// Raw buffer for conversion
	std::vector<uint64_t> raw(cbLine/8);

	const uint64_t MASK=0x000000000000000FULL;

	uint64_t done=0;

	for(unsigned i=0;i<w/16;i++){

		uint8_t *input_ptr = (uint8_t*) &input[i];

		raw[i]=raw[i] | (uint64_t(input_ptr[0]&MASK)<< 0);
		raw[i]=raw[i] | (uint64_t(input_ptr[1]&MASK)<< 4);
		raw[i]=raw[i] | (uint64_t(input_ptr[2]&MASK)<< 8);
		raw[i]=raw[i] | (uint64_t(input_ptr[3]&MASK)<< 12);
		raw[i]=raw[i] | (uint64_t(input_ptr[4]&MASK)<< 16);
		raw[i]=raw[i] | (uint64_t(input_ptr[5]&MASK)<< 20);
		raw[i]=raw[i] | (uint64_t(input_ptr[6]&MASK)<< 24);
		raw[i]=raw[i] | (uint64_t(input_ptr[7]&MASK)<< 28);
		raw[i]=raw[i] | (uint64_t(input_ptr[8]&MASK)<< 32);
		raw[i]=raw[i] | (uint64_t(input_ptr[9]&MASK)<< 36);
		raw[i]=raw[i] | (uint64_t(input_ptr[10]&MASK)<< 40);
		raw[i]=raw[i] | (uint64_t(input_ptr[11]&MASK)<< 44);
		raw[i]=raw[i] | (uint64_t(input_ptr[12]&MASK)<< 48);
		raw[i]=raw[i] | (uint64_t(input_ptr[13]&MASK)<< 52);
		raw[i]=raw[i] | (uint64_t(input_ptr[14]&MASK)<< 56);
		raw[i]=raw[i] | (uint64_t(input_ptr[15]&MASK)<< 60);

		// Twiddle
		raw[i]=((raw[i]&0x0f0f0f0f0f0f0f0full)<<4) | ((raw[i]&0xf0f0f0f0f0f0f0f0ull)>>4);

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
	uint64_t cbLine=uint64_t(w/2);

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
		// Twiddle
		raw[i] = ((raw[i] & 0x0f0f0f0f0f0f0f0full)<<4) | ((raw[i] & 0xf0f0f0f0f0f0f0f0ull)>>4);

		output[i] = _mm_setr_epi8(
			raw[i] & 0x000000000000000FULL,
			(raw[i] & 0x00000000000000F0ULL) >> 4,
			(raw[i] & 0x0000000000000F00ULL) >> 8,
			(raw[i] & 0x000000000000F000ULL) >> 12,
			(raw[i] & 0x00000000000F0000ULL) >> 16,
			(raw[i] & 0x0000000000F00000ULL) >> 20,
			(raw[i] & 0x000000000F000000ULL) >> 24,
			(raw[i] & 0x00000000F0000000ULL) >> 28,
			(raw[i] & 0x0000000F00000000ULL) >> 32,
			(raw[i] & 0x000000F000000000ULL) >> 36,
			(raw[i] & 0x00000F0000000000ULL) >> 40,
			(raw[i] & 0x0000F00000000000ULL) >> 44,
			(raw[i] & 0x000F000000000000ULL) >> 48,
			(raw[i] & 0x00F0000000000000ULL) >> 52,
			(raw[i] & 0x0F00000000000000ULL) >> 56,
			(raw[i] & 0xF000000000000000ULL) >> 60
			);

	}

	return 0;
}

void erode_line_sse_4(unsigned w, const std::vector<__m128i> &inputA, const std::vector<__m128i> &inputB, const std::vector<__m128i> &inputC, std::vector<__m128i> &output)
{
	// Each vector element now contains 16 unsigned packed ints
	unsigned num_elements = w/16;

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
	shiftedright_ptr[0] = 15; // Nothing to bring in so assign maxmium possible value
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
	shiftedleft_ptr[15] = 15; // Nothing to bring in so assign maxmium possible value
	// Get minimum
	output[num_elements-1] = _mm_min_epu8(_mm_min_epu8(inputB[num_elements-1], _mm_min_epu8(shiftedleft, shiftedright)), _mm_min_epu8(inputA[num_elements-1], inputC[num_elements-1]));
}

void dilate_line_sse_4(unsigned w, const std::vector<__m128i> &inputA, const std::vector<__m128i> &inputB, const std::vector<__m128i> &inputC, std::vector<__m128i> &output)
{
	// Each vector element now contains 16 unsigned packed ints
	unsigned num_elements = w/16;

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
	shiftedleft_ptr[15] = 0; // Nothing to bring in so assign minmium possible value
	// Get maximum
	output[num_elements-1] = _mm_max_epu8(_mm_max_epu8(inputB[num_elements-1], _mm_max_epu8(shiftedleft, shiftedright)), _mm_max_epu8(inputA[num_elements-1], inputC[num_elements-1]));
}

void erode_line_top_sse_4(unsigned w, const std::vector<__m128i> &inputB, const std::vector<__m128i> &inputC, std::vector<__m128i> &output)
{
	// Each vector element now contains 16 unsigned packed ints
	unsigned num_elements = w/16;

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
	shiftedright_ptr[0] = 15; // Nothing to bring in so assign maxmium possible value
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
	shiftedleft_ptr[15] = 15; // Nothing to bring in so assign maxmium possible value
	// Get minimum
	output[num_elements-1] = _mm_min_epu8(_mm_min_epu8(shiftedleft, shiftedright), _mm_min_epu8(inputB[num_elements-1], inputC[num_elements-1]));
}

void dilate_line_top_sse_4(unsigned w, const std::vector<__m128i> &inputB, const std::vector<__m128i> &inputC, std::vector<__m128i> &output)
{
	// Each vector element now contains 16 unsigned packed ints
	unsigned num_elements = w/16;

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
	shiftedleft_ptr[15] = 0; // Nothing to bring in so assign minmium possible value
	// Get maximum
	output[num_elements-1] = _mm_max_epu8(_mm_max_epu8(shiftedleft, shiftedright), _mm_max_epu8(inputB[num_elements-1], inputC[num_elements-1]));
}

int process_recursive_function_sse_4(unsigned recursionlevel,const unsigned w,const unsigned h,const int inputhandle,const int outputhandle, std::vector<std::vector<__m128i>> &pixelsA, std::vector<std::vector<__m128i>> &pixelsB, std::vector<std::vector<__m128i>> &pixelsC, std::vector<uint32_t> &line, std::vector<__m128i> &output, std::vector<uint32_t> &status, const uint32_t toplevel){

	// Minimum width is 16
	// Bits is 4

	// Get pointers for this level to inrease readability
	uint32_t *thisline = &line[recursionlevel];

	// Pointers for circular addressing
	__m128i *pixelptr[3] = {&pixelsA[recursionlevel][0],&pixelsB[recursionlevel][0],&pixelsC[recursionlevel][0]};
	std::vector<__m128i> *pixelptr_vec[3] = {&pixelsA[recursionlevel],&pixelsB[recursionlevel],&pixelsC[recursionlevel]};

	///////////////////////// Try to read first line into A /////////////////////////////
	if (*thisline == 0) {

		if (recursionlevel == 0){
			if( readandunpack_sse_4 (w ,inputhandle , pixelptr[0]) != 0 )
				return 5;	// No image
		}
		else{
			if(process_recursive_function_sse_4(recursionlevel-1, w, h, inputhandle, outputhandle, pixelsA, pixelsB, pixelsC, line, *pixelptr_vec[0], status, toplevel))
				return 5;
		}

		*thisline = *thisline + 1;
	}

	///////////////////////// Read second line into B and unpack ////////////////////////////
	if (*thisline == 1) {

		if (recursionlevel == 0){
			readandunpack_sse_4 (w ,inputhandle , pixelptr[1]);
		}
		else{
			if(process_recursive_function_sse_4(recursionlevel-1, w, h, inputhandle, outputhandle, pixelsA, pixelsB, pixelsC, line, *pixelptr_vec[1], status, toplevel))
				return 5;
		}

		// Process first line
		if( status[recursionlevel] == 1){
			dilate_line_top_sse_4(w, *pixelptr_vec[0], *pixelptr_vec[1], output);
		}
		else if( status[recursionlevel] == 0){
			erode_line_top_sse_4(w, *pixelptr_vec[0], *pixelptr_vec[1], output);
		}

		if (recursionlevel == toplevel){
			// Pack and write first line
			packandwriteline_sse_4(w, &output[0],outputhandle);
		}

		*thisline = *thisline + 1;

		if (recursionlevel != toplevel){
			return 0;
		}
	}
	//////////////////////////// Steady State Section /////////////////////////////

	while(*thisline < h) {

		if (recursionlevel == 0){
			readandunpack_sse_4 (w ,inputhandle , pixelptr[*thisline%3]);
		}
		else{
			if(process_recursive_function_sse_4(recursionlevel-1, w, h, inputhandle, outputhandle, pixelsA, pixelsB, pixelsC, line, *pixelptr_vec[*thisline%3], status, toplevel))
				return 5;
		}

		if( status[recursionlevel] == 1 ){
			dilate_line_sse_4(w, *pixelptr_vec[(*thisline+1)%3], *pixelptr_vec[(*thisline+2)%3], *pixelptr_vec[*thisline%3], output);
		}
		else if( status[recursionlevel] == 0 ){
			erode_line_sse_4(w, *pixelptr_vec[(*thisline+1)%3], *pixelptr_vec[(*thisline+2)%3], *pixelptr_vec[*thisline%3], output);
		}

		if (recursionlevel == toplevel){
			packandwriteline_sse_4(w, &output[0],outputhandle);
		}

		*thisline = *thisline + 1;

		if (recursionlevel != toplevel){
			return 0;
		}

	}

	/////////////////////////// Last row of pixels /////////////////////////

	if( status[recursionlevel] == 1){
		dilate_line_top_sse_4(w, *pixelptr_vec[(*thisline+2)%3], *pixelptr_vec[(*thisline+1)%3], output);
	}
	else if( status[recursionlevel] == 0 ){
		erode_line_top_sse_4(w, *pixelptr_vec[(*thisline+2)%3], *pixelptr_vec[(*thisline+1)%3], output);
	}

	if (recursionlevel == toplevel){
		packandwriteline_sse_4(w, &output[0],outputhandle);
	}

	return 0;

}

void process_recursive_sse_4(const int levels, const unsigned w, const unsigned h,const int inputhandle,const int outputhandle){

	// Each vector element now contains 16 unsigned packed ints
	unsigned num_elements = w/16;

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
	while(process_recursive_function_sse_4(reclevel, w, h, inputhandle, outputhandle, pixelsA, pixelsB, pixelsC, line, outbuff, status, reclevel) == 0)
	{
		// Reset input status
		for(int i = 0; i < 2*std::abs(levels); i++){
			line[i] = 0;
		};
	}

	return;

}

//////////////////////////// 2-Bit ////////////////////////////////////////

void packandwriteline_sse_2(unsigned w, __m128i *input, int fd)
{
	// Minimum width is 16 so this should be fine

	// Size of one line
	uint64_t cbLine=uint64_t(w/4);

	// Raw buffer for conversion
	std::vector<uint64_t> raw(cbLine/8);

	const uint64_t MASK=0x0000000000000003ULL;

	uint64_t done=0;

	for(unsigned i=0;i<w/32;i++){

		uint8_t *input_ptr = (uint8_t*) &input[2*i];
		uint8_t *input_ptr2 = (uint8_t*) &input[2*i+1];

		raw[i]=raw[i] | (uint64_t(input_ptr[0]&MASK)<< 0);
		raw[i]=raw[i] | (uint64_t(input_ptr[1]&MASK)<< 2);
		raw[i]=raw[i] | (uint64_t(input_ptr[2]&MASK)<< 4);
		raw[i]=raw[i] | (uint64_t(input_ptr[3]&MASK)<< 6);
		raw[i]=raw[i] | (uint64_t(input_ptr[4]&MASK)<< 8);
		raw[i]=raw[i] | (uint64_t(input_ptr[5]&MASK)<< 10);
		raw[i]=raw[i] | (uint64_t(input_ptr[6]&MASK)<< 12);
		raw[i]=raw[i] | (uint64_t(input_ptr[7]&MASK)<< 14);
		raw[i]=raw[i] | (uint64_t(input_ptr[8]&MASK)<< 16);
		raw[i]=raw[i] | (uint64_t(input_ptr[9]&MASK)<< 18);
		raw[i]=raw[i] | (uint64_t(input_ptr[10]&MASK)<< 20);
		raw[i]=raw[i] | (uint64_t(input_ptr[11]&MASK)<< 22);
		raw[i]=raw[i] | (uint64_t(input_ptr[12]&MASK)<< 24);
		raw[i]=raw[i] | (uint64_t(input_ptr[13]&MASK)<< 26);
		raw[i]=raw[i] | (uint64_t(input_ptr[14]&MASK)<< 28);
		raw[i]=raw[i] | (uint64_t(input_ptr[15]&MASK)<< 30);

		raw[i]=raw[i] | (uint64_t(input_ptr2[0]&MASK)<< 32);
		raw[i]=raw[i] | (uint64_t(input_ptr2[1]&MASK)<< 34);
		raw[i]=raw[i] | (uint64_t(input_ptr2[2]&MASK)<< 36);
		raw[i]=raw[i] | (uint64_t(input_ptr2[3]&MASK)<< 38);
		raw[i]=raw[i] | (uint64_t(input_ptr2[4]&MASK)<< 40);
		raw[i]=raw[i] | (uint64_t(input_ptr2[5]&MASK)<< 42);
		raw[i]=raw[i] | (uint64_t(input_ptr2[6]&MASK)<< 44);
		raw[i]=raw[i] | (uint64_t(input_ptr2[7]&MASK)<< 46);
		raw[i]=raw[i] | (uint64_t(input_ptr2[8]&MASK)<< 48);
		raw[i]=raw[i] | (uint64_t(input_ptr2[9]&MASK)<< 50);
		raw[i]=raw[i] | (uint64_t(input_ptr2[10]&MASK)<< 52);
		raw[i]=raw[i] | (uint64_t(input_ptr2[11]&MASK)<< 54);
		raw[i]=raw[i] | (uint64_t(input_ptr2[12]&MASK)<< 56);
		raw[i]=raw[i] | (uint64_t(input_ptr2[13]&MASK)<< 58);
		raw[i]=raw[i] | (uint64_t(input_ptr2[14]&MASK)<< 60);
		raw[i]=raw[i] | (uint64_t(input_ptr2[15]&MASK)<< 62);

		// Twiddle
		raw[i]=	((raw[i]&0x0303030303030303ull)<<6)
			|	((raw[i]&0x0c0c0c0c0c0c0c0cull)<<2)
			|	((raw[i]&0x3030303030303030ull)>>2)
			|	((raw[i]&0xc0c0c0c0c0c0c0c0ull)>>6);

	}

	while(done<cbLine){
		int todo=(int)std::min(uint64_t(1)<<30, cbLine-done);

		int got=write(fd, &raw[0]+done, todo);
		if(got<=0)
			throw std::invalid_argument("Write failure.");
		done+=got;
	}
}

int readandunpack_sse_2(unsigned w, int fd, __m128i *output)
{
	// Size of one line
	uint64_t cbLine=uint64_t(w/4);

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

	for(unsigned i=0;i<w/32;i++){
		// Twiddle
		raw[i]=	((raw[i]&0x0303030303030303ull)<<6)
			|	((raw[i]&0x0c0c0c0c0c0c0c0cull)<<2)
			|	((raw[i]&0x3030303030303030ull)>>2)
			|	((raw[i]&0xc0c0c0c0c0c0c0c0ull)>>6);

		output[2*i] = _mm_setr_epi8(
			raw[i] & 0x0000000000000003ULL,
			(raw[i] & 0x000000000000000CULL) >> 2,
			(raw[i] & 0x0000000000000030ULL) >> 4,
			(raw[i] & 0x00000000000000C0ULL) >> 6,
			(raw[i] & 0x0000000000000300ULL) >> 8,
			(raw[i] & 0x0000000000000C00ULL) >> 10,
			(raw[i] & 0x0000000000003000ULL) >> 12,
			(raw[i] & 0x000000000000C000ULL) >> 14,
			(raw[i] & 0x0000000000030000ULL) >> 16,
			(raw[i] & 0x00000000000C0000ULL) >> 18,
			(raw[i] & 0x0000000000300000ULL) >> 20,
			(raw[i] & 0x0000000000C00000ULL) >> 22,
			(raw[i] & 0x0000000003000000ULL) >> 24,
			(raw[i] & 0x000000000C000000ULL) >> 26,
			(raw[i] & 0x0000000030000000ULL) >> 28,
			(raw[i] & 0x00000000C0000000ULL) >> 30
			);

		output[2*i+1] = _mm_setr_epi8(
			(raw[i] & 0x0000000300000000ULL) >> 32,
			(raw[i] & 0x0000000C00000000ULL) >> 34,
			(raw[i] & 0x0000003000000000ULL) >> 36,
			(raw[i] & 0x000000C000000000ULL) >> 38,
			(raw[i] & 0x0000030000000000ULL) >> 40,
			(raw[i] & 0x00000C0000000000ULL) >> 42,
			(raw[i] & 0x0000300000000000ULL) >> 44,
			(raw[i] & 0x0000C00000000000ULL) >> 46,
			(raw[i] & 0x0003000000000000ULL) >> 48,
			(raw[i] & 0x000C000000000000ULL) >> 50,
			(raw[i] & 0x0030000000000000ULL) >> 52,
			(raw[i] & 0x00C0000000000000ULL) >> 54,
			(raw[i] & 0x0300000000000000ULL) >> 56,
			(raw[i] & 0x0C00000000000000ULL) >> 58,
			(raw[i] & 0x3000000000000000ULL) >> 60,
			(raw[i] & 0xC000000000000000ULL) >> 62
			);

	}

	return 0;
}

void erode_line_sse_2(unsigned w, const std::vector<__m128i> &inputA, const std::vector<__m128i> &inputB, const std::vector<__m128i> &inputC, std::vector<__m128i> &output)
{
	// Each vector element now contains 16 unsigned packed ints
	unsigned num_elements = w/16;

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
	shiftedright_ptr[0] = 3; // Nothing to bring in so assign maxmium possible value
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
	shiftedleft_ptr[15] = 3; // Nothing to bring in so assign maxmium possible value
	// Get minimum
	output[num_elements-1] = _mm_min_epu8(_mm_min_epu8(inputB[num_elements-1], _mm_min_epu8(shiftedleft, shiftedright)), _mm_min_epu8(inputA[num_elements-1], inputC[num_elements-1]));
}

void dilate_line_sse_2(unsigned w, const std::vector<__m128i> &inputA, const std::vector<__m128i> &inputB, const std::vector<__m128i> &inputC, std::vector<__m128i> &output)
{
	// Each vector element now contains 16 unsigned packed ints
	unsigned num_elements = w/16;

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
	shiftedleft_ptr[15] = 0; // Nothing to bring in so assign minmium possible value
	// Get maximum
	output[num_elements-1] = _mm_max_epu8(_mm_max_epu8(inputB[num_elements-1], _mm_max_epu8(shiftedleft, shiftedright)), _mm_max_epu8(inputA[num_elements-1], inputC[num_elements-1]));
}

void erode_line_top_sse_2(unsigned w, const std::vector<__m128i> &inputB, const std::vector<__m128i> &inputC, std::vector<__m128i> &output)
{
	// Each vector element now contains 16 unsigned packed ints
	unsigned num_elements = w/16;

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
	shiftedright_ptr[0] = 3; // Nothing to bring in so assign maxmium possible value
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
	shiftedleft_ptr[15] = 3; // Nothing to bring in so assign maxmium possible value
	// Get minimum
	output[num_elements-1] = _mm_min_epu8(_mm_min_epu8(shiftedleft, shiftedright), _mm_min_epu8(inputB[num_elements-1], inputC[num_elements-1]));
}

void dilate_line_top_sse_2(unsigned w, const std::vector<__m128i> &inputB, const std::vector<__m128i> &inputC, std::vector<__m128i> &output)
{
	// Each vector element now contains 16 unsigned packed ints
	unsigned num_elements = w/16;

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
	shiftedleft_ptr[15] = 0; // Nothing to bring in so assign minmium possible value
	// Get maximum
	output[num_elements-1] = _mm_max_epu8(_mm_max_epu8(shiftedleft, shiftedright), _mm_max_epu8(inputB[num_elements-1], inputC[num_elements-1]));
}

int process_recursive_function_sse_2(unsigned recursionlevel,const unsigned w,const unsigned h,const int inputhandle,const int outputhandle, std::vector<std::vector<__m128i>> &pixelsA, std::vector<std::vector<__m128i>> &pixelsB, std::vector<std::vector<__m128i>> &pixelsC, std::vector<uint32_t> &line, std::vector<__m128i> &output, std::vector<uint32_t> &status, const uint32_t toplevel){

	// Minimum width is 32
	// Bits is 2

	// Get pointers for this level to inrease readability
	uint32_t *thisline = &line[recursionlevel];

	// Pointers for circular addressing
	__m128i *pixelptr[3] = {&pixelsA[recursionlevel][0],&pixelsB[recursionlevel][0],&pixelsC[recursionlevel][0]};
	std::vector<__m128i> *pixelptr_vec[3] = {&pixelsA[recursionlevel],&pixelsB[recursionlevel],&pixelsC[recursionlevel]};

	///////////////////////// Try to read first line into A /////////////////////////////
	if (*thisline == 0) {

		if (recursionlevel == 0){
			if( readandunpack_sse_2 (w ,inputhandle , pixelptr[0]) != 0 )
				return 5;	// No image
		}
		else{
			if(process_recursive_function_sse_2(recursionlevel-1, w, h, inputhandle, outputhandle, pixelsA, pixelsB, pixelsC, line, *pixelptr_vec[0], status, toplevel))
				return 5;
		}

		*thisline = *thisline + 1;
	}

	///////////////////////// Read second line into B and unpack ////////////////////////////
	if (*thisline == 1) {

		if (recursionlevel == 0){
			readandunpack_sse_2 (w ,inputhandle , pixelptr[1]);
		}
		else{
			if(process_recursive_function_sse_2(recursionlevel-1, w, h, inputhandle, outputhandle, pixelsA, pixelsB, pixelsC, line, *pixelptr_vec[1], status, toplevel))
				return 5;
		}

		// Process first line
		if( status[recursionlevel] == 1){
			dilate_line_top_sse_2(w, *pixelptr_vec[0], *pixelptr_vec[1], output);
		}
		else if( status[recursionlevel] == 0){
			erode_line_top_sse_2(w, *pixelptr_vec[0], *pixelptr_vec[1], output);
		}

		if (recursionlevel == toplevel){
			// Pack and write first line
			packandwriteline_sse_2(w, &output[0],outputhandle);
		}

		*thisline = *thisline + 1;

		if (recursionlevel != toplevel){
			return 0;
		}
	}
	//////////////////////////// Steady State Section /////////////////////////////

	while(*thisline < h) {

		if (recursionlevel == 0){
			readandunpack_sse_2 (w ,inputhandle , pixelptr[*thisline%3]);
		}
		else{
			if(process_recursive_function_sse_2(recursionlevel-1, w, h, inputhandle, outputhandle, pixelsA, pixelsB, pixelsC, line, *pixelptr_vec[*thisline%3], status, toplevel))
				return 5;
		}

		if( status[recursionlevel] == 1 ){
			dilate_line_sse_2(w, *pixelptr_vec[(*thisline+1)%3], *pixelptr_vec[(*thisline+2)%3], *pixelptr_vec[*thisline%3], output);
		}
		else if( status[recursionlevel] == 0 ){
			erode_line_sse_2(w, *pixelptr_vec[(*thisline+1)%3], *pixelptr_vec[(*thisline+2)%3], *pixelptr_vec[*thisline%3], output);
		}

		if (recursionlevel == toplevel){
			packandwriteline_sse_2(w, &output[0],outputhandle);
		}

		*thisline = *thisline + 1;

		if (recursionlevel != toplevel){
			return 0;
		}

	}

	/////////////////////////// Last row of pixels /////////////////////////

	if( status[recursionlevel] == 1){
		dilate_line_top_sse_2(w, *pixelptr_vec[(*thisline+2)%3], *pixelptr_vec[(*thisline+1)%3], output);
	}
	else if( status[recursionlevel] == 0 ){
		erode_line_top_sse_2(w, *pixelptr_vec[(*thisline+2)%3], *pixelptr_vec[(*thisline+1)%3], output);
	}

	if (recursionlevel == toplevel){
		packandwriteline_sse_2(w, &output[0],outputhandle);
	}

	return 0;

}

void process_recursive_sse_2(const int levels, const unsigned w, const unsigned h,const int inputhandle,const int outputhandle){

	// Each vector element now contains 16 unsigned packed ints
	unsigned num_elements = w/16;

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
	while(process_recursive_function_sse_2(reclevel, w, h, inputhandle, outputhandle, pixelsA, pixelsB, pixelsC, line, outbuff, status, reclevel) == 0)
	{
		// Reset input status
		for(int i = 0; i < 2*std::abs(levels); i++){
			line[i] = 0;
		};
	}

	return;

}

//////////////////////////// 1-Bit ////////////////////////////////////////

void packandwriteline_sse_1(unsigned w, __m128i *input, int fd)
{
	// Minimum width is 64 so this should be fine

	// Size of one line
	uint64_t cbLine=uint64_t(w/8);

	// Raw buffer for conversion
	std::vector<uint64_t> raw(cbLine/8);

	const uint64_t MASK=0x0000000000000001ULL;

	uint64_t done=0;

	for(unsigned i=0;i<w/64;i++){

		uint8_t *input_ptr = (uint8_t*) &input[4*i];
		uint8_t *input_ptr2 = (uint8_t*) &input[4*i+1];
		uint8_t *input_ptr3 = (uint8_t*) &input[4*i+2];
		uint8_t *input_ptr4 = (uint8_t*) &input[4*i+3];

		raw[i]=raw[i] | (uint64_t(input_ptr[0]&MASK)<< 0);
		raw[i]=raw[i] | (uint64_t(input_ptr[1]&MASK)<< 1);
		raw[i]=raw[i] | (uint64_t(input_ptr[2]&MASK)<< 2);
		raw[i]=raw[i] | (uint64_t(input_ptr[3]&MASK)<< 3);
		raw[i]=raw[i] | (uint64_t(input_ptr[4]&MASK)<< 4);
		raw[i]=raw[i] | (uint64_t(input_ptr[5]&MASK)<< 5);
		raw[i]=raw[i] | (uint64_t(input_ptr[6]&MASK)<< 6);
		raw[i]=raw[i] | (uint64_t(input_ptr[7]&MASK)<< 7);
		raw[i]=raw[i] | (uint64_t(input_ptr[8]&MASK)<< 8);
		raw[i]=raw[i] | (uint64_t(input_ptr[9]&MASK)<< 9);
		raw[i]=raw[i] | (uint64_t(input_ptr[10]&MASK)<< 10);
		raw[i]=raw[i] | (uint64_t(input_ptr[11]&MASK)<< 11);
		raw[i]=raw[i] | (uint64_t(input_ptr[12]&MASK)<< 12);
		raw[i]=raw[i] | (uint64_t(input_ptr[13]&MASK)<< 13);
		raw[i]=raw[i] | (uint64_t(input_ptr[14]&MASK)<< 14);
		raw[i]=raw[i] | (uint64_t(input_ptr[15]&MASK)<< 15);

		raw[i]=raw[i] | (uint64_t(input_ptr2[0]&MASK)<< 16);
		raw[i]=raw[i] | (uint64_t(input_ptr2[1]&MASK)<< 17);
		raw[i]=raw[i] | (uint64_t(input_ptr2[2]&MASK)<< 18);
		raw[i]=raw[i] | (uint64_t(input_ptr2[3]&MASK)<< 19);
		raw[i]=raw[i] | (uint64_t(input_ptr2[4]&MASK)<< 20);
		raw[i]=raw[i] | (uint64_t(input_ptr2[5]&MASK)<< 21);
		raw[i]=raw[i] | (uint64_t(input_ptr2[6]&MASK)<< 22);
		raw[i]=raw[i] | (uint64_t(input_ptr2[7]&MASK)<< 23);
		raw[i]=raw[i] | (uint64_t(input_ptr2[8]&MASK)<< 24);
		raw[i]=raw[i] | (uint64_t(input_ptr2[9]&MASK)<< 25);
		raw[i]=raw[i] | (uint64_t(input_ptr2[10]&MASK)<< 26);
		raw[i]=raw[i] | (uint64_t(input_ptr2[11]&MASK)<< 27);
		raw[i]=raw[i] | (uint64_t(input_ptr2[12]&MASK)<< 28);
		raw[i]=raw[i] | (uint64_t(input_ptr2[13]&MASK)<< 29);
		raw[i]=raw[i] | (uint64_t(input_ptr2[14]&MASK)<< 30);
		raw[i]=raw[i] | (uint64_t(input_ptr2[15]&MASK)<< 31);

		raw[i]=raw[i] | (uint64_t(input_ptr3[0]&MASK)<< 32);
		raw[i]=raw[i] | (uint64_t(input_ptr3[1]&MASK)<< 33);
		raw[i]=raw[i] | (uint64_t(input_ptr3[2]&MASK)<< 34);
		raw[i]=raw[i] | (uint64_t(input_ptr3[3]&MASK)<< 35);
		raw[i]=raw[i] | (uint64_t(input_ptr3[4]&MASK)<< 36);
		raw[i]=raw[i] | (uint64_t(input_ptr3[5]&MASK)<< 37);
		raw[i]=raw[i] | (uint64_t(input_ptr3[6]&MASK)<< 38);
		raw[i]=raw[i] | (uint64_t(input_ptr3[7]&MASK)<< 39);
		raw[i]=raw[i] | (uint64_t(input_ptr3[8]&MASK)<< 40);
		raw[i]=raw[i] | (uint64_t(input_ptr3[9]&MASK)<< 41);
		raw[i]=raw[i] | (uint64_t(input_ptr3[10]&MASK)<< 42);
		raw[i]=raw[i] | (uint64_t(input_ptr3[11]&MASK)<< 43);
		raw[i]=raw[i] | (uint64_t(input_ptr3[12]&MASK)<< 44);
		raw[i]=raw[i] | (uint64_t(input_ptr3[13]&MASK)<< 45);
		raw[i]=raw[i] | (uint64_t(input_ptr3[14]&MASK)<< 46);
		raw[i]=raw[i] | (uint64_t(input_ptr3[15]&MASK)<< 47);

		raw[i]=raw[i] | (uint64_t(input_ptr4[0]&MASK)<< 48);
		raw[i]=raw[i] | (uint64_t(input_ptr4[1]&MASK)<< 49);
		raw[i]=raw[i] | (uint64_t(input_ptr4[2]&MASK)<< 50);
		raw[i]=raw[i] | (uint64_t(input_ptr4[3]&MASK)<< 51);
		raw[i]=raw[i] | (uint64_t(input_ptr4[4]&MASK)<< 52);
		raw[i]=raw[i] | (uint64_t(input_ptr4[5]&MASK)<< 53);
		raw[i]=raw[i] | (uint64_t(input_ptr4[6]&MASK)<< 54);
		raw[i]=raw[i] | (uint64_t(input_ptr4[7]&MASK)<< 55);
		raw[i]=raw[i] | (uint64_t(input_ptr4[8]&MASK)<< 56);
		raw[i]=raw[i] | (uint64_t(input_ptr4[9]&MASK)<< 57);
		raw[i]=raw[i] | (uint64_t(input_ptr4[10]&MASK)<< 58);
		raw[i]=raw[i] | (uint64_t(input_ptr4[11]&MASK)<< 59);
		raw[i]=raw[i] | (uint64_t(input_ptr4[12]&MASK)<< 60);
		raw[i]=raw[i] | (uint64_t(input_ptr4[13]&MASK)<< 61);
		raw[i]=raw[i] | (uint64_t(input_ptr4[14]&MASK)<< 62);
		raw[i]=raw[i] | (uint64_t(input_ptr4[15]&MASK)<< 63);

		// Twiddle
		raw[i]=	((raw[i]&0x0101010101010101ull)<<7)
			|	((raw[i]&0x0202020202020202ull)<<5)
			|	((raw[i]&0x0404040404040404ull)<<3)
			|	((raw[i]&0x0808080808080808ull)<<1)
			|	((raw[i]&0x1010101010101010ull)>>1)
			|	((raw[i]&0x2020202020202020ull)>>3)
			|	((raw[i]&0x4040404040404040ull)>>5)
			|	((raw[i]&0x8080808080808080ull)>>7);

	}

	while(done<cbLine){
		int todo=(int)std::min(uint64_t(1)<<30, cbLine-done);

		int got=write(fd, &raw[0]+done, todo);
		if(got<=0)
			throw std::invalid_argument("Write failure.");
		done+=got;
	}
}

int readandunpack_sse_1(unsigned w, int fd, __m128i *output)
{
	// Size of one line
	uint64_t cbLine=uint64_t(w/8);

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

	for(unsigned i=0;i<w/64;i++){
		// Twiddle
		raw[i]=	((raw[i]&0x0101010101010101ull)<<7)
			|	((raw[i]&0x0202020202020202ull)<<5)
			|	((raw[i]&0x0404040404040404ull)<<3)
			|	((raw[i]&0x0808080808080808ull)<<1)
			|	((raw[i]&0x1010101010101010ull)>>1)
			|	((raw[i]&0x2020202020202020ull)>>3)
			|	((raw[i]&0x4040404040404040ull)>>5)
			|	((raw[i]&0x8080808080808080ull)>>7);

		output[4*i] = _mm_setr_epi8(
			raw[i] & 0x0000000000000001ULL,
			(raw[i] & 0x0000000000000002ULL) >> 1,
			(raw[i] & 0x0000000000000004ULL) >> 2,
			(raw[i] & 0x0000000000000008ULL) >> 3,
			(raw[i] & 0x0000000000000010ULL) >> 4,
			(raw[i] & 0x0000000000000020ULL) >> 5,
			(raw[i] & 0x0000000000000040ULL) >> 6,
			(raw[i] & 0x0000000000000080ULL) >> 7,
			(raw[i] & 0x0000000000000100ULL) >> 8,
			(raw[i] & 0x0000000000000200ULL) >> 9,
			(raw[i] & 0x0000000000000400ULL) >> 10,
			(raw[i] & 0x0000000000000800ULL) >> 11,
			(raw[i] & 0x0000000000001000ULL) >> 12,
			(raw[i] & 0x0000000000002000ULL) >> 13,
			(raw[i] & 0x0000000000004000ULL) >> 14,
			(raw[i] & 0x0000000000008000ULL) >> 15
			);

		output[4*i+1] = _mm_setr_epi8(
			(raw[i] & 0x0000000000010000ULL) >> 16,
			(raw[i] & 0x0000000000020000ULL) >> 17,
			(raw[i] & 0x0000000000040000ULL) >> 18,
			(raw[i] & 0x0000000000080000ULL) >> 19,
			(raw[i] & 0x0000000000100000ULL) >> 20,
			(raw[i] & 0x0000000000200000ULL) >> 21,
			(raw[i] & 0x0000000000400000ULL) >> 22,
			(raw[i] & 0x0000000000800000ULL) >> 23,
			(raw[i] & 0x0000000001000000ULL) >> 24,
			(raw[i] & 0x0000000002000000ULL) >> 25,
			(raw[i] & 0x0000000004000000ULL) >> 26,
			(raw[i] & 0x0000000008000000ULL) >> 27,
			(raw[i] & 0x0000000010000000ULL) >> 28,
			(raw[i] & 0x0000000020000000ULL) >> 29,
			(raw[i] & 0x0000000040000000ULL) >> 30,
			(raw[i] & 0x0000000080000000ULL) >> 31
			);

		output[4*i+2] = _mm_setr_epi8(
			(raw[i] & 0x0000000100000000ULL) >> 32,
			(raw[i] & 0x0000000200000000ULL) >> 33,
			(raw[i] & 0x0000000400000000ULL) >> 34,
			(raw[i] & 0x0000000800000000ULL) >> 35,
			(raw[i] & 0x0000001000000000ULL) >> 36,
			(raw[i] & 0x0000002000000000ULL) >> 37,
			(raw[i] & 0x0000004000000000ULL) >> 38,
			(raw[i] & 0x0000008000000000ULL) >> 39,
			(raw[i] & 0x0000010000000000ULL) >> 40,
			(raw[i] & 0x0000020000000000ULL) >> 41,
			(raw[i] & 0x0000040000000000ULL) >> 42,
			(raw[i] & 0x0000080000000000ULL) >> 43,
			(raw[i] & 0x0000100000000000ULL) >> 44,
			(raw[i] & 0x0000200000000000ULL) >> 45,
			(raw[i] & 0x0000400000000000ULL) >> 46,
			(raw[i] & 0x0000800000000000ULL) >> 47
			);

		output[4*i+3] = _mm_setr_epi8(
			(raw[i] & 0x0001000000000000ULL) >> 48,
			(raw[i] & 0x0002000000000000ULL) >> 49,
			(raw[i] & 0x0004000000000000ULL) >> 50,
			(raw[i] & 0x0008000000000000ULL) >> 51,
			(raw[i] & 0x0010000000000000ULL) >> 52,
			(raw[i] & 0x0020000000000000ULL) >> 53,
			(raw[i] & 0x0040000000000000ULL) >> 54,
			(raw[i] & 0x0080000000000000ULL) >> 55,
			(raw[i] & 0x0100000000000000ULL) >> 56,
			(raw[i] & 0x0200000000000000ULL) >> 57,
			(raw[i] & 0x0400000000000000ULL) >> 58,
			(raw[i] & 0x0800000000000000ULL) >> 59,
			(raw[i] & 0x1000000000000000ULL) >> 60,
			(raw[i] & 0x2000000000000000ULL) >> 61,
			(raw[i] & 0x4000000000000000ULL) >> 62,
			(raw[i] & 0x8000000000000000ULL) >> 63
			);

	}

	return 0;
}

void erode_line_sse_1(unsigned w, const std::vector<__m128i> &inputA, const std::vector<__m128i> &inputB, const std::vector<__m128i> &inputC, std::vector<__m128i> &output)
{
	// Each vector element now contains 16 unsigned packed ints
	unsigned num_elements = w/16;

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
	shiftedright_ptr[0] = 1; // Nothing to bring in so assign maxmium possible value
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
	shiftedleft_ptr[15] = 1; // Nothing to bring in so assign maxmium possible value
	// Get minimum
	output[num_elements-1] = _mm_min_epu8(_mm_min_epu8(inputB[num_elements-1], _mm_min_epu8(shiftedleft, shiftedright)), _mm_min_epu8(inputA[num_elements-1], inputC[num_elements-1]));
}

void dilate_line_sse_1(unsigned w, const std::vector<__m128i> &inputA, const std::vector<__m128i> &inputB, const std::vector<__m128i> &inputC, std::vector<__m128i> &output)
{
	// Each vector element now contains 16 unsigned packed ints
	unsigned num_elements = w/16;

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
	shiftedleft_ptr[15] = 0; // Nothing to bring in so assign minmium possible value
	// Get maximum
	output[num_elements-1] = _mm_max_epu8(_mm_max_epu8(inputB[num_elements-1], _mm_max_epu8(shiftedleft, shiftedright)), _mm_max_epu8(inputA[num_elements-1], inputC[num_elements-1]));
}

void erode_line_top_sse_1(unsigned w, const std::vector<__m128i> &inputB, const std::vector<__m128i> &inputC, std::vector<__m128i> &output)
{
	// Each vector element now contains 16 unsigned packed ints
	unsigned num_elements = w/16;

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
	shiftedright_ptr[0] = 1; // Nothing to bring in so assign maxmium possible value
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
	shiftedleft_ptr[15] = 1; // Nothing to bring in so assign maxmium possible value
	// Get minimum
	output[num_elements-1] = _mm_min_epu8(_mm_min_epu8(shiftedleft, shiftedright), _mm_min_epu8(inputB[num_elements-1], inputC[num_elements-1]));
}

void dilate_line_top_sse_1(unsigned w, const std::vector<__m128i> &inputB, const std::vector<__m128i> &inputC, std::vector<__m128i> &output)
{
	// Each vector element now contains 16 unsigned packed ints
	unsigned num_elements = w/16;

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
	shiftedleft_ptr[15] = 0; // Nothing to bring in so assign minmium possible value
	// Get maximum
	output[num_elements-1] = _mm_max_epu8(_mm_max_epu8(shiftedleft, shiftedright), _mm_max_epu8(inputB[num_elements-1], inputC[num_elements-1]));
}

int process_recursive_function_sse_1(unsigned recursionlevel,const unsigned w,const unsigned h,const int inputhandle,const int outputhandle, std::vector<std::vector<__m128i>> &pixelsA, std::vector<std::vector<__m128i>> &pixelsB, std::vector<std::vector<__m128i>> &pixelsC, std::vector<uint32_t> &line, std::vector<__m128i> &output, std::vector<uint32_t> &status, const uint32_t toplevel){

	// Minimum width is 64
	// Bits is 1

	// Get pointers for this level to inrease readability
	uint32_t *thisline = &line[recursionlevel];

	// Pointers for circular addressing
	__m128i *pixelptr[3] = {&pixelsA[recursionlevel][0],&pixelsB[recursionlevel][0],&pixelsC[recursionlevel][0]};
	std::vector<__m128i> *pixelptr_vec[3] = {&pixelsA[recursionlevel],&pixelsB[recursionlevel],&pixelsC[recursionlevel]};

	///////////////////////// Try to read first line into A /////////////////////////////
	if (*thisline == 0) {

		if (recursionlevel == 0){
			if( readandunpack_sse_1 (w ,inputhandle , pixelptr[0]) != 0 )
				return 5;	// No image
		}
		else{
			if(process_recursive_function_sse_1(recursionlevel-1, w, h, inputhandle, outputhandle, pixelsA, pixelsB, pixelsC, line, *pixelptr_vec[0], status, toplevel))
				return 5;
		}

		*thisline = *thisline + 1;
	}

	///////////////////////// Read second line into B and unpack ////////////////////////////
	if (*thisline == 1) {

		if (recursionlevel == 0){
			readandunpack_sse_1 (w ,inputhandle , pixelptr[1]);
		}
		else{
			if(process_recursive_function_sse_1(recursionlevel-1, w, h, inputhandle, outputhandle, pixelsA, pixelsB, pixelsC, line, *pixelptr_vec[1], status, toplevel))
				return 5;
		}

		// Process first line
		if( status[recursionlevel] == 1){
			dilate_line_top_sse_1(w, *pixelptr_vec[0], *pixelptr_vec[1], output);
		}
		else if( status[recursionlevel] == 0){
			erode_line_top_sse_1(w, *pixelptr_vec[0], *pixelptr_vec[1], output);
		}

		if (recursionlevel == toplevel){
			// Pack and write first line
			packandwriteline_sse_1(w, &output[0],outputhandle);
		}

		*thisline = *thisline + 1;

		if (recursionlevel != toplevel){
			return 0;
		}
	}
	//////////////////////////// Steady State Section /////////////////////////////

	while(*thisline < h) {

		if (recursionlevel == 0){
			readandunpack_sse_1 (w ,inputhandle , pixelptr[*thisline%3]);
		}
		else{
			if(process_recursive_function_sse_1(recursionlevel-1, w, h, inputhandle, outputhandle, pixelsA, pixelsB, pixelsC, line, *pixelptr_vec[*thisline%3], status, toplevel))
				return 5;
		}

		if( status[recursionlevel] == 1 ){
			dilate_line_sse_1(w, *pixelptr_vec[(*thisline+1)%3], *pixelptr_vec[(*thisline+2)%3], *pixelptr_vec[*thisline%3], output);
		}
		else if( status[recursionlevel] == 0 ){
			erode_line_sse_1(w, *pixelptr_vec[(*thisline+1)%3], *pixelptr_vec[(*thisline+2)%3], *pixelptr_vec[*thisline%3], output);
		}

		if (recursionlevel == toplevel){
			packandwriteline_sse_1(w, &output[0],outputhandle);
		}

		*thisline = *thisline + 1;

		if (recursionlevel != toplevel){
			return 0;
		}

	}

	/////////////////////////// Last row of pixels /////////////////////////

	if( status[recursionlevel] == 1){
		dilate_line_top_sse_1(w, *pixelptr_vec[(*thisline+2)%3], *pixelptr_vec[(*thisline+1)%3], output);
	}
	else if( status[recursionlevel] == 0 ){
		erode_line_top_sse_1(w, *pixelptr_vec[(*thisline+2)%3], *pixelptr_vec[(*thisline+1)%3], output);
	}

	if (recursionlevel == toplevel){
		packandwriteline_sse_1(w, &output[0],outputhandle);
	}

	return 0;

}

void process_recursive_sse_1(const int levels, const unsigned w, const unsigned h,const int inputhandle,const int outputhandle){

	// Each vector element now contains 16 unsigned packed ints
	unsigned num_elements = w/16;

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
	while(process_recursive_function_sse_1(reclevel, w, h, inputhandle, outputhandle, pixelsA, pixelsB, pixelsC, line, outbuff, status, reclevel) == 0)
	{
		// Reset input status
		for(int i = 0; i < 2*std::abs(levels); i++){
			line[i] = 0;
		};
	}

	return;

}

//////////////////////////// 16-Bit ////////////////////////////////////////

__m128i min16(__m128i a, __m128i b)
{
	__m128i altzero, bltzero, xor, signedmin, signedmax, resultsame, resultdiff;

	altzero = _mm_cmplt_epi16(a, _mm_set1_epi16(0));
	bltzero = _mm_cmplt_epi16(b, _mm_set1_epi16(0));
	xor = _mm_xor_si128(altzero, bltzero);
	signedmin = _mm_min_epi16(a, b);
	signedmax = _mm_max_epi16(a, b);
	resultsame = _mm_andnot_si128(xor, signedmin);
	resultdiff = _mm_and_si128(xor, signedmax);
	return _mm_or_si128(resultsame, resultdiff);
}

__m128i max16(__m128i a, __m128i b)
{
	__m128i altzero, bltzero, xor, signedmin, signedmax, resultsame, resultdiff;

	altzero = _mm_cmplt_epi16(a, _mm_set1_epi16(0));
	bltzero = _mm_cmplt_epi16(b, _mm_set1_epi16(0));
	xor = _mm_xor_si128(altzero, bltzero);
	signedmin = _mm_min_epi16(a, b);
	signedmax = _mm_max_epi16(a, b);
	resultsame = _mm_andnot_si128(xor, signedmax);
	resultdiff = _mm_and_si128(xor, signedmin);
	return _mm_or_si128(resultsame, resultdiff);
}

void packandwriteline_sse_16(unsigned w, __m128i *input, int fd)
{
	// Minimum width is 4

	// Size of one line
	uint64_t cbLine=uint64_t(2*w);

	// Raw buffer for conversion
	std::vector<uint64_t> raw(cbLine/8);

	const uint64_t MASK=0x000000000000FFFFULL;

	uint64_t done=0;

	for(unsigned i=0;i<w/8;i++){

		uint16_t *input_ptr = (uint16_t*) &input[i];

		raw[2*i]=raw[2*i] | (uint64_t(input_ptr[0]&MASK)<< 0);
		raw[2*i]=raw[2*i] | (uint64_t(input_ptr[1]&MASK)<< 16);
		raw[2*i]=raw[2*i] | (uint64_t(input_ptr[2]&MASK)<< 32);
		raw[2*i]=raw[2*i] | (uint64_t(input_ptr[3]&MASK)<< 48);

		raw[2*i+1]=raw[2*i+1] | (uint64_t(input_ptr[4]&MASK)<< 0);
		raw[2*i+1]=raw[2*i+1] | (uint64_t(input_ptr[5]&MASK)<< 16);
		raw[2*i+1]=raw[2*i+1] | (uint64_t(input_ptr[6]&MASK)<< 32);
		raw[2*i+1]=raw[2*i+1] | (uint64_t(input_ptr[7]&MASK)<< 48);
	}

	if(w%8 == 4) // This is a possible case by the specifications
	{
		unsigned index = w/8;
		unsigned index_raw = 2*index; // We need to use index so it rounds down 

		uint16_t *input_ptr = (uint16_t*) &input[index];

		raw[index_raw]=raw[index_raw] | (uint64_t(input_ptr[0]&MASK)<< 0);
		raw[index_raw]=raw[index_raw] | (uint64_t(input_ptr[1]&MASK)<< 16);
		raw[index_raw]=raw[index_raw] | (uint64_t(input_ptr[2]&MASK)<< 32);
		raw[index_raw]=raw[index_raw] | (uint64_t(input_ptr[3]&MASK)<< 48);
	}

	while(done<cbLine){
		int todo=(int)std::min(uint64_t(1)<<30, cbLine-done);

		int got=write(fd, &raw[0]+done, todo);
		if(got<=0)
			throw std::invalid_argument("Write failure.");
		done+=got;
	}
}

int readandunpack_sse_16(unsigned w, int fd, __m128i *output)
{
	// Size of one line
	uint64_t cbLine=uint64_t(2*w);

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

	for(unsigned i=0;i<w/8;i++){

		output[i] = _mm_setr_epi16(
			raw[2*i] & 0x000000000000FFFFULL,
			(raw[2*i] & 0x00000000FFFF0000ULL) >> 16,
			(raw[2*i] & 0x0000FFFF00000000ULL) >> 32,
			(raw[2*i] & 0xFFFF000000000000ULL) >> 48,
			raw[2*i+1] & 0x000000000000FFFFULL,
			(raw[2*i+1] & 0x00000000FFFF0000ULL) >> 16,
			(raw[2*i+1] & 0x0000FFFF00000000ULL) >> 32,
			(raw[2*i+1] & 0xFFFF000000000000ULL) >> 48
			);
	}

	if(w%8 == 4) // This is a possible case by the specifications
	{
		unsigned index = w/8;
		unsigned index_raw = 2*index; // We need to use index so it rounds down 

		output[index] = _mm_setr_epi16(
			raw[index_raw] & 0x000000000000FFFFULL,
			(raw[index_raw] & 0x00000000FFFF0000ULL) >> 16,
			(raw[index_raw] & 0x0000FFFF00000000ULL) >> 32,
			(raw[index_raw] & 0xFFFF000000000000ULL) >> 48,
			0,0,0,0
			);
	}

	return 0;
}

void erode_line_sse_16(unsigned w, const std::vector<__m128i> &inputA, const std::vector<__m128i> &inputB, const std::vector<__m128i> &inputC, std::vector<__m128i> &output)
{
	// Each vector element now contains 8 unsigned packed ints
	unsigned num_elements = w/8;
	if(w%8 == 4) num_elements++;

	//Temporary buffers
	__m128i shiftedright, shiftedleft;
	uint16_t *shiftedright_ptr = (uint16_t*) &shiftedright;
	uint16_t *shiftedleft_ptr = (uint16_t*) &shiftedleft;
	uint16_t *next_container, *previous_container;

	for (unsigned i = 1; i < num_elements - 1; i++)
	{
		previous_container = (uint16_t*) &inputB[i-1];
		next_container = (uint16_t*) &inputB[i+1];
		// Shift self so we can find minimum
		shiftedright = _mm_slli_si128(inputB[i], 2);
		shiftedleft = _mm_srli_si128(inputB[i], 2);
		// Need to bring in int from next element in vector
		shiftedright_ptr[0] = previous_container[7];
		shiftedleft_ptr[7] = next_container[0];
		// Get minimum
		output[i] = min16(min16(inputB[i], min16(shiftedleft, shiftedright)), min16(inputA[i], inputC[i]));
	}

	// When i = 0
	next_container = (uint16_t*) &inputB[1];
	// Shift self so we can find minimum
	shiftedright = _mm_slli_si128(inputB[0], 2);
	shiftedleft = _mm_srli_si128(inputB[0], 2);
	// Need to bring in int from next element in vector
	shiftedright_ptr[0] = 65535; // Nothing to bring in so assign maxmium possible value
	shiftedleft_ptr[7] = next_container[0];
	// Get minimum
	output[0] = min16(min16(inputB[0], min16(shiftedleft, shiftedright)), min16(inputA[0], inputC[0]));

	// When i = num_elements - 1
	previous_container = (uint16_t*) &inputB[num_elements-2];
	// Shift self so we can find minimum
	shiftedright = _mm_slli_si128(inputB[num_elements-1], 2);
	shiftedleft = _mm_srli_si128(inputB[num_elements-1], 2);
	// Need to bring in int from next element in vector
	shiftedright_ptr[0] = previous_container[7];
	shiftedleft_ptr[7-w%8] = 65535; // Accounts for w%8 == 4 case
	// Get minimum
	output[num_elements-1] = min16(min16(inputB[num_elements-1], min16(shiftedleft, shiftedright)), min16(inputA[num_elements-1], inputC[num_elements-1]));
}

void dilate_line_sse_16(unsigned w, const std::vector<__m128i> &inputA, const std::vector<__m128i> &inputB, const std::vector<__m128i> &inputC, std::vector<__m128i> &output)
{
	// Each vector element now contains 16 unsigned packed ints
	unsigned num_elements = w/8;
	if(w%8 == 4) num_elements++;

	//Temporary buffers
	__m128i shiftedright, shiftedleft;
	uint16_t *shiftedright_ptr = (uint16_t*) &shiftedright;
	uint16_t *shiftedleft_ptr = (uint16_t*) &shiftedleft;
	uint16_t *next_container, *previous_container;

	for (unsigned i = 1; i < num_elements - 1; i++)
	{
		previous_container = (uint16_t*) &inputB[i-1];
		next_container = (uint16_t*) &inputB[i+1];
		// Shift self so we can find maximum
		shiftedright = _mm_slli_si128(inputB[i], 2);
		shiftedleft = _mm_srli_si128(inputB[i], 2);
		// Need to bring in int from next element in vector
		shiftedright_ptr[0] = previous_container[7];
		shiftedleft_ptr[7] = next_container[0];
		// Get maximum
		output[i] = max16(max16(inputB[i], max16(shiftedleft, shiftedright)), max16(inputA[i], inputC[i]));
	}

	// When i = 0
	next_container = (uint16_t*) &inputB[1];
	// Shift self so we can find maximum
	shiftedright = _mm_slli_si128(inputB[0], 2);
	shiftedleft = _mm_srli_si128(inputB[0], 2);
	// Need to bring in int from next element in vector
	shiftedright_ptr[0] = 0; // Nothing to bring in so assign minmium possible value
	shiftedleft_ptr[7] = next_container[0];
	// Get maximum
	output[0] = max16(max16(inputB[0], max16(shiftedleft, shiftedright)), max16(inputA[0], inputC[0]));

	// When i = num_elements - 1
	previous_container = (uint16_t*) &inputB[num_elements-2];
	// Shift self so we can find maximum
	shiftedright = _mm_slli_si128(inputB[num_elements-1], 2);
	shiftedleft = _mm_srli_si128(inputB[num_elements-1], 2);
	// Need to bring in int from next element in vector
	shiftedright_ptr[0] = previous_container[7];
	shiftedleft_ptr[7-w%8] = 0; // Accounts for w%8 == 4 case
	// Get maximum
	output[num_elements-1] = max16(max16(inputB[num_elements-1], max16(shiftedleft, shiftedright)), max16(inputA[num_elements-1], inputC[num_elements-1]));
}

void erode_line_top_sse_16(unsigned w, const std::vector<__m128i> &inputB, const std::vector<__m128i> &inputC, std::vector<__m128i> &output)
{
	// Each vector element now contains 16 unsigned packed ints
	unsigned num_elements = w/8;
	if(w%8 == 4) num_elements++;

	//Temporary buffers
	__m128i shiftedright, shiftedleft;
	uint16_t *shiftedright_ptr = (uint16_t*) &shiftedright;
	uint16_t *shiftedleft_ptr = (uint16_t*) &shiftedleft;
	uint16_t *next_container, *previous_container;

	for (unsigned i = 1; i < num_elements - 1; i++)
	{
		previous_container = (uint16_t*) &inputB[i-1];
		next_container = (uint16_t*) &inputB[i+1];
		// Shift self so we can find minimum
		shiftedright = _mm_slli_si128(inputB[i], 2);
		shiftedleft = _mm_srli_si128(inputB[i], 2);
		// Need to bring in int from next element in vector
		shiftedright_ptr[0] = previous_container[7];
		shiftedleft_ptr[7] = next_container[0];
		// Get minimum
		output[i] = min16(min16(shiftedleft, shiftedright), min16(inputB[i], inputC[i]));
	}

	// When i = 0
	next_container = (uint16_t*) &inputB[1];
	// Shift self so we can find minimum
	shiftedright = _mm_slli_si128(inputB[0], 2);
	shiftedleft = _mm_srli_si128(inputB[0], 2);
	// Need to bring in int from next element in vector
	shiftedright_ptr[0] = 65535; // Nothing to bring in so assign maxmium possible value
	shiftedleft_ptr[7] = next_container[0];
	// Get minimum
	output[0] = min16(min16(shiftedleft, shiftedright), min16(inputB[0], inputC[0]));

	// When i = num_elements - 1
	previous_container = (uint16_t*) &inputB[num_elements-2];
	// Shift self so we can find minimum
	shiftedright = _mm_slli_si128(inputB[num_elements-1], 2);
	shiftedleft = _mm_srli_si128(inputB[num_elements-1], 2);
	// Need to bring in int from next element in vector
	shiftedright_ptr[0] = previous_container[7];
	shiftedleft_ptr[7-w%8] = 65535; // Accounts for w%8 == 4 case
	// Get minimum
	output[num_elements-1] = min16(min16(shiftedleft, shiftedright), min16(inputB[num_elements-1], inputC[num_elements-1]));
}

void dilate_line_top_sse_16(unsigned w, const std::vector<__m128i> &inputB, const std::vector<__m128i> &inputC, std::vector<__m128i> &output)
{
	// Each vector element now contains 16 unsigned packed ints
	unsigned num_elements = w/8;
	if(w%8 == 4) num_elements++;

	//Temporary buffers
	__m128i shiftedright, shiftedleft;
	uint16_t *shiftedright_ptr = (uint16_t*) &shiftedright;
	uint16_t *shiftedleft_ptr = (uint16_t*) &shiftedleft;
	uint16_t *next_container, *previous_container;

	for (unsigned i = 1; i < num_elements - 1; i++)
	{
		previous_container = (uint16_t*) &inputB[i-1];
		next_container = (uint16_t*) &inputB[i+1];
		// Shift self so we can find maximum
		shiftedright = _mm_slli_si128(inputB[i], 2);
		shiftedleft = _mm_srli_si128(inputB[i], 2);
		// Need to bring in int from next element in vector
		shiftedright_ptr[0] = previous_container[7];
		shiftedleft_ptr[7] = next_container[0];
		// Get maximum
		output[i] = max16(max16(shiftedleft, shiftedright), max16(inputB[i], inputC[i]));
	}

	// When i = 0
	next_container = (uint16_t*) &inputB[1];
	// Shift self so we can find maximum
	shiftedright = _mm_slli_si128(inputB[0], 2);
	shiftedleft = _mm_srli_si128(inputB[0], 2);
	// Need to bring in int from next element in vector
	shiftedright_ptr[0] = 0; // Nothing to bring in so assign minmium possible value
	shiftedleft_ptr[7] = next_container[0];
	// Get maximum
	output[0] = max16(max16(shiftedleft, shiftedright), max16(inputB[0], inputC[0]));

	// When i = num_elements - 1
	previous_container = (uint16_t*) &inputB[num_elements-2];
	// Shift self so we can find maximum
	shiftedright = _mm_slli_si128(inputB[num_elements-1], 2);
	shiftedleft = _mm_srli_si128(inputB[num_elements-1], 2);
	// Need to bring in int from next element in vector
	shiftedright_ptr[0] = previous_container[7];
	shiftedleft_ptr[7-w%8] = 0; // Accounts for w%8 == 4 case
	// Get maximum
	output[num_elements-1] = max16(max16(shiftedleft, shiftedright), max16(inputB[num_elements-1], inputC[num_elements-1]));
}

int process_recursive_function_sse_16(unsigned recursionlevel,const unsigned w,const unsigned h,const int inputhandle,const int outputhandle, std::vector<std::vector<__m128i>> &pixelsA, std::vector<std::vector<__m128i>> &pixelsB, std::vector<std::vector<__m128i>> &pixelsC, std::vector<uint32_t> &line, std::vector<__m128i> &output, std::vector<uint32_t> &status, const uint32_t toplevel){

	// Minimum width is 4
	// Bits is 16

	// Get pointers for this level to inrease readability
	uint32_t *thisline = &line[recursionlevel];

	// Pointers for circular addressing
	__m128i *pixelptr[3] = {&pixelsA[recursionlevel][0],&pixelsB[recursionlevel][0],&pixelsC[recursionlevel][0]};
	std::vector<__m128i> *pixelptr_vec[3] = {&pixelsA[recursionlevel],&pixelsB[recursionlevel],&pixelsC[recursionlevel]};

	///////////////////////// Try to read first line into A /////////////////////////////
	if (*thisline == 0) {

		if (recursionlevel == 0){
			if( readandunpack_sse_16 (w ,inputhandle , pixelptr[0]) != 0 )
				return 5;	// No image
		}
		else{
			if(process_recursive_function_sse_16(recursionlevel-1, w, h, inputhandle, outputhandle, pixelsA, pixelsB, pixelsC, line, *pixelptr_vec[0], status, toplevel))
				return 5;
		}

		*thisline = *thisline + 1;
	}

	///////////////////////// Read second line into B and unpack ////////////////////////////
	if (*thisline == 1) {

		if (recursionlevel == 0){
			readandunpack_sse_16 (w ,inputhandle , pixelptr[1]);
		}
		else{
			if(process_recursive_function_sse_16(recursionlevel-1, w, h, inputhandle, outputhandle, pixelsA, pixelsB, pixelsC, line, *pixelptr_vec[1], status, toplevel))
				return 5;
		}

		// Process first line
		if( status[recursionlevel] == 1){
			dilate_line_top_sse_16(w, *pixelptr_vec[0], *pixelptr_vec[1], output);
		}
		else if( status[recursionlevel] == 0){
			erode_line_top_sse_16(w, *pixelptr_vec[0], *pixelptr_vec[1], output);
		}

		if (recursionlevel == toplevel){
			// Pack and write first line
			packandwriteline_sse_16(w, &output[0],outputhandle);
		}

		*thisline = *thisline + 1;

		if (recursionlevel != toplevel){
			return 0;
		}
	}
	//////////////////////////// Steady State Section /////////////////////////////

	while(*thisline < h) {

		if (recursionlevel == 0){
			readandunpack_sse_16 (w ,inputhandle , pixelptr[*thisline%3]);
		}
		else{
			if(process_recursive_function_sse_16(recursionlevel-1, w, h, inputhandle, outputhandle, pixelsA, pixelsB, pixelsC, line, *pixelptr_vec[*thisline%3], status, toplevel))
				return 5;
		}

		if( status[recursionlevel] == 1 ){
			dilate_line_sse_16(w, *pixelptr_vec[(*thisline+1)%3], *pixelptr_vec[(*thisline+2)%3], *pixelptr_vec[*thisline%3], output);
		}
		else if( status[recursionlevel] == 0 ){
			erode_line_sse_16(w, *pixelptr_vec[(*thisline+1)%3], *pixelptr_vec[(*thisline+2)%3], *pixelptr_vec[*thisline%3], output);
		}

		if (recursionlevel == toplevel){
			packandwriteline_sse_16(w, &output[0],outputhandle);
		}

		*thisline = *thisline + 1;

		if (recursionlevel != toplevel){
			return 0;
		}

	}

	/////////////////////////// Last row of pixels /////////////////////////

	if( status[recursionlevel] == 1){
		dilate_line_top_sse_16(w, *pixelptr_vec[(*thisline+2)%3], *pixelptr_vec[(*thisline+1)%3], output);
	}
	else if( status[recursionlevel] == 0 ){
		erode_line_top_sse_16(w, *pixelptr_vec[(*thisline+2)%3], *pixelptr_vec[(*thisline+1)%3], output);
	}

	if (recursionlevel == toplevel){
		packandwriteline_sse_16(w, &output[0],outputhandle);
	}

	return 0;

}

void process_recursive_sse_16(const int levels, const unsigned w, const unsigned h,const int inputhandle,const int outputhandle){

	// Each vector element now contains 8 unsigned packed ints
	unsigned num_elements = w/8;
	if(w%8 == 4) num_elements++;

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
	while(process_recursive_function_sse_16(reclevel, w, h, inputhandle, outputhandle, pixelsA, pixelsB, pixelsC, line, outbuff, status, reclevel) == 0)
	{
		// Reset input status
		for(int i = 0; i < 2*std::abs(levels); i++){
			line[i] = 0;
		};
	}

	return;

}

//////////////////////////// 32-Bit ////////////////////////////////////////

__m128i min32(__m128i a, __m128i b)
{
	__m128i altzero, bltzero, agtbsigned, agtbunsignedpt1, agtbunsignedpt2, xor, agtbunsigned, resulta, resultb;

	altzero = _mm_cmplt_epi32(a, _mm_set1_epi32(0));
	bltzero = _mm_cmplt_epi32(b, _mm_set1_epi32(0));
	xor = _mm_xor_si128(altzero, bltzero);
	agtbsigned = _mm_cmpgt_epi32(a, b);
	agtbunsignedpt1 = _mm_andnot_si128(xor, agtbsigned);
	agtbunsignedpt2 = _mm_andnot_si128(agtbsigned, xor);
	agtbunsigned = _mm_or_si128(agtbunsignedpt1, agtbunsignedpt2);
	resulta = _mm_andnot_si128(agtbunsigned, a);
	resultb = _mm_and_si128(agtbunsigned, b);
	return _mm_or_si128(resulta, resultb);
}

__m128i max32(__m128i a, __m128i b)
{
	__m128i altzero, bltzero, agtbsigned, agtbunsignedpt1, agtbunsignedpt2, xor, agtbunsigned, resulta, resultb;

	altzero = _mm_cmplt_epi32(a, _mm_set1_epi32(0));
	bltzero = _mm_cmplt_epi32(b, _mm_set1_epi32(0));
	xor = _mm_xor_si128(altzero, bltzero);
	agtbsigned = _mm_cmpgt_epi32(a, b);
	agtbunsignedpt1 = _mm_andnot_si128(xor, agtbsigned);
	agtbunsignedpt2 = _mm_andnot_si128(agtbsigned, xor);
	agtbunsigned = _mm_or_si128(agtbunsignedpt1, agtbunsignedpt2);
	resulta = _mm_and_si128(agtbunsigned, a);
	resultb = _mm_andnot_si128(agtbunsigned, b);
	return _mm_or_si128(resulta, resultb);
}

void packandwriteline_sse_32(unsigned w, __m128i *input, int fd)
{
	// Minimum width is 2

	// Size of one line
	uint64_t cbLine=uint64_t(4*w);

	// Raw buffer for conversion
	std::vector<uint64_t> raw(cbLine/8);

	const uint64_t MASK=0x00000000FFFFFFFFULL;

	uint64_t done=0;

	for(unsigned i=0;i<w/4;i++){

		uint32_t *input_ptr = (uint32_t*) &input[i];

		raw[2*i]=raw[2*i] | (uint64_t(input_ptr[0]&MASK)<< 0);
		raw[2*i]=raw[2*i] | (uint64_t(input_ptr[1]&MASK)<< 32);

		raw[2*i+1]=raw[2*i+1] | (uint64_t(input_ptr[2]&MASK)<< 0);
		raw[2*i+1]=raw[2*i+1] | (uint64_t(input_ptr[3]&MASK)<< 32);
	}

	if(w%4 == 2) // This is a possible case by the specifications
	{
		unsigned index = w/4;
		unsigned index_raw = 2*index; // We need to use index so it rounds down 

		uint32_t *input_ptr = (uint32_t*) &input[index];

		raw[index_raw]=raw[index_raw] | (uint64_t(input_ptr[0]&MASK)<< 0);
		raw[index_raw]=raw[index_raw] | (uint64_t(input_ptr[1]&MASK)<< 32);
	}

	while(done<cbLine){
		int todo=(int)std::min(uint64_t(1)<<30, cbLine-done);

		int got=write(fd, &raw[0]+done, todo);
		if(got<=0)
			throw std::invalid_argument("Write failure.");
		done+=got;
	}
}

int readandunpack_sse_32(unsigned w, int fd, __m128i *output)
{
	// Size of one line
	uint64_t cbLine=uint64_t(4*w);

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

	for(unsigned i=0;i<w/4;i++){

		output[i] = _mm_setr_epi32(
			 raw[2*i]   & 0x00000000FFFFFFFFULL,
			(raw[2*i]   & 0xFFFFFFFF00000000ULL) >> 32,
			 raw[2*i+1] & 0x00000000FFFFFFFFULL,
			(raw[2*i+1] & 0xFFFFFFFF00000000ULL) >> 32
			);
	}

	if(w%4 == 2) // This is a possible case by the specifications
	{
		unsigned index = w/4;
		unsigned index_raw = 2*index; // We need to use index so it rounds down 

		output[index] = _mm_setr_epi32(
			 raw[index_raw]   & 0x00000000FFFFFFFFULL,
			(raw[index_raw]   & 0xFFFFFFFF00000000ULL) >> 32,
			0,0
			);
	}

	return 0;
}

void erode_line_sse_32(unsigned w, const std::vector<__m128i> &inputA, const std::vector<__m128i> &inputB, const std::vector<__m128i> &inputC, std::vector<__m128i> &output)
{
	// Each vector element now contains 8 unsigned packed ints
	unsigned num_elements = w/4;
	if(w%4 == 2) num_elements++;

	//Temporary buffers
	__m128i shiftedright, shiftedleft;
	uint32_t *shiftedright_ptr = (uint32_t*) &shiftedright;
	uint32_t *shiftedleft_ptr = (uint32_t*) &shiftedleft;
	uint32_t *next_container, *previous_container;

	for (unsigned i = 1; i < num_elements - 1; i++)
	{
		previous_container = (uint32_t*) &inputB[i-1];
		next_container = (uint32_t*) &inputB[i+1];
		// Shift self so we can find minimum
		shiftedright = _mm_slli_si128(inputB[i], 4);
		shiftedleft = _mm_srli_si128(inputB[i], 4);
		// Need to bring in int from next element in vector
		shiftedright_ptr[0] = previous_container[3];
		shiftedleft_ptr[3] = next_container[0];
		// Get minimum
		output[i] = min32(min32(inputB[i], min32(shiftedleft, shiftedright)), min32(inputA[i], inputC[i]));
	}

	// When i = 0
	next_container = (uint32_t*) &inputB[1];
	// Shift self so we can find minimum
	shiftedright = _mm_slli_si128(inputB[0], 4);
	shiftedleft = _mm_srli_si128(inputB[0], 4);
	// Need to bring in int from next element in vector
	shiftedright_ptr[0] = 4294967295; // Nothing to bring in so assign maxmium possible value
	shiftedleft_ptr[3] = next_container[0];
	// Get minimum
	output[0] = min32(min32(inputB[0], min32(shiftedleft, shiftedright)), min32(inputA[0], inputC[0]));

	// When i = num_elements - 1
	previous_container = (uint32_t*) &inputB[num_elements-2];
	// Shift self so we can find minimum
	shiftedright = _mm_slli_si128(inputB[num_elements-1], 4);
	shiftedleft = _mm_srli_si128(inputB[num_elements-1], 4);
	// Need to bring in int from next element in vector
	shiftedright_ptr[0] = previous_container[3];
	shiftedleft_ptr[3-w%4] = 4294967295; // Accounts for w%4 == 2 case
	// Get minimum
	output[num_elements-1] = min32(min32(inputB[num_elements-1], min32(shiftedleft, shiftedright)), min32(inputA[num_elements-1], inputC[num_elements-1]));
}

void dilate_line_sse_32(unsigned w, const std::vector<__m128i> &inputA, const std::vector<__m128i> &inputB, const std::vector<__m128i> &inputC, std::vector<__m128i> &output)
{
	// Each vector element now contains 16 unsigned packed ints
	unsigned num_elements = w/4;
	if(w%4 == 2) num_elements++;

	//Temporary buffers
	__m128i shiftedright, shiftedleft;
	uint32_t *shiftedright_ptr = (uint32_t*) &shiftedright;
	uint32_t *shiftedleft_ptr = (uint32_t*) &shiftedleft;
	uint32_t *next_container, *previous_container;

	for (unsigned i = 1; i < num_elements - 1; i++)
	{
		previous_container = (uint32_t*) &inputB[i-1];
		next_container = (uint32_t*) &inputB[i+1];
		// Shift self so we can find maximum
		shiftedright = _mm_slli_si128(inputB[i], 4);
		shiftedleft = _mm_srli_si128(inputB[i], 4);
		// Need to bring in int from next element in vector
		shiftedright_ptr[0] = previous_container[3];
		shiftedleft_ptr[3] = next_container[0];
		// Get maximum
		output[i] = max32(max32(inputB[i], max32(shiftedleft, shiftedright)), max32(inputA[i], inputC[i]));
	}

	// When i = 0
	next_container = (uint32_t*) &inputB[1];
	// Shift self so we can find maximum
	shiftedright = _mm_slli_si128(inputB[0], 4);
	shiftedleft = _mm_srli_si128(inputB[0], 4);
	// Need to bring in int from next element in vector
	shiftedright_ptr[0] = 0; // Nothing to bring in so assign minmium possible value
	shiftedleft_ptr[3] = next_container[0];
	// Get maximum
	output[0] = max32(max32(inputB[0], max32(shiftedleft, shiftedright)), max32(inputA[0], inputC[0]));

	// When i = num_elements - 1
	previous_container = (uint32_t*) &inputB[num_elements-2];
	// Shift self so we can find maximum
	shiftedright = _mm_slli_si128(inputB[num_elements-1], 4);
	shiftedleft = _mm_srli_si128(inputB[num_elements-1], 4);
	// Need to bring in int from next element in vector
	shiftedright_ptr[0] = previous_container[3];
	shiftedleft_ptr[3-w%4] = 0; // Accounts for w%4 == 2 case
	// Get maximum
	output[num_elements-1] = max32(max32(inputB[num_elements-1], max32(shiftedleft, shiftedright)), max32(inputA[num_elements-1], inputC[num_elements-1]));
}

void erode_line_top_sse_32(unsigned w, const std::vector<__m128i> &inputB, const std::vector<__m128i> &inputC, std::vector<__m128i> &output)
{
	// Each vector element now contains 16 unsigned packed ints
	unsigned num_elements = w/4;
	if(w%4 == 2) num_elements++;

	//Temporary buffers
	__m128i shiftedright, shiftedleft;
	uint32_t *shiftedright_ptr = (uint32_t*) &shiftedright;
	uint32_t *shiftedleft_ptr = (uint32_t*) &shiftedleft;
	uint32_t *next_container, *previous_container;

	for (unsigned i = 1; i < num_elements - 1; i++)
	{
		previous_container = (uint32_t*) &inputB[i-1];
		next_container = (uint32_t*) &inputB[i+1];
		// Shift self so we can find minimum
		shiftedright = _mm_slli_si128(inputB[i], 4);
		shiftedleft = _mm_srli_si128(inputB[i], 4);
		// Need to bring in int from next element in vector
		shiftedright_ptr[0] = previous_container[3];
		shiftedleft_ptr[3] = next_container[0];
		// Get minimum
		output[i] = min32(min32(shiftedleft, shiftedright), min32(inputB[i], inputC[i]));
	}

	// When i = 0
	next_container = (uint32_t*) &inputB[1];
	// Shift self so we can find minimum
	shiftedright = _mm_slli_si128(inputB[0], 4);
	shiftedleft = _mm_srli_si128(inputB[0], 4);
	// Need to bring in int from next element in vector
	shiftedright_ptr[0] = 4294967295; // Nothing to bring in so assign maxmium possible value
	shiftedleft_ptr[3] = next_container[0];
	// Get minimum
	output[0] = min32(min32(shiftedleft, shiftedright), min32(inputB[0], inputC[0]));

	// When i = num_elements - 1
	previous_container = (uint32_t*) &inputB[num_elements-2];
	// Shift self so we can find minimum
	shiftedright = _mm_slli_si128(inputB[num_elements-1], 4);
	shiftedleft = _mm_srli_si128(inputB[num_elements-1], 4);
	// Need to bring in int from next element in vector
	shiftedright_ptr[0] = previous_container[3];
	shiftedleft_ptr[3-w%4] = 4294967295; // Accounts for w%4 == 2 case
	// Get minimum
	output[num_elements-1] = min32(min32(shiftedleft, shiftedright), min32(inputB[num_elements-1], inputC[num_elements-1]));
}

void dilate_line_top_sse_32(unsigned w, const std::vector<__m128i> &inputB, const std::vector<__m128i> &inputC, std::vector<__m128i> &output)
{
	// Each vector element now contains 16 unsigned packed ints
	unsigned num_elements = w/4;
	if(w%4 == 2) num_elements++;

	//Temporary buffers
	__m128i shiftedright, shiftedleft;
	uint32_t *shiftedright_ptr = (uint32_t*) &shiftedright;
	uint32_t *shiftedleft_ptr = (uint32_t*) &shiftedleft;
	uint32_t *next_container, *previous_container;

	for (unsigned i = 1; i < num_elements - 1; i++)
	{
		previous_container = (uint32_t*) &inputB[i-1];
		next_container = (uint32_t*) &inputB[i+1];
		// Shift self so we can find maximum
		shiftedright = _mm_slli_si128(inputB[i], 4);
		shiftedleft = _mm_srli_si128(inputB[i], 4);
		// Need to bring in int from next element in vector
		shiftedright_ptr[0] = previous_container[3];
		shiftedleft_ptr[3] = next_container[0];
		// Get maximum
		output[i] = max32(max32(shiftedleft, shiftedright), max32(inputB[i], inputC[i]));
	}

	// When i = 0
	next_container = (uint32_t*) &inputB[1];
	// Shift self so we can find maximum
	shiftedright = _mm_slli_si128(inputB[0], 4);
	shiftedleft = _mm_srli_si128(inputB[0], 4);
	// Need to bring in int from next element in vector
	shiftedright_ptr[0] = 0; // Nothing to bring in so assign minmium possible value
	shiftedleft_ptr[3] = next_container[0];
	// Get maximum
	output[0] = max32(max32(shiftedleft, shiftedright), max32(inputB[0], inputC[0]));

	// When i = num_elements - 1
	previous_container = (uint32_t*) &inputB[num_elements-2];
	// Shift self so we can find maximum
	shiftedright = _mm_slli_si128(inputB[num_elements-1], 4);
	shiftedleft = _mm_srli_si128(inputB[num_elements-1], 4);
	// Need to bring in int from next element in vector
	shiftedright_ptr[0] = previous_container[3];
	shiftedleft_ptr[3-w%4] = 0; // Accounts for w%4 == 2 case
	// Get maximum
	output[num_elements-1] = max32(max32(shiftedleft, shiftedright), max32(inputB[num_elements-1], inputC[num_elements-1]));
}

int process_recursive_function_sse_32(unsigned recursionlevel,const unsigned w,const unsigned h,const int inputhandle,const int outputhandle, std::vector<std::vector<__m128i>> &pixelsA, std::vector<std::vector<__m128i>> &pixelsB, std::vector<std::vector<__m128i>> &pixelsC, std::vector<uint32_t> &line, std::vector<__m128i> &output, std::vector<uint32_t> &status, const uint32_t toplevel){

	// Minimum width is 2
	// Bits is 32

	// Get pointers for this level to inrease readability
	uint32_t *thisline = &line[recursionlevel];

	// Pointers for circular addressing
	__m128i *pixelptr[3] = {&pixelsA[recursionlevel][0],&pixelsB[recursionlevel][0],&pixelsC[recursionlevel][0]};
	std::vector<__m128i> *pixelptr_vec[3] = {&pixelsA[recursionlevel],&pixelsB[recursionlevel],&pixelsC[recursionlevel]};

	///////////////////////// Try to read first line into A /////////////////////////////
	if (*thisline == 0) {

		if (recursionlevel == 0){
			if( readandunpack_sse_32 (w ,inputhandle , pixelptr[0]) != 0 )
				return 5;	// No image
		}
		else{
			if(process_recursive_function_sse_32(recursionlevel-1, w, h, inputhandle, outputhandle, pixelsA, pixelsB, pixelsC, line, *pixelptr_vec[0], status, toplevel))
				return 5;
		}

		*thisline = *thisline + 1;
	}

	///////////////////////// Read second line into B and unpack ////////////////////////////
	if (*thisline == 1) {

		if (recursionlevel == 0){
			readandunpack_sse_32 (w ,inputhandle , pixelptr[1]);
		}
		else{
			if(process_recursive_function_sse_32(recursionlevel-1, w, h, inputhandle, outputhandle, pixelsA, pixelsB, pixelsC, line, *pixelptr_vec[1], status, toplevel))
				return 5;
		}

		// Process first line
		if( status[recursionlevel] == 1){
			dilate_line_top_sse_32(w, *pixelptr_vec[0], *pixelptr_vec[1], output);
		}
		else if( status[recursionlevel] == 0){
			erode_line_top_sse_32(w, *pixelptr_vec[0], *pixelptr_vec[1], output);
		}

		if (recursionlevel == toplevel){
			// Pack and write first line
			packandwriteline_sse_32(w, &output[0],outputhandle);
		}

		*thisline = *thisline + 1;

		if (recursionlevel != toplevel){
			return 0;
		}
	}
	//////////////////////////// Steady State Section /////////////////////////////

	while(*thisline < h) {

		if (recursionlevel == 0){
			readandunpack_sse_32 (w ,inputhandle , pixelptr[*thisline%3]);
		}
		else{
			if(process_recursive_function_sse_32(recursionlevel-1, w, h, inputhandle, outputhandle, pixelsA, pixelsB, pixelsC, line, *pixelptr_vec[*thisline%3], status, toplevel))
				return 5;
		}

		if( status[recursionlevel] == 1 ){
			dilate_line_sse_32(w, *pixelptr_vec[(*thisline+1)%3], *pixelptr_vec[(*thisline+2)%3], *pixelptr_vec[*thisline%3], output);
		}
		else if( status[recursionlevel] == 0 ){
			erode_line_sse_32(w, *pixelptr_vec[(*thisline+1)%3], *pixelptr_vec[(*thisline+2)%3], *pixelptr_vec[*thisline%3], output);
		}

		if (recursionlevel == toplevel){
			packandwriteline_sse_32(w, &output[0],outputhandle);
		}

		*thisline = *thisline + 1;

		if (recursionlevel != toplevel){
			return 0;
		}

	}

	/////////////////////////// Last row of pixels /////////////////////////

	if( status[recursionlevel] == 1){
		dilate_line_top_sse_32(w, *pixelptr_vec[(*thisline+2)%3], *pixelptr_vec[(*thisline+1)%3], output);
	}
	else if( status[recursionlevel] == 0 ){
		erode_line_top_sse_32(w, *pixelptr_vec[(*thisline+2)%3], *pixelptr_vec[(*thisline+1)%3], output);
	}

	if (recursionlevel == toplevel){
		packandwriteline_sse_32(w, &output[0],outputhandle);
	}

	return 0;

}

void process_recursive_sse_32(const int levels, const unsigned w, const unsigned h,const int inputhandle,const int outputhandle){

	// Each vector element now contains 4 unsigned packed ints
	unsigned num_elements = w/4;
	if(w%4 == 2) num_elements++;

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
	while(process_recursive_function_sse_32(reclevel, w, h, inputhandle, outputhandle, pixelsA, pixelsB, pixelsC, line, outbuff, status, reclevel) == 0)
	{
		// Reset input status
		for(int i = 0; i < 2*std::abs(levels); i++){
			line[i] = 0;
		};
	}

	return;

}