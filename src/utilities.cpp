// Header files for windows compilation
#ifdef _WIN32
#include <string>
#endif

// Shared Headers
#include <cassert>
#include <stdexcept>
#include <fstream>
#include <streambuf>

#include "utilities.h"

//#include "tmmintrin.h"

// SSE3
#include "pmmintrin.h"

//TODO: Move OpenCL initialization code into here.

std::string LoadSource(const char *fileName)
{
	std::string baseDir="src/kernels";
	if(getenv("HPCE_CL_SRC_DIR")){
		baseDir=getenv("HPCE_CL_SRC_DIR");
	}

	std::string fullName=baseDir+"/"+fileName;

	std::ifstream src(fullName.c_str(), std::ios::in | std::ios::binary);
	if(!src.is_open())
		throw std::runtime_error("LoadSource : Couldn't load cl file from '"+fullName+"'.");

	return std::string((std::istreambuf_iterator<char>(src)),std::istreambuf_iterator<char>());
}

uint64_t shuffle64(unsigned bits, uint64_t x)
{
	if(bits==1){
		x=((x&0x0101010101010101ull)<<7)
			| ((x&0x0202020202020202ull)<<5)
			| ((x&0x0404040404040404ull)<<3)
			| ((x&0x0808080808080808ull)<<1)
			| ((x&0x1010101010101010ull)>>1)
			| ((x&0x2020202020202020ull)>>3)
			| ((x&0x4040404040404040ull)>>5)
			| ((x&0x8080808080808080ull)>>7);
	}else if(bits==2){
		x=((x&0x0303030303030303ull)<<6)
			| ((x&0x0c0c0c0c0c0c0c0cull)<<2)
			| ((x&0x3030303030303030ull)>>2)
			| ((x&0xc0c0c0c0c0c0c0c0ull)>>6);
	}else if(bits==4){
		x=((x&0x0f0f0f0f0f0f0f0full)<<4)
			| ((x&0xf0f0f0f0f0f0f0f0ull)>>4);
	}
	return x;
}

uint32_t vmin(uint32_t a, uint32_t b)
{ return std::min(a,b); }

uint32_t vmin(uint32_t a, uint32_t b, uint32_t c)
{ return std::min(a,std::min(b,c)); }

uint32_t vmin(uint32_t a, uint32_t b, uint32_t c, uint32_t d)
{ return std::min(std::min(a,d),std::min(b,c)); }

uint32_t vmin(uint32_t a, uint32_t b, uint32_t c, uint32_t d, uint32_t e)
{ return std::min(e, std::min(std::min(a,d),std::min(b,c))); }

uint32_t vmax(uint32_t a, uint32_t b)
{ return std::max(a,b); }

uint32_t vmax(uint32_t a, uint32_t b, uint32_t c)
{ return std::max(a,std::max(b,c)); }

uint32_t vmax(uint32_t a, uint32_t b, uint32_t c, uint32_t d)
{ return std::max(std::max(a,d),std::max(b,c)); }

uint32_t vmax(uint32_t a, uint32_t b, uint32_t c, uint32_t d, uint32_t e)
{ return std::max(e, std::max(std::max(a,d),std::max(b,c))); }

void unpack_blob(unsigned w, unsigned h, unsigned bits, const uint64_t *pRaw, uint32_t *pUnpacked)
{
	uint64_t buffer=0;
	unsigned bufferedBits=0;

	const uint64_t MASK=0xFFFFFFFFFFFFFFFFULL>>(64-bits);

	for(unsigned i=0;i<w*h;i++){
		if(bufferedBits==0){
			buffer=shuffle64(bits, *pRaw++); // Note that this also flips the order of the bits in each pixel value.
			bufferedBits=64;
		}

		pUnpacked[i]=buffer&MASK;
		buffer=buffer>>bits;
		bufferedBits-=bits;
	}

	assert(bufferedBits==0);
}


void pack_blob(unsigned w, unsigned h, unsigned bits, const uint32_t *pUnpacked, uint64_t *pRaw)
{
	uint64_t buffer=0;
	unsigned bufferedBits=0;

	const uint64_t MASK=0xFFFFFFFFFFFFFFFFULL>>(64-bits);

	for(unsigned i=0;i<w*h;i++){
		buffer=buffer | (uint64_t(pUnpacked[i]&MASK)<< bufferedBits);
		bufferedBits+=bits;

		if(bufferedBits==64){
			*pRaw++ = shuffle64(bits, buffer);
			buffer=0;
			bufferedBits=0;
		}
	}

	assert(bufferedBits==0);
}

bool read_blob(int fd, uint64_t cbBlob, void *pBlob)
{
	uint8_t *pBytes=(uint8_t*)pBlob;

	uint64_t done=0;
	while(done<cbBlob){
		int todo=(int)std::min(uint64_t(1)<<30, cbBlob-done);

		int got=read(fd, pBytes+done, todo);
		if(got==0 && done==0)
			return false;	// end of file
		if(got<=0)
			throw std::invalid_argument("Read failure.");
		done+=got;
	}

	return true;
}

void write_blob(int fd, uint64_t cbBlob, const void *pBlob)
{
	const uint8_t *pBytes=(const uint8_t*)pBlob;

	uint64_t done=0;
	while(done<cbBlob){
		int todo=(int)std::min(uint64_t(1)<<30, cbBlob-done);

		int got=write(fd, pBytes+done, todo);
		if(got<=0)
			throw std::invalid_argument("Write failure.");
		done+=got;
	}
}

void invert(int levels, unsigned w, unsigned h, unsigned bits, std::vector<uint32_t> &pixels)
{
	uint32_t mask=0xFFFFFFFFul>>bits;

	for(unsigned i=0;i<w*h;i++){
		pixels[i]=mask-pixels[i];
	}
}

void packandwriteline_8(unsigned w, const uint32_t *pUnpacked, int fd)
{
	// Minimum width is 8 so this should be fine

	// Size of one line
	uint64_t cbLine=uint64_t(w);

	// Raw buffer for conversion
	std::vector<uint64_t> raw(cbLine/8);

	const uint64_t MASK=0x00000000000000FFULL;

	uint64_t done=0;

	for(unsigned i=0;i<w/8;i++){
		raw[i]=raw[i] | (uint64_t(pUnpacked[8*i]&MASK)<< 0);
		raw[i]=raw[i] | (uint64_t(pUnpacked[8*i+1]&MASK)<< 8);
		raw[i]=raw[i] | (uint64_t(pUnpacked[8*i+2]&MASK)<< 16);
		raw[i]=raw[i] | (uint64_t(pUnpacked[8*i+3]&MASK)<< 24);
		raw[i]=raw[i] | (uint64_t(pUnpacked[8*i+4]&MASK)<< 32);
		raw[i]=raw[i] | (uint64_t(pUnpacked[8*i+5]&MASK)<< 40);
		raw[i]=raw[i] | (uint64_t(pUnpacked[8*i+6]&MASK)<< 48);
		raw[i]=raw[i] | (uint64_t(pUnpacked[8*i+7]&MASK)<< 56);
	}

	while(done<cbLine){
		int todo=(int)std::min(uint64_t(1)<<30, cbLine-done);

		int got=write(fd, &raw[0]+done, todo);
		if(got<=0)
			throw std::invalid_argument("Write failure.");
		done+=got;
	}
}

bool readandunpack_8 (int fd, unsigned w, uint32_t *pUnpacked)
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
			return false;	// end of file
		if(got<=0)
			throw std::invalid_argument("Read failure.");
		done+=got;
	}

	const uint64_t MASK=0x00000000000000FFULL;

	for(unsigned i=0;i<w/8;i++){

		pUnpacked[i*8]= raw[i] & 0x00000000000000FFULL;

		pUnpacked[i*8+1]= (raw[i] & 0x000000000000FF00ULL) >> 8;

		pUnpacked[i*8+2]= (raw[i] & 0x0000000000FF0000ULL) >> 16;

		pUnpacked[i*8+3]= (raw[i] & 0x00000000FF000000ULL) >> 24;

		pUnpacked[i*8+4]= (raw[i] & 0x000000FF00000000ULL) >> 32;

		pUnpacked[i*8+5]= (raw[i] & 0x0000FF0000000000ULL) >> 40;

		pUnpacked[i*8+6]= (raw[i] & 0x00FF000000000000ULL) >> 48;

		pUnpacked[i*8+7]= (raw[i] & 0xFF00000000000000ULL) >> 56;

	}

	return true;
}

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

bool readandunpack_sse_8 (unsigned w, int fd, __m128i *output)
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
			return false;	// end of file
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

	return true;
}