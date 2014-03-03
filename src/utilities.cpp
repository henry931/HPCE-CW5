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