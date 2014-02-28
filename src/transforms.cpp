//
//  utilities.cpp
//  Process
//
//  Created by Richard Worrall on 28/02/2014.
//
//

#include "transforms.h"
#include "utilities.h"

void erode(unsigned w, unsigned h, const std::vector<uint32_t> &input, std::vector<uint32_t> &output)
{
	auto in=[&](int x, int y) -> uint32_t { return input[y*w+x]; };
	auto out=[&](int x, int y) -> uint32_t & {return output[y*w+x]; };
	
	for(unsigned x=0;x<w;x++){
		if(x==0){
			out(0,0)=vmin(in(0,0), in(0,1), in(1,0));
			for(unsigned y=1;y<h-1;y++){
				out(0,y)=vmin(in(0,y), in(0,y-1), in(1,y), in(0,y+1));
			}
			out(0,h-1)=vmin(in(0,h-1), in(0,h-2), in(1,h-1));
		}else if(x<w-1){
			out(x,0)=vmin(in(x,0), in(x-1,0), in(x,1), in(x+1,0));
			for(unsigned y=1;y<h-1;y++){
				out(x,y)=vmin(in(x,y), in(x-1,y), in(x,y-1), in(x,y+1), in(x+1,y));
			}
			out(x,h-1)=vmin(in(x,h-1), in(x-1,h-1), in(x,h-2), in(x+1,h-1));
		}else{
			out(w-1,0)=vmin(in(w-1,0), in(w-1,1), in(w-2,0));
			for(unsigned y=1;y<h-1;y++){
				out(w-1,y)=vmin(in(w-1,y), in(w-1,y-1), in(w-2,y), in(w-1,y+1));
			}
			out(w-1,h-1)=vmin(in(w-1,h-1), in(w-1,h-2), in(w-2,h-1));
		}
	}
}

void dilate(unsigned w, unsigned h, const std::vector<uint32_t> &input, std::vector<uint32_t> &output)
{
	auto in=[&](int x, int y) -> uint32_t { return input[y*w+x]; };
	auto out=[&](int x, int y) -> uint32_t & {return output[y*w+x]; };
	
	for(unsigned x=0;x<w;x++){
		if(x==0){
			out(0,0)=vmax(in(0,0), in(0,1), in(1,0));
			for(unsigned y=1;y<h-1;y++){
				out(0,y)=vmax(in(0,y), in(0,y-1), in(1,y), in(0,y+1));
			}
			out(0,h-1)=vmax(in(0,h-1), in(0,h-2), in(1,h-1));
		}else if(x<w-1){
			out(x,0)=vmax(in(x,0), in(x-1,0), in(x,1), in(x+1,0));
			for(unsigned y=1;y<h-1;y++){
				out(x,y)=vmax(in(x,y), in(x-1,y), in(x,y-1), in(x,y+1), in(x+1,y));
			}
			out(x,h-1)=vmax(in(x,h-1), in(x-1,h-1), in(x,h-2), in(x+1,h-1));
		}else{
			out(w-1,0)=vmax(in(w-1,0), in(w-1,1), in(w-2,0));
			for(unsigned y=1;y<h-1;y++){
				out(w-1,y)=vmax(in(w-1,y), in(w-1,y-1), in(w-2,y), in(w-1,y+1));
			}
			out(w-1,h-1)=vmax(in(w-1,h-1), in(w-1,h-2), in(w-2,h-1));
		}
	}
}

void process(int levels, unsigned w, unsigned h, unsigned /*bits*/, std::vector<uint32_t> &pixels)
{
	std::vector<uint32_t> buffer(w*h);
	
	// Depending on whether levels is positive or negative,
	// we flip the order round.
	auto fwd=levels < 0 ? erode : dilate;
	auto rev=levels < 0 ? dilate : erode;
	
	for(int i=0;i<std::abs(levels);i++){
		fwd(w, h, pixels, buffer);
		std::swap(pixels, buffer);
	}
	for(int i=0;i<std::abs(levels);i++){
		rev(w,h,pixels, buffer);
		std::swap(pixels, buffer);
	}
}
