//TODO: Reverse other kernels

uint min2(uint a,uint b)
{
    uint mask = 0xC0000000;
    uint output = 0x00000000;
    
    uint x;
    uint y;
    
    for(int i=0;i<16;i++)
    {
        x = a & mask;
        y = b & mask;
        
        // Non-branching minimum taken from http://graphics.stanford.edu/~seander/bithacks.html
        
        output |= y ^ ((x ^ y) & -(x < y));
        
        mask >>= 2;
    }
    
    return output;
}

uint max2(uint a,uint b)
{
    uint mask = 0xC0000000;
    uint output = 0x00000000;
    
    uint x;
    uint y;
    
    for(int i=0;i<16;i++)
    {
        x = a & mask;
        y = b & mask;
        
        // Non-branching minimum taken from http://graphics.stanford.edu/~seander/bithacks.html
        
        output |= x ^ ((x ^ y) & -(x < y));
        
        mask >>= 2;
    }
    return output;
}

uint min4(uint a,uint b)
{
    uint mask = 0xF0000000;
    uint output = 0x00000000;
    
    uint x;
    uint y;
    
    for(int i=0;i<8;i++)
    {
        x = a & mask;
        y = b & mask;
        
        // Non-branching minimum taken from http://graphics.stanford.edu/~seander/bithacks.html
        
        output |= y ^ ((x ^ y) & -(x < y));
        
        mask >>= 4;
    }
    
    return output;
}

uint max4(uint a,uint b)
{
    uint mask = 0xF0000000;
    uint output = 0x00000000;
    
    uint x;
    uint y;
    
    for(int i=0;i<8;i++)
    {
        x = a & mask;
        y = b & mask;
        
        // Non-branching minimum taken from http://graphics.stanford.edu/~seander/bithacks.html
        
        output |= x ^ ((x ^ y) & -(x < y));
        
        mask >>= 4;
    }
    
    return output;
}

uint min8(uint a,uint b)
{
    uint mask = 0xFF000000;
    uint output = 0x00000000;
    
    uint x;
    uint y;
    
    for(int i=0;i<4;i++)
    {
        x = a & mask;
        y = b & mask;
        
        // Non-branching minimum taken from http://graphics.stanford.edu/~seander/bithacks.html
        
        output |= y ^ ((x ^ y) & -(x < y));
        
        mask >>= 8;
    }
    
    return output;
}

uint max8(uint a,uint b)
{
    uint mask = 0xFF000000;
    uint output = 0x00000000;
    
    uint x;
    uint y;
    
    for(int i=0;i<4;i++)
    {
        x = a & mask;
        y = b & mask;
        
        // Non-branching minimum taken from http://graphics.stanford.edu/~seander/bithacks.html
        
        output |= x ^ ((x ^ y) & -(x < y));
        
        mask >>= 8;
    }
    
    return output;
}

uint min16(uint a,uint b)
{
    uint mask = 0xFFFF0000;
    uint output = 0x00000000;
    
    uint x;
    uint y;
    
    x = a & mask;
    y = b & mask;
    
    output |= y ^ ((x ^ y) & -(x < y));
    
    mask >>= 16;
    
    x = a & mask;
    y = b & mask;
    
    output |= y ^ ((x ^ y) & -(x < y));
    
    // Non-branching minimum taken from http://graphics.stanford.edu/~seander/bithacks.html
    
    return output;
}

uint max16(uint a,uint b)
{
    uint mask = 0xFFFF0000;
    uint output = 0x00000000;
    
    uint x;
    uint y;
    
    x = a & mask;
    y = b & mask;
    
    output |= y ^ ((x ^ y) & -(x < y));
    
    mask >>= 16;
    
    x = a & mask;
    y = b & mask;
    
    output |= x ^ ((x ^ y) & -(x < y));
    
    // Non-branching minimum taken from http://graphics.stanford.edu/~seander/bithacks.html
    
    return output;
}

__kernel void erode_kernel_32(__global uint* input,__global uint* output)
{
    uint x=get_global_id(0);
    uint y=get_global_id(1);
    
    uint w=get_global_size(0);
    uint h=get_global_size(1);
    
    uint index=y*w + x;
    
    uint above;
    uint below;
    uint left;
    uint right;
    
    if(y > 0) above = input[index-w];
    else above = 0xFFFFFFFF;
    if(y < h-1) below = input[index+w];
    else below = 0xFFFFFFFF;
    if (x > 0) left = input[index-1];
    else left = 0xFFFFFFFF;
    if (x < w-1) right = input[index+1];
    else right = 0xFFFFFFFF;
    
    output[index] = min(min(min(above,below),min(left,right)),input[index]);

}

__kernel void dilate_kernel_32(__global uint* input,__global uint* output)
{
    uint x=get_global_id(0);
    uint y=get_global_id(1);
    
    uint w=get_global_size(0);
    uint h=get_global_size(1);
    
    uint index=y*w + x;
    
    uint above;
    uint below;
    uint left;
    uint right;
    
    if(y > 0) above = input[index-w];
    else above = 0x0;
    if(y < h-1) below = input[index+w];
    else below = 0x0;
    if (x > 0) left = input[index-1];
    else left = 0x0;
    if (x < w-1) right = input[index+1];
    else right = 0x0;
    
    output[index] = max(max(max(above,below),max(left,right)),input[index]);
    
}

__kernel void erode_kernel_1(__global uint* input,__global uint* output)
{
    uint x=get_global_id(0);
    uint y=get_global_id(1);
    
    uint w=get_global_size(0);
    uint h=get_global_size(1);
    
    uint index=y*w + x;
    
    uint above;
    uint below;
    uint left;
    uint right;
    
    if(y > 0) above = input[index-w];
    else above = 0xFFFFFFFF;
    if(y < h-1) below = input[index+w];
    else below = 0xFFFFFFFF;
    if (x > 0) left = input[index-1];
    else left = 0xFFFFFFFF;
    if (x < w-1) right = input[index+1];
    else right = 0xFFFFFFFF;
    
    uint center = input[index];
    
    center &= ((center << 1) | ((right >> 31) & 0x1));
    center &= ((center >> 1) | ((left & 0x1) << 31));
    center &= above;
    center &= below;

    output[index] = center;
}

__kernel void dilate_kernel_1(__global uint* input,__global uint* output)
{
    uint x=get_global_id(0);
    uint y=get_global_id(1);
    
    uint w=get_global_size(0);
    uint h=get_global_size(1);
    
    uint index=y*w + x;
    
    uint above;
    uint below;
    uint left;
    uint right;
    
    if(y > 0) above = input[index-w];
    else above = 0x0;
    if(y < h-1) below = input[index+w];
    else below = 0x0;
    if (x > 0) left = input[index-1];
    else left = 0x0;
    if (x < w-1) right = input[index+1];
    else right = 0x0;
    
    uint center = input[index];
    
    center |= ((center << 1) | ((right >> 31) & 0x1));
    center |= ((center >> 1) | ((left & 0x1) << 31));
    center |= above;
    center |= below;
    
    output[index] = center;

}

__kernel void erode_kernel_2(__global uint* input,__global uint* output)
{
    uint x=get_global_id(0);
    uint y=get_global_id(1);
    
    uint w=get_global_size(0);
    uint h=get_global_size(1);
    
    uint index=y*w + x;
    
    uint above;
    uint below;
    uint left;
    uint right;
    
    if(y > 0) above = input[index-w];
    else above = 0xFFFFFFFF;
    if(y < h-1) below = input[index+w];
    else below = 0xFFFFFFFF;
    if (x > 0) left = input[index-1];
    else left = 0xFFFFFFFF;
    if (x < w-1) right = input[index+1];
    else right = 0xFFFFFFFF;
    
    uint center = input[index];
    
    center = min2(center, (center << 2) | ((right >> 30) & 0x3));
    center = min2(center, (center >> 2) | ((left & 0x3) << 30));
    center = min2(center, above);
    center = min2(center, below);
    
    output[index] = center;
}

__kernel void dilate_kernel_2(__global uint* input,__global uint* output)
{
    uint x=get_global_id(0);
    uint y=get_global_id(1);
    
    uint w=get_global_size(0);
    uint h=get_global_size(1);
    
    uint index=y*w + x;
    
    uint above;
    uint below;
    uint left;
    uint right;
    
    if(y > 0) above = input[index-w];
    else above = 0x0;
    if(y < h-1) below = input[index+w];
    else below = 0x0;
    if (x > 0) left = input[index-1];
    else left = 0x0;
    if (x < w-1) right = input[index+1];
    else right = 0x0;
    
    uint center = input[index];
    
    center = max2(center, (center << 2) | ((right >> 30) & 0x3));
    center = max2(center, (center >> 2) | ((left & 0x3) << 30));
    center = max2(center, above);
    center = max2(center, below);
    
    output[index] = center;
}

__kernel void erode_kernel_4(__global uint* input,__global uint* output)
{
    uint x=get_global_id(0);
    uint y=get_global_id(1);
    
    uint w=get_global_size(0);
    uint h=get_global_size(1);
    
    uint index=y*w + x;
    
    uint above;
    uint below;
    uint left;
    uint right;
    
    if(y > 0) above = input[index-w];
    else above = 0xFFFFFFFF;
    if(y < h-1) below = input[index+w];
    else below = 0xFFFFFFFF;
    if (x > 0) left = input[index-1];
    else left = 0xFFFFFFFF;
    if (x < w-1) right = input[index+1];
    else right = 0xFFFFFFFF;
    
    uint center = input[index];

    center = min4(center, (center << 4) | ((right >> 28) & 0xF));
    center = min4(center, (center >> 4) | ((left & 0xF) << 28));
    center = min4(center, above);
    center = min4(center, below);
        
    output[index] = center;
}

__kernel void dilate_kernel_4(__global uint* input,__global uint* output)
{
    uint x=get_global_id(0);
    uint y=get_global_id(1);
    
    uint w=get_global_size(0);
    uint h=get_global_size(1);
    
    uint index=y*w + x;
    
    uint above;
    uint below;
    uint left;
    uint right;
    
    if(y > 0) above = input[index-w];
    else above = 0x0;
    if(y < h-1) below = input[index+w];
    else below = 0x0;
    if (x > 0) left = input[index-1];
    else left = 0x0;
    if (x < w-1) right = input[index+1];
    else right = 0x0;
    
    uint center = input[index];

    center = max4(center, (center << 4) | ((right >> 28) & 0xF));
    center = max4(center, (center >> 4) | ((left & 0xF) << 28));
    center = max4(center, above);
    center = max4(center, below);
    
    output[index] = center;
}

__kernel void erode_kernel_8(__global uint* input,__global uint* output)
{
    uint x=get_global_id(0);
    uint y=get_global_id(1);
    
    uint w=get_global_size(0);
    uint h=get_global_size(1);
    
    uint index=y*w + x;
    
    uint above;
    uint below;
    uint left;
    uint right;
    
    if(y > 0) above = input[index-w];
    else above = 0xFFFFFFFF;
    if(y < h-1) below = input[index+w];
    else below = 0xFFFFFFFF;
    if (x > 0) left = input[index-1];
    else left = 0xFFFFFFFF;
    if (x < w-1) right = input[index+1];
    else right = 0xFFFFFFFF;
    
    uint center = input[index];
    
    center = min8(center, (center << 8) | ((left >> 24) & 0xFF));
    center = min8(center, (center >> 8) | ((right & 0xFF) << 24));
    
    center = min8(center, above);
    center = min8(center, below);
    
    output[index] = center;
}

__kernel void dilate_kernel_8(__global uint* input,__global uint* output)
{
    uint x=get_global_id(0);
    uint y=get_global_id(1);
    
    uint w=get_global_size(0);
    uint h=get_global_size(1);
    
    uint index=y*w + x;
    
    uint above;
    uint below;
    uint left;
    uint right;
    
    if(y > 0) above = input[index-w];
    else above = 0x0;
    if(y < h-1) below = input[index+w];
    else below = 0x0;
    if (x > 0) left = input[index-1];
    else left = 0x0;
    if (x < w-1) right = input[index+1];
    else right = 0x0;
    
    uint center = input[index];
    
    center = max8(center, (center << 8) | ((left >> 24) & 0xFF));
    center = max8(center, (center >> 8) | ((right & 0xFF) << 24));
    
    center = max8(center, above);
    center = max8(center, below);
    
    output[index] = center;
}

__kernel void erode_kernel_16(__global uint* input,__global uint* output)
{
    uint x=get_global_id(0);
    uint y=get_global_id(1);
    
    uint w=get_global_size(0);
    uint h=get_global_size(1);
    
    uint index=y*w + x;
    
    uint above;
    uint below;
    uint left;
    uint right;
    
    if(y > 0) above = input[index-w];
    else above = 0xFFFFFFFF;
    if(y < h-1) below = input[index+w];
    else below = 0xFFFFFFFF;
    if (x > 0) left = input[index-1];
    else left = 0xFFFFFFFF;
    if (x < w-1) right = input[index+1];
    else right = 0xFFFFFFFF;
    
    uint center = input[index];
    
    center = min16(center, (center << 16) | ((right >> 16) & 0xFFFF));
    center = min16(center, (center >> 16) | ((left & 0xFFFF) << 16));
    center = min16(center, above);
    center = min16(center, below);
    
    output[index] = center;
}

__kernel void dilate_kernel_16(__global uint* input,__global uint* output)
{
    uint x=get_global_id(0);
    uint y=get_global_id(1);
    
    uint w=get_global_size(0);
    uint h=get_global_size(1);
    
    uint index=y*w + x;
    
    uint above;
    uint below;
    uint left;
    uint right;
    
    if(y > 0) above = input[index-w];
    else above = 0x0;
    if(y < h-1) below = input[index+w];
    else below = 0x0;
    if (x > 0) left = input[index-1];
    else left = 0x0;
    if (x < w-1) right = input[index+1];
    else right = 0x0;
    
    uint center = input[index];
    
    center = max16(center, (center << 16) | ((right >> 16) & 0xFFFF));
    center = max16(center, (center >> 16) | ((left & 0xFFFF) << 16));
    center = max16(center, above);
    center = max16(center, below);
    
    output[index] = center;
}