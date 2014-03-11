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

__kernel void erode_kernel_32(uint aboveOverride,uint belowOverride,uint inputOffset,uint outputOffset,__global uint* input,__global uint* output)
{
    uint x=get_global_id(0);
    
    int w=get_global_size(0);
    
    uint maxsize = 4*w;
    
    int inputIndex=inputOffset*w + x;
    uint outputIndex=outputOffset*w + x;
    
    uint above = input[(inputIndex-w) + (maxsize & -((inputIndex-w) < 0))]  | aboveOverride;
    uint below = input[(inputIndex+w) - (maxsize & -((inputIndex+w) >= maxsize))] | belowOverride;
    uint left =  input[(inputIndex-1) + (maxsize & -((inputIndex-1) < 0))]  | -(x == 0);
    uint right = input[(inputIndex+1) - (maxsize & -((inputIndex+1) >= maxsize))] | -(x == w-1);
    
    output[outputIndex] = min(min(min(above,below),min(left,right)),input[inputIndex]);
}

__kernel void dilate_kernel_32(uint aboveOverride,uint belowOverride,uint inputOffset,uint outputOffset,__global uint* input,__global uint* output)
{
    uint x=get_global_id(0);
    
    int w=get_global_size(0);
    
    uint maxsize = 4*w;
    
    int inputIndex=inputOffset*w + x;
    uint outputIndex=outputOffset*w + x;
    
    uint above = input[(inputIndex-w) + (maxsize & -((inputIndex-w) < 0))]  & aboveOverride;
    uint below = input[(inputIndex+w) - (maxsize & -((inputIndex+w) >= maxsize))] & belowOverride;
    uint left =  input[(inputIndex-1) + (maxsize & -((inputIndex-1) < 0))]  & -(x != 0);
    uint right = input[(inputIndex+1) - (maxsize & -((inputIndex+1) >= maxsize))] & -(x != w-1);
    
    output[outputIndex] = max(max(max(above,below),max(left,right)),input[inputIndex]);
}

__kernel void erode_kernel_1(uint aboveOverride,uint belowOverride,uint inputOffset,uint outputOffset,__global uint* input,__global uint* output)
{
    uint x=get_global_id(0);
    
    int w=get_global_size(0);
    
    uint maxsize = 4*w;
    
    int inputIndex=inputOffset*w + x;
    uint outputIndex=outputOffset*w + x;
    
    uint above = input[(inputIndex-w) + (maxsize & -((inputIndex-w) < 0))]  | aboveOverride;
    uint below = input[(inputIndex+w) - (maxsize & -((inputIndex+w) >= maxsize))] | belowOverride;
    uint left =  input[(inputIndex-1) + (maxsize & -((inputIndex-1) < 0))]  | -(x == 0);
    uint right = input[(inputIndex+1) - (maxsize & -((inputIndex+1) >= maxsize))] | -(x == w-1);
    uint center = input[inputIndex];
    
    center &= ((center << 1) | ((left >> 31) & 0x1));
    center &= ((center >> 1) | ((right & 0x1) << 31));
    center &= above;
    center &= below;
    
    output[outputIndex] = center;
}

__kernel void dilate_kernel_1(uint aboveOverride,uint belowOverride,uint inputOffset,uint outputOffset,__global uint* input,__global uint* output)
{
    uint x=get_global_id(0);
    
    int w=get_global_size(0);
    
    uint maxsize = 4*w;
    
    int inputIndex=inputOffset*w + x;
    uint outputIndex=outputOffset*w + x;
    
    uint above = input[(inputIndex-w) + (maxsize & -((inputIndex-w) < 0))]  & aboveOverride;
    uint below = input[(inputIndex+w) - (maxsize & -((inputIndex+w) >= maxsize))] & belowOverride;
    uint left =  input[(inputIndex-1) + (maxsize & -((inputIndex-1) < 0))]  & -(x != 0);
    uint right = input[(inputIndex+1) - (maxsize & -((inputIndex+1) >= maxsize))] & -(x != w-1);
    uint center = input[inputIndex];
    
    center |= ((center << 1) | ((left >> 31) & 0x1));
    center |= ((center >> 1) | ((right & 0x1) << 31));
    center |= above;
    center |= below;
    
    output[outputIndex] = center;
    
}

__kernel void erode_kernel_2(uint aboveOverride,uint belowOverride,uint inputOffset,uint outputOffset,__global uint* input,__global uint* output)
{
    uint x=get_global_id(0);
    
    int w=get_global_size(0);
    
    uint maxsize = 4*w;
    
    int inputIndex=inputOffset*w + x;
    uint outputIndex=outputOffset*w + x;
    
    uint above = input[(inputIndex-w) + (maxsize & -((inputIndex-w) < 0))]  | aboveOverride;
    uint below = input[(inputIndex+w) - (maxsize & -((inputIndex+w) >= maxsize))] | belowOverride;
    uint left =  input[(inputIndex-1) + (maxsize & -((inputIndex-1) < 0))]  | -(x == 0);
    uint right = input[(inputIndex+1) - (maxsize & -((inputIndex+1) >= maxsize))] | -(x == w-1);
    uint center = input[inputIndex];
    
    center = min2(center, (center << 2) | ((left >> 30) & 0x3));
    center = min2(center, (center >> 2) | ((right & 0x3) << 30));
    center = min2(center, above);
    center = min2(center, below);
    
    output[outputIndex] = center;
}

__kernel void dilate_kernel_2(uint aboveOverride,uint belowOverride,uint inputOffset,uint outputOffset,__global uint* input,__global uint* output)
{
    uint x=get_global_id(0);
    
    int w=get_global_size(0);
    
    uint maxsize = 4*w;
    
    int inputIndex=inputOffset*w + x;
    uint outputIndex=outputOffset*w + x;
    
    uint above = input[(inputIndex-w) + (maxsize & -((inputIndex-w) < 0))]  & aboveOverride;
    uint below = input[(inputIndex+w) - (maxsize & -((inputIndex+w) >= maxsize))] & belowOverride;
    uint left =  input[(inputIndex-1) + (maxsize & -((inputIndex-1) < 0))]  & -(x != 0);
    uint right = input[(inputIndex+1) - (maxsize & -((inputIndex+1) >= maxsize))] & -(x != w-1);
    uint center = input[inputIndex];
    
    center = max2(center, (center << 2) | ((left >> 30) & 0x3));
    center = max2(center, (center >> 2) | ((right & 0x3) << 30));
    center = max2(center, above);
    center = max2(center, below);
    
    output[outputIndex] = center;
}

__kernel void erode_kernel_4(uint aboveOverride,uint belowOverride,uint inputOffset,uint outputOffset,__global uint* input,__global uint* output)
{
    uint x=get_global_id(0);
    
    int w=get_global_size(0);
    
    uint maxsize = 4*w;
    
    int inputIndex=inputOffset*w + x;
    uint outputIndex=outputOffset*w + x;
    
    uint above = input[(inputIndex-w) + (maxsize & -((inputIndex-w) < 0))]  | aboveOverride;
    uint below = input[(inputIndex+w) - (maxsize & -((inputIndex+w) >= maxsize))] | belowOverride;
    uint left =  input[(inputIndex-1) + (maxsize & -((inputIndex-1) < 0))]  | -(x == 0);
    uint right = input[(inputIndex+1) - (maxsize & -((inputIndex+1) >= maxsize))] | -(x == w-1);
    uint center = input[inputIndex];
    
    center = min4(center, (center << 4) | ((left >> 28) & 0xF));
    center = min4(center, (center >> 4) | ((right & 0xF) << 28));
    center = min4(center, above);
    center = min4(center, below);
    
    output[outputIndex] = center;
}

__kernel void dilate_kernel_4(uint aboveOverride,uint belowOverride,uint inputOffset,uint outputOffset,__global uint* input,__global uint* output)
{
    uint x=get_global_id(0);
    
    int w=get_global_size(0);
    
    uint maxsize = 4*w;
    
    int inputIndex=inputOffset*w + x;
    uint outputIndex=outputOffset*w + x;
    
    uint above = input[(inputIndex-w) + (maxsize & -((inputIndex-w) < 0))]  & aboveOverride;
    uint below = input[(inputIndex+w) - (maxsize & -((inputIndex+w) >= maxsize))] & belowOverride;
    uint left =  input[(inputIndex-1) + (maxsize & -((inputIndex-1) < 0))]  & -(x != 0);
    uint right = input[(inputIndex+1) - (maxsize & -((inputIndex+1) >= maxsize))] & -(x != w-1);
    uint center = input[inputIndex];
    
    center = max4(center, (center << 4) | ((left >> 28) & 0xF));
    center = max4(center, (center >> 4) | ((right & 0xF) << 28));
    center = max4(center, above);
    center = max4(center, below);
    
    output[outputIndex] = center;
}

__kernel void erode_kernel_8(uint aboveOverride,uint belowOverride,uint inputOffset,uint outputOffset,__global uint* input,__global uint* output)
{
    uint x=get_global_id(0);
    
    int w=get_global_size(0);
    
    uint maxsize = 4*w;
    
    int inputIndex=inputOffset*w + x;
    uint outputIndex=outputOffset*w + x;
    
    uint above = input[(inputIndex-w) + (maxsize & -((inputIndex-w) < 0))]  | aboveOverride;
    uint below = input[(inputIndex+w) - (maxsize & -((inputIndex+w) >= maxsize))] | belowOverride;
    uint left =  input[(inputIndex-1) + (maxsize & -((inputIndex-1) < 0))]  | -(x == 0);
    uint right = input[(inputIndex+1) - (maxsize & -((inputIndex+1) >= maxsize))] | -(x == w-1);
    uint center = input[inputIndex];
    
    center = min8(center, (center << 8) | ((left >> 24) & 0xFF));
    center = min8(center, (center >> 8) | ((right & 0xFF) << 24));
    
    center = min8(center, above);
    center = min8(center, below);
    
    output[outputIndex] = center;
}

__kernel void dilate_kernel_8(uint aboveOverride,uint belowOverride,uint inputOffset,uint outputOffset,__global uint* input,__global uint* output)
{
    uint x=get_global_id(0);
    
    int w=get_global_size(0);
    
    uint maxsize = 4*w;
    
    int inputIndex=inputOffset*w + x;
    uint outputIndex=outputOffset*w + x;
    
    uint above = input[(inputIndex-w) + (maxsize & -((inputIndex-w) < 0))]  & aboveOverride;
    uint below = input[(inputIndex+w) - (maxsize & -((inputIndex+w) >= maxsize))] & belowOverride;
    uint left =  input[(inputIndex-1) + (maxsize & -((inputIndex-1) < 0))]  & -(x != 0);
    uint right = input[(inputIndex+1) - (maxsize & -((inputIndex+1) >= maxsize))] & -(x != w-1);
    uint center = input[inputIndex];
    
    center = max8(center, (center << 8) | ((left >> 24) & 0xFF));
    center = max8(center, (center >> 8) | ((right & 0xFF) << 24));
    
    center = max8(center, above);
    center = max8(center, below);
    
    output[outputIndex] = center;
}

__kernel void erode_kernel_16(uint aboveOverride,uint belowOverride,uint inputOffset,uint outputOffset,__global uint* input,__global uint* output)
{
    uint x=get_global_id(0);
    
    int w=get_global_size(0);
    
    uint maxsize = 4*w;
    
    int inputIndex=inputOffset*w + x;
    uint outputIndex=outputOffset*w + x;
    
    uint above = input[(inputIndex-w) + (maxsize & -((inputIndex-w) < 0))]  | aboveOverride;
    uint below = input[(inputIndex+w) - (maxsize & -((inputIndex+w) >= maxsize))] | belowOverride;
    uint left =  input[(inputIndex-1) + (maxsize & -((inputIndex-1) < 0))]  | -(x == 0);
    uint right = input[(inputIndex+1) - (maxsize & -((inputIndex+1) >= maxsize))] | -(x == w-1);
    uint center = input[inputIndex];
    
    center = min16(center, (center << 16) | ((left >> 16) & 0xFFFF));
    center = min16(center, (center >> 16) | ((right & 0xFFFF) << 16));
    center = min16(center, above);
    center = min16(center, below);
    
    output[outputIndex] = center;
}

__kernel void dilate_kernel_16(uint aboveOverride,uint belowOverride,uint inputOffset,uint outputOffset,__global uint* input,__global uint* output)
{
    uint x=get_global_id(0);
    
    int w=get_global_size(0);
    
    uint maxsize = 4*w;
    
    int inputIndex=inputOffset*w + x;
    uint outputIndex=outputOffset*w + x;
    
    uint above = input[(inputIndex-w) + (maxsize & -((inputIndex-w) < 0))]  & aboveOverride;
    uint below = input[(inputIndex+w) - (maxsize & -((inputIndex+w) >= maxsize))] & belowOverride;
    uint left =  input[(inputIndex-1) + (maxsize & -((inputIndex-1) < 0))]  & -(x != 0);
    uint right = input[(inputIndex+1) - (maxsize & -((inputIndex+1) >= maxsize))] & -(x != w-1);
    uint center = input[inputIndex];
    
    center = max16(center, (center << 16) | ((left >> 16) & 0xFFFF));
    center = max16(center, (center >> 16) | ((right & 0xFFFF) << 16));
    center = max16(center, above);
    center = max16(center, below);
    
    output[outputIndex] = center;
}