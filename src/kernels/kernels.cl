__kernel void erode_kernel(__global uint* input,__global uint* output)
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

__kernel void dilate_kernel(__global uint* input,__global uint* output)
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