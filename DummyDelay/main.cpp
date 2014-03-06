//
//  main.cpp
//  LatencyTimer
//
//  Created by Richard Worrall on 04/03/2014.
//
//

#include <iostream>
#include <stdexcept>
#include <fstream>
#include <streambuf>
#include <unistd.h>
#include <sys/time.h>

#define batchsize 1000

int main(int argc, const char * argv[])
{
    char* out = new char[batchsize];
    
    while(1)
    {
        int done = 0;
        while (done < batchsize)
        {
            int got = read(0, out, batchsize);
            done += got;
        }
        usleep(40000);
        done = 0;
        while (done < batchsize)
        {
            int got = write(1, out, batchsize);
            done += got;
        }
    }
}

