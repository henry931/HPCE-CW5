#include <fcntl.h>
#include <limits.h>
#include <poll.h>
#include <unistd.h>
#include <sys/time.h>
#include <queue>
#include <iostream>

int is_ready(int fd) {
    fd_set fdset;
    struct timeval timeout;
    FD_ZERO(&fdset);
    FD_SET(fd, &fdset);
    timeout.tv_sec = 0;
    timeout.tv_usec = 1;
    return select(fd+1, &fdset, NULL, NULL, &timeout) == 1 ? 1 : 0;
}

int main(int argc, const char * argv[])
{
    if(argc<3){
        fprintf(stderr, "Usage: width bits\n");
        exit(1);
    }
    
    unsigned w=atoi(argv[1]);
    unsigned bits=atoi(argv[2]);
    
    unsigned batchSize = (w*bits)/32;
    
    char* out = new char[batchSize];
    
    std::deque<timeval*> ts_queue;
    std::deque<timeval*> old;
    
    timeval t_valAfter;
    
    long min = LONG_MAX;
    long max = 0;
    
    int bytesOutstanding = 0;
    
    int got = 0;
    
    while(1)
    {
        got = write(1, out, batchSize);
        
        timeval* t;
        
        if (old.empty()) {
            t = new timeval;
        }
        else
        {
            t = old.front();
            old.pop_front();
        }
        
        gettimeofday(t, NULL);
        
        ts_queue.push_back(t);
        
        bytesOutstanding += got;
        
        
        while(is_ready(fileno(stdin)))
        {
            if (ts_queue.empty()) break;
            
            got = read(0, out, batchSize);
            
            bytesOutstanding -= got;
            
            gettimeofday(&t_valAfter, NULL);
            
            timeval* t_valBefore_p = ts_queue.front();
            
            old.push_back(t_valBefore_p);
            
            ts_queue.pop_front();
            
            long microseconds =(t_valAfter.tv_sec - t_valBefore_p->tv_sec)*1000000L + t_valAfter.tv_usec - t_valBefore_p->tv_usec;
            
            if (microseconds < min) min = microseconds;
            if (microseconds > max) max = microseconds;
            
            if (ts_queue.size()+old.size() == 0) continue;
            
            fprintf(stderr, "Adjusted Last: %7.0f  Est. Pipe Buffer Size: %7lu Est. Application Buffer Size: %7lu\r",((float)microseconds)*(((float)old.size())/(((float)(ts_queue.size()-1))+((float)old.size()))),ts_queue.size()-1,old.size());
            
        }
    }
}

