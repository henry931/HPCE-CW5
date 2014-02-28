#include <unistd.h>
#include <vector>

void erode(unsigned w, unsigned h, const std::vector<uint32_t> &input, std::vector<uint32_t> &output);
void dilate(unsigned w, unsigned h, const std::vector<uint32_t> &input, std::vector<uint32_t> &output);

void process(int levels, unsigned w, unsigned h, unsigned /*bits*/, std::vector<uint32_t> &pixels);