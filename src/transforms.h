// Header files for windows compilation
#ifdef _WIN32
#include <io.h>
#include <stdint.h>

// Header files for OSX compilation
#else
#include <unistd.h>
#endif

// Shared Headers
#include <vector>

void erode(unsigned w, unsigned h, const std::vector<uint32_t> &input, std::vector<uint32_t> &output);
void dilate(unsigned w, unsigned h, const std::vector<uint32_t> &input, std::vector<uint32_t> &output);

void process(int levels, unsigned w, unsigned h, unsigned /*bits*/, std::vector<uint32_t> &pixels);

void process_opencl(int levels, unsigned w, unsigned h, unsigned /*bits*/, std::vector<uint32_t> &pixels);

void process_tbb(int levels, unsigned w, unsigned h, unsigned /*bits*/, std::vector<uint32_t> &pixels);

void erode_parfor(unsigned w, unsigned h, const std::vector<uint32_t> &input, std::vector<uint32_t> &output);
void dilate_parfor(unsigned w, unsigned h, const std::vector<uint32_t> &input, std::vector<uint32_t> &output);

void erode_line(unsigned w, const std::vector<uint32_t> &inputA, const std::vector<uint32_t> &inputB, const std::vector<uint32_t> &inputC, std::vector<uint32_t> &output);
void dilate_line(unsigned w, const std::vector<uint32_t> &inputA, const std::vector<uint32_t> &inputB, const std::vector<uint32_t> &inputC, std::vector<uint32_t> &output);

void erode_line_top(unsigned w, const std::vector<uint32_t> &inputB, const std::vector<uint32_t> &inputC, std::vector<uint32_t> &output);
void erode_line_bottom(unsigned w, const std::vector<uint32_t> &inputA, const std::vector<uint32_t> &inputB, std::vector<uint32_t> &output);
void dilate_line_top(unsigned w, const std::vector<uint32_t> &inputB, const std::vector<uint32_t> &inputC, std::vector<uint32_t> &output);
void dilate_line_bottom(unsigned w, const std::vector<uint32_t> &inputA, const std::vector<uint32_t> &inputB, std::vector<uint32_t> &output);