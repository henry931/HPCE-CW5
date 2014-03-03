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

std::string LoadSource(const char *fileName);

uint64_t shuffle64(unsigned bits, uint64_t x);

uint32_t vmin(uint32_t a, uint32_t b);
uint32_t vmin(uint32_t a, uint32_t b, uint32_t c);
uint32_t vmin(uint32_t a, uint32_t b, uint32_t c, uint32_t d);
uint32_t vmin(uint32_t a, uint32_t b, uint32_t c, uint32_t d, uint32_t e);

uint32_t vmax(uint32_t a, uint32_t b);
uint32_t vmax(uint32_t a, uint32_t b, uint32_t c);
uint32_t vmax(uint32_t a, uint32_t b, uint32_t c, uint32_t d);
uint32_t vmax(uint32_t a, uint32_t b, uint32_t c, uint32_t d, uint32_t e);

void unpack_blob(unsigned w, unsigned h, unsigned bits, const uint64_t *pRaw, uint32_t *pUnpacked);
void pack_blob(unsigned w, unsigned h, unsigned bits, const uint32_t *pUnpacked, uint64_t *pRaw);

bool read_blob(int fd, uint64_t cbBlob, void *pBlob);
void write_blob(int fd, uint64_t cbBlob, const void *pBlob);

void invert(int levels, unsigned w, unsigned h, unsigned bits, std::vector<uint32_t> &pixels);