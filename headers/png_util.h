#include "config.h"
#ifdef IMG_PATH
#ifndef PNG_UTIL_H
#define PNG_UTIL_H

#include <png.h>


unsigned char *read_png_file(const char *filename, int *width, int *height);

#endif
#endif