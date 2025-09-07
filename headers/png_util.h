#ifndef PNG_UTIL_H
#define PNG_UTIL_H

#include "config.h"
#ifdef IMG_PATH

#include <png.h>

/**
 * @brief Reads a PNG file and returns its RGB image data.
 *
 * This function loads a PNG image from the specified file path and converts it
 * into a raw RGB buffer (8-bit per channel). The image is vertically flipped so
 * that the first row corresponds to the top of the image.
 *
 * @param filename The path to the PNG file.
 * @param width Pointer to an integer to store the image width.
 * @param height Pointer to an integer to store the image height.
 * @return Pointer to a malloc-allocated RGB buffer, or NULL if loading fails.
 *         The caller is responsible for freeing this buffer.
 */
unsigned char *read_png_file(const char *filename, int *width, int *height);

#endif // IMG_PATH
#endif // PNG_UTIL_H
