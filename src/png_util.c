#include "png_util.h"

#ifdef IMG_PATH
#include <stdlib.h>

// Read a PNG file and return image data (RGB format)
unsigned char *read_png_file(const char *filename, int *width, int *height)
{
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        perror("Error opening file");
        return NULL;
    }

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) {
        fclose(fp);
        perror("Error creating png_structp");
        return NULL;
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        png_destroy_read_struct(&png, NULL, NULL);
        fclose(fp);
        perror("Error creating png_infop");
        return NULL;
    }

    if (setjmp(png_jmpbuf(png))) {
        png_destroy_read_struct(&png, &info, NULL);
        fclose(fp);
        perror("Error during PNG reading");
        return NULL;
    }

    png_init_io(png, fp);
    png_read_info(png, info);

    *width       = png_get_image_width(png, info);
    *height      = png_get_image_height(png, info);
    png_byte color_type = png_get_color_type(png, info);
    png_byte bit_depth  = png_get_bit_depth(png, info);

    // Adjust transformations for different PNG types
    png_set_strip_16(png);
    png_set_packing(png);
    png_set_expand(png);

    if (color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_palette_to_rgb(png);

    if (color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
        png_set_gray_to_rgb(png);

    if (color_type == PNG_COLOR_TYPE_RGB_ALPHA)
        png_set_strip_alpha(png);
    else if (png_get_valid(png, info, PNG_INFO_tRNS)) {
        png_set_tRNS_to_alpha(png);
        png_set_strip_alpha(png);
    }

    png_read_update_info(png, info);

    int row_bytes = png_get_rowbytes(png, info);
    unsigned char *image_data = (unsigned char *)malloc(row_bytes * (*height));
    png_bytep *row_pointers   = (png_bytep *)malloc(sizeof(png_bytep) * (*height));

    for (int y = 0; y < *height; y++) {
        // Invert rows so the image appears right-side up
        row_pointers[y] = image_data + ((*height - y - 1) * row_bytes);
    }

    png_read_image(png, row_pointers);

    fclose(fp);
    png_destroy_read_struct(&png, &info, NULL);
    free(row_pointers);

    return image_data;
}

#endif
