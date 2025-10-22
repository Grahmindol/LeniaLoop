#include "cuda_convolve_utils.h"
#include "cuda_fft_utils.h"
#include "config.h"

#include <stdlib.h>
#include <stdio.h>

//======================================================================
// 📦 GPU Buffers (globales)
//======================================================================

// --- Kernel FFT buffer
static cufftComplex* d_KernelFFT_reg = NULL;

// --- FFT intermediate buffers
static cufftComplex* d_imageFFT_reg_batchs  = NULL;
static cufftComplex* d_resultFFT_reg_batchs = NULL;

#ifdef REGION_SIZE
// --- Region batching buffers
static int* batch_x_dev   = NULL;
static int* batch_y_dev   = NULL;
static int* d_new_batches = NULL;

// Nombre de batchs actifs
static int g_batch_count  = 0;
#endif

//======================================================================
// ⚡ Utility GPU Functions
//======================================================================

/**
 * @brief GPU helper to fetch a pixel value (R channel only).
 *
 * @param img Source buffer (device memory).
 * @param x   Pixel x coordinate.
 * @param y   Pixel y coordinate.
 * @return    Pixel value as complex number (Re=R, Im=0).
 */
__device__ cufftComplex get_value_device(const float* img, int x, int y) {
    int xi = x & (GRID_SIZE - 1);   // wrap-around modulo GRID_SIZE
    int yi = y & (GRID_SIZE - 1);
    int idx = ((xi * GRID_SIZE) + yi) * 3; // pixel index, 3 channels

    return (cufftComplex){ img[idx], 0.0f };
}

// ========================================================
// Section 2 — Précompute kernel FFT & buffers allocation
// ========================================================

/**
 * @brief Pré-calculer la FFT d'un kernel symétrique (padding + FFT) et
 *        charger le résultat sur GPU.
 *
 * @param kernel Pointeur vers les valeurs floats du kernel (row-major).
 * @param size   Taille du kernel (size x size). Doit être <= FFT_SIZE_REG (ou REGION_SIZE si défini).
 */
void precompute_kernelFFT(const float* kernel, int size) {
#ifdef REGION_SIZE
    if (size > REGION_SIZE) {
        fprintf(stderr, "ERROR: region size too small for the kernel (got %d, max %d)\n", size, REGION_SIZE);
        exit(EXIT_FAILURE);
    }
#else
    if (size > GRID_SIZE) {
        fprintf(stderr, "ERROR: grid size too small for the kernel (got %d, max %d)\n", size, GRID_SIZE);
        exit(EXIT_FAILURE);
    }
#endif

    // libérer ancien buffer si existant
    if (d_KernelFFT_reg) {
        CHECK_CUDA(cudaFree(d_KernelFFT_reg));
        d_KernelFFT_reg = NULL;
    }

    printf("INFO: pre-calculating kernel FFT of size %dx%d\n", size, size);

    // allocation hôte pour construire la version centrée/paddée
    size_t fft_elems = (size_t)FFT_SIZE_REG * (size_t)FFT_SIZE_REG;
    cufftComplex* hostBuf = (cufftComplex*)malloc(fft_elems * sizeof(cufftComplex));
    if (!hostBuf) {
        fprintf(stderr, "ERROR: host allocation failed for KernelFFT buffer\n");
        exit(EXIT_FAILURE);
    }

    // initialiser à zéro
    for (size_t i = 0; i < fft_elems; ++i) {
        hostBuf[i].x = 0.0f;
        hostBuf[i].y = 0.0f;
    }

    // centrer le kernel dans la grille FFT
    int startX = (FFT_SIZE_REG - size) / 2;
    int startY = (FFT_SIZE_REG - size) / 2;
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            int tx = startX + x;
            int ty = startY + y;
            if (tx >= 0 && tx < FFT_SIZE_REG && ty >= 0 && ty < FFT_SIZE_REG) {
                hostBuf[(size_t)ty * FFT_SIZE_REG + tx].x = kernel[y * size + x];
            }
        }
    }

    // copier sur le device
    CHECK_CUDA(cudaMalloc((void**)&d_KernelFFT_reg, fft_elems * sizeof(cufftComplex)));
    CHECK_CUDA(cudaMemcpy(d_KernelFFT_reg, hostBuf, fft_elems * sizeof(cufftComplex), cudaMemcpyHostToDevice));

    // init plans, fft, et setup (fonctions externes)
#ifdef FFT_OPTI_ENABLE // si on optimise pas via FFT alors on ne calcul pas la fft et on garde le spatial
    init_FFT_plans(1);
    fft2d_batch(d_KernelFFT_reg);
    setup_kernel_texture(d_KernelFFT_reg);
#endif
    free(hostBuf);
}

/**
 * @brief Allouer les buffers device nécessaires au batching de convolution.
 *        Si REGION_SIZE est défini, alloue les buffers pour plusieurs batchs.
 */
void init_convolve_buffer() {
#ifdef REGION_SIZE
    size_t batch_array_size = 2 * (size_t)REGION_NUMBER * (size_t)REGION_NUMBER * sizeof(int);
    size_t total_elems = 2 * (size_t)REGION_NUMBER * (size_t)REGION_NUMBER * (size_t)FFT_SIZE_REG * (size_t)FFT_SIZE_REG;
    size_t total_bytes = total_elems * sizeof(cufftComplex);
#else
    size_t total_bytes = (size_t)FFT_SIZE_REG * (size_t)FFT_SIZE_REG * sizeof(cufftComplex);
#endif

    CHECK_CUDA(cudaMalloc((void**)&d_imageFFT_reg_batchs,  total_bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_resultFFT_reg_batchs, total_bytes));

#ifdef REGION_SIZE
    CHECK_CUDA(cudaMalloc((void**)&batch_x_dev, batch_array_size));
    CHECK_CUDA(cudaMalloc((void**)&batch_y_dev, batch_array_size));
    CHECK_CUDA(cudaMalloc((void**)&d_new_batches, sizeof(int)));
    // init counter
    g_batch_count = 0;
#endif
}

/**
 * @brief Libérer tous les buffers alloués pour la convolution.
 */
void destroy_convolve_buffer() {
#ifdef REGION_SIZE
    if (batch_x_dev)    { CHECK_CUDA(cudaFree(batch_x_dev));    batch_x_dev = NULL; }
    if (batch_y_dev)    { CHECK_CUDA(cudaFree(batch_y_dev));    batch_y_dev = NULL; }
    if (d_new_batches)  { CHECK_CUDA(cudaFree(d_new_batches));  d_new_batches = NULL; }
    g_batch_count = 0;
#endif

    // liberer les ressources FFT & textures
    destroy_FFT_plans();
    free_kernel_texture();

    if (d_imageFFT_reg_batchs)  { CHECK_CUDA(cudaFree(d_imageFFT_reg_batchs));  d_imageFFT_reg_batchs  = NULL; }
    if (d_resultFFT_reg_batchs) { CHECK_CUDA(cudaFree(d_resultFFT_reg_batchs)); d_resultFFT_reg_batchs = NULL; }
    if (d_KernelFFT_reg)        { CHECK_CUDA(cudaFree(d_KernelFFT_reg));       d_KernelFFT_reg         = NULL; }
}

// ========================================================
// Section 3 — Kernels de copie / padding / FFT-shift
// ========================================================


#ifdef REGION_SIZE
// ========================================================
// Version avec batching (REGION_SIZE activé)
// ========================================================

/**
 * @brief Kernel device — copie d’une région (batch) avec padding centré.
 */
__global__ void copy_and_pad_kernel_large(
    cufftComplex* dst_batches,
    const float* image,
    const int* batch_positions_x,
    const int* batch_positions_y,
    int batch_count
) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x  = blockIdx.x * blockDim.x + tx;
    int y  = blockIdx.y * blockDim.y + ty;
    int region_idx = blockIdx.z;

    if (region_idx >= batch_count || x >= FFT_SIZE_REG || y >= FFT_SIZE_REG) return;

    __shared__ int xc, yc;
    if (tx == 0 && ty == 0) {
        xc = batch_positions_x[region_idx];
        yc = batch_positions_y[region_idx];
    }
    __syncthreads();

    int quarter = FFT_SIZE_REG >> 2;
    int img_x = xc + x - quarter;
    int img_y = yc + y - quarter;

    cufftComplex val = get_value_device(image, img_x, img_y);
    size_t dst_idx = (size_t)region_idx * FFT_SIZE_REG * FFT_SIZE_REG + y * FFT_SIZE_REG + x;
    dst_batches[dst_idx] = val;
}

/**
 * @brief Wrapper host — copie des régions avec padding (appel kernel).
 */
static inline void copy_region_buffer_with_padding(
    cufftComplex* dst_dev,
    const float* image_dev
) {
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid(
        (FFT_SIZE_REG + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (FFT_SIZE_REG + threadsPerBlock.y - 1) / threadsPerBlock.y,
        g_batch_count
    );

    copy_and_pad_kernel_large<<<blocksPerGrid, threadsPerBlock>>>(
        dst_dev, image_dev, batch_x_dev, batch_y_dev, g_batch_count
    );
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

/**
 * @brief Kernel device — supprime padding + applique FFT-shift + recopie vers image finale.
 */
__global__ void remove_padding_and_center_shift_kernel(
    const cufftComplex* __restrict__ src_batch,
    float* __restrict__ result_out,
    const int* __restrict__ pos_x,
    const int* __restrict__ pos_y,
    int batch_count
) {
    int k = blockIdx.z;
    if (k >= batch_count) return;

    int half    = FFT_SIZE_REG >> 1;
    int quarter = FFT_SIZE_REG >> 2;

    int i = blockIdx.y * blockDim.y + threadIdx.y; // row
    int j = blockIdx.x * blockDim.x + threadIdx.x; // col
    if (i >= half || j >= half) return;

    // FFT shift (swap quadrants via XOR)
    int shift_i = (i + quarter) ^ half;
    int shift_j = (j + quarter) ^ half;
    int shifted_idx = shift_i * FFT_SIZE_REG + shift_j;

    float val = src_batch[k * FFT_SIZE_REG * FFT_SIZE_REG + shifted_idx].x;

    // coordonnées sortie wrap
    int out_x = (pos_x[k] + j + GRID_SIZE) % GRID_SIZE;
    int out_y = (pos_y[k] + i + GRID_SIZE) % GRID_SIZE;

    result_out[out_y + out_x * GRID_SIZE] = val;
}

/**
 * @brief Wrapper host — suppression du padding + FFT-shift.
 */
static inline void remove_padding_and_center_shift_cuda(
    cufftComplex* d_resultFFT_reg_batchs,
    float* d_result_out
) {
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid(
        (FFT_SIZE_REG / 2 + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (FFT_SIZE_REG / 2 + threadsPerBlock.y - 1) / threadsPerBlock.y,
        g_batch_count
    );

    remove_padding_and_center_shift_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_resultFFT_reg_batchs, d_result_out,
        batch_x_dev, batch_y_dev, g_batch_count
    );
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

#else
// ========================================================
// Version simple (sans REGION_SIZE)
// ========================================================

/**
 * @brief Kernel device — applique FFT-shift et écrit dans la sortie.
 */
__global__ void center_shift_kernel(
    const cufftComplex* __restrict__ src,
    float* __restrict__ result_out
) {
    int half = FFT_SIZE_REG >> 1;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= FFT_SIZE_REG || j >= FFT_SIZE_REG) return;

    int shift_i = i ^ half;
    int shift_j = j ^ half;
    int shifted_idx = shift_i * FFT_SIZE_REG + shift_j;

    float val = src[shifted_idx].x;
    int out_x = (j+1 + GRID_SIZE) % GRID_SIZE; // ici on se decal de (1,1) mais on ne sait pas pourquoi ....
    int out_y = (i+1 + GRID_SIZE) % GRID_SIZE;

    result_out[out_y + out_x * GRID_SIZE] = val;
}

/**
 * @brief Wrapper host — applique FFT-shift sur image entière.
 */
static inline void center_shift_cuda(
    cufftComplex* d_resultFFT_reg_batchs,
    float* d_result_out
) {
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid(
        (FFT_SIZE_REG + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (FFT_SIZE_REG + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    center_shift_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_resultFFT_reg_batchs, d_result_out
    );
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

/**
 * @brief Kernel device — copie image vers buffer FFT (padded).
 */
__global__ void copy_and_pad_kernel_large(
    cufftComplex* dst,
    const float* image
) {
    int img_x = blockIdx.x * blockDim.x + threadIdx.x;
    int img_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (img_x >= FFT_SIZE_REG || img_y >= FFT_SIZE_REG) return;

    cufftComplex val = get_value_device(image, img_x, img_y);
    dst[img_y * FFT_SIZE_REG + img_x] = val;
}

/**
 * @brief Wrapper host — copie image complète dans buffer FFT.
 */
static inline void copy_image(
    cufftComplex* dst_dev,
    const float* image_dev
) {
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid(
        (FFT_SIZE_REG + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (FFT_SIZE_REG + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    copy_and_pad_kernel_large<<<blocksPerGrid, threadsPerBlock>>>(dst_dev, image_dev);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}
#endif

// ========================================================
// Section 4 — Détection et masquage des régions actives
// ========================================================
#ifdef REGION_SIZE

#define FLAG_ACTIVE  0b0011
#define FLAG_CORNER  0b0100
#define FLAG_MASKED  0b1000

/**
 * @brief Kernel device — détecte les coins de régions actives et incrémente le compteur de nouveaux batchs.
 *
 * @param d_sub_roi     Tableau device des sous-régions (flags).
 * @param d_batch_x     Tableau device des positions X des batchs détectés.
 * @param d_batch_y     Tableau device des positions Y des batchs détectés.
 * @param d_new_batches Compteur device des nouveaux batchs détectés.
 */
__global__ void detectCornersKernel(
    int* d_sub_roi,
    int* d_batch_x,
    int* d_batch_y,
    int* d_new_batches
) {
    int sx = blockIdx.x * blockDim.x + threadIdx.x;
    int sy = blockIdx.y * blockDim.y + threadIdx.y;
    if (sx >= SUB_REG_NUMBER || sy >= SUB_REG_NUMBER) return;

    int idx = sy * SUB_REG_NUMBER + sx;
    if (!(d_sub_roi[idx] & FLAG_ACTIVE)) return;

    bool left_valid = (sy > 0) && (d_sub_roi[(sy - 1) * SUB_REG_NUMBER + sx] & FLAG_ACTIVE);
    bool up_valid   = (sx > 0) && (d_sub_roi[sy * SUB_REG_NUMBER + (sx - 1)] & FLAG_ACTIVE);
    if (left_valid || up_valid) return;

    int insert = atomicAdd(d_new_batches, 1);
    d_batch_x[insert] = sx * SUB_REG_SIZE;
    d_batch_y[insert] = sy * SUB_REG_SIZE;

#ifdef DEBUG_LIFE_MOD
    atomicOr(&d_sub_roi[idx], FLAG_CORNER);
#endif
}

/**
 * @brief Kernel device — masque les sous-régions déjà incluses dans un batch.
 *
 * @param d_sub_roi     Tableau device des sous-régions (flags).
 * @param d_batch_x     Tableau device des positions X des batchs.
 * @param d_batch_y     Tableau device des positions Y des batchs.
 * @param d_new_batches Compteur device des nouveaux batchs à traiter.
 */
__global__ void maskRegionsKernel(
    int* d_sub_roi,
    int* d_batch_x,
    int* d_batch_y,
    int* d_new_batches
) {
    int dx = threadIdx.x + blockIdx.x * blockDim.x;
    int dy = threadIdx.y + blockIdx.y * blockDim.y;
    int r_id = threadIdx.z + blockIdx.z * blockDim.z;

    if (r_id >= *d_new_batches) return;
    if (dx >= SUB_REG_PER_REG || dy >= SUB_REG_PER_REG) return;

    int gx = d_batch_x[r_id] / SUB_REG_SIZE + dx;
    int gy = d_batch_y[r_id] / SUB_REG_SIZE + dy;
    if (gx >= SUB_REG_NUMBER || gy >= SUB_REG_NUMBER) return;

    if (d_sub_roi[gy * SUB_REG_NUMBER + gx] & FLAG_ACTIVE) {
#ifdef DEBUG_LIFE_MOD
        atomicOr(&d_sub_roi[gy * SUB_REG_NUMBER + gx], FLAG_MASKED);
#else
        atomicOr(&d_sub_roi[gy * SUB_REG_NUMBER + gx], FLAG_MASKED);
#endif
        atomicAnd(&d_sub_roi[gy * SUB_REG_NUMBER + gx], ~FLAG_ACTIVE);
    }
}

/**
 * @brief Compute batch positions à partir des régions actives.
 *
 * @param d_sub_roi Tableau device des flags de sous-régions.
 * @return Nombre de régions connexes détectées.
 */
int compute_batch_positions(int* d_sub_roi) {
    int total_batches = 0;
    int h_new;
    int connex_reg;

    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((SUB_REG_NUMBER + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (SUB_REG_NUMBER + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cudaMemset(d_new_batches, 0, sizeof(int));

    // Détecte les coins initiaux
    detectCornersKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_sub_roi, batch_x_dev + total_batches, batch_y_dev + total_batches, d_new_batches
    );
    CHECK_CUDA(cudaGetLastError());

    cudaMemcpy(&h_new, d_new_batches, sizeof(int), cudaMemcpyDeviceToHost);
    if ((connex_reg = h_new) == 0) return 0;

    while (true) {
        // Masque les sous-régions des batchs détectés
        dim3 threadsPerBlock3D(16, 16, 4);
        dim3 blocksPerGrid3D(
            (SUB_REG_PER_REG + threadsPerBlock3D.x - 1) / threadsPerBlock3D.x,
            (SUB_REG_PER_REG + threadsPerBlock3D.y - 1) / threadsPerBlock3D.y,
            (h_new + threadsPerBlock3D.z - 1) / threadsPerBlock3D.z
        );

        maskRegionsKernel<<<blocksPerGrid3D, threadsPerBlock3D>>>(
            d_sub_roi, batch_x_dev + total_batches, batch_y_dev + total_batches, d_new_batches
        );
        CHECK_CUDA(cudaGetLastError());

        total_batches += h_new;

        // Reset compteur
        cudaMemset(d_new_batches, 0, sizeof(int));

        // Détecte les nouveaux coins
        detectCornersKernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_sub_roi, batch_x_dev + total_batches, batch_y_dev + total_batches, d_new_batches
        );
        CHECK_CUDA(cudaGetLastError());

        cudaMemcpy(&h_new, d_new_batches, sizeof(int), cudaMemcpyDeviceToHost);
        if (h_new == 0) break;
    }

    g_batch_count = total_batches;
    return connex_reg;
}
#endif

/**
 * @brief Convolution 2D circulaire en FFT.
 *
 * Pour REGION_SIZE : batching avec détection de sous-régions actives.
 * Sans REGION_SIZE : convolution simple sur toute l'image.
 *
 * @param d_image       Image source (device float*).
 * @param d_roi         Tableau des flags de régions actives (device int*), ou NULL si pas REGION_SIZE.
 * @param d_result_out  Résultat de la convolution (device float*).
 * @param debug         Buffer debug (optionnel, device float*), peut être NULL.
 * @return Nombre de régions connexes traitées (1 si pas REGION_SIZE).
 */
int convolve2d_fft_circular(const float* d_image, int* d_roi, float* d_result_out, float* debug) {

#ifdef REGION_SIZE

CUDA_TIMING_BEGIN(fft2d);
    // ----------------------------------------------------
    // 1️⃣ Calcul des positions de batchs
    // ----------------------------------------------------
    int connex_reg = compute_batch_positions(d_roi);
    static int last_batch_count = 0;
    if (g_batch_count == 0) return connex_reg;

    if (g_batch_count != last_batch_count) {
        init_FFT_plans(g_batch_count);
        setup_kernel_texture(d_KernelFFT_reg);
        last_batch_count = g_batch_count;
    }

    // ----------------------------------------------------
    // 2️⃣ Copie de l'image vers le buffer batch avec padding
    // ----------------------------------------------------
    copy_region_buffer_with_padding(d_imageFFT_reg_batchs, d_image);
CUDA_TIMING_END(fft2d);

CUDA_TIMING_BEGIN(pointwise);
    // ----------------------------------------------------
    // 3️⃣ FFT sur tous les batchs
    // ----------------------------------------------------
    fft2d_batch(d_imageFFT_reg_batchs);
CUDA_TIMING_END(pointwise);
    // ----------------------------------------------------
    // 4️⃣ Pointwise multiplication avec le kernel FFT
    // ----------------------------------------------------
    complex_pointwise_broadcast(d_imageFFT_reg_batchs, d_resultFFT_reg_batchs, g_batch_count);

CUDA_TIMING_BEGIN(ifft2d);
    // ----------------------------------------------------
    // 5️⃣ IFFT sur tous les batchs
    // ----------------------------------------------------
    ifft2d_batch(d_resultFFT_reg_batchs);

    // ----------------------------------------------------
    // 6️⃣ Retrait du padding et recentrage
    // ----------------------------------------------------
    remove_padding_and_center_shift_cuda(d_resultFFT_reg_batchs, d_result_out);
CUDA_TIMING_END(ifft2d);
    return connex_reg;

#else
    // ----------------------------------------------------
    // Cas simple : pas de REGION_SIZE
    // ----------------------------------------------------
CUDA_TIMING_BEGIN(fft2d);
    copy_image(d_imageFFT_reg_batchs, d_image);
    fft2d_batch(d_imageFFT_reg_batchs);
CUDA_TIMING_END(fft2d);

CUDA_TIMING_BEGIN(pointwise);
    complex_pointwise_broadcast(d_imageFFT_reg_batchs, d_resultFFT_reg_batchs, 1);
CUDA_TIMING_END(pointwise);

CUDA_TIMING_BEGIN(ifft2d);
    ifft2d_batch(d_resultFFT_reg_batchs);
    center_shift_cuda(d_resultFFT_reg_batchs, d_result_out);
CUDA_TIMING_END(ifft2d);
    CHECK_CUDA(cudaGetLastError());
    return 1;
#endif
}

// ========================================================
// Section 5 — Vertion encore plus simple sans FFT
// ========================================================

__global__ void spatial_convolution_kernel(
    const float* __restrict__ image,
    float* __restrict__ result,const cufftComplex* d_Kernel
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= GRID_SIZE || y >= GRID_SIZE) return;

    double sum = 0.0;

    for (int ky = -RADIUS; ky <= RADIUS; ++ky) {
        for (int kx = -RADIUS; kx <= RADIUS; ++kx) {
            float val = get_value_device(image, x + kx, y + ky).x; // canal R
            float coeff = d_Kernel[(ky-1 + (FFT_SIZE_REG / 2)) * FFT_SIZE_REG + (kx-1 + (FFT_SIZE_REG / 2))].x;
            sum += val * coeff;
        }
    }
    result[x * GRID_SIZE + y] = (float)sum;
}

int convolve2d_circular(const float* d_image, int* d_roi, float* d_result_out, float* debug) {

#ifdef FFT_OPTI_ENABLE
    // ======================
    // 🔹 Version FFT
    // ======================
    return convolve2d_fft_circular(d_image, d_roi, d_result_out, debug);

#else
    // ======================
    // 🔹 Version kernel spatial
    // ======================

    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid(
        (GRID_SIZE + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (GRID_SIZE + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    // Kernel spatial simple : convolution directe
    spatial_convolution_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_image, d_result_out,d_KernelFFT_reg
    );

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    return 1; // (ou nombre de régions traitées si REGION_SIZE)
#endif
}