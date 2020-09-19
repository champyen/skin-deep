#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "vec_type.h"

#ifndef min
#define min(a, b) (((a) < (b)) ? (a) : (b))
#endif
#define min3(x, y, z) min(min(x, y), z)

#ifndef max
#define max(a, b) (((a) > (b)) ? (a) : (b))
#endif
#define max3(x, y, z) max(max(x, y), z)

#define CLAMP2BYTE(v) (((unsigned) (v)) < 255 ? (v) : (v < 0) ? 0 : 255)

#define TILING 1
#define OPT_VECTOR 1

unsigned int detect(uint8_t *pixel,
                    uint8_t **plane,
                    int width,
                    int height,
                    int channels)
{
    int stride = width * channels;
    int last_col = width * channels - channels;
    int last_row = height * stride - stride;

    unsigned int row_sum[16384] = {0};
    #pragma omp parallel for
    for (int y = 0; y < height; y++) {
        int cur_row = stride * y;
        int next_row = min(cur_row + stride, last_row);
        uint8_t *next_scanline = pixel + next_row;
        uint8_t *cur_scanline = pixel + cur_row;
        int x = 0;
#if OPT_VECTOR
        /* clang-format off */
        for (; x+16 < width; x += 8){
            int cur_col = x * channels;
            u16_16 vsrc0 = __builtin_convertvector(*(u8_16*)(cur_scanline + cur_col), u16_16);
            u16_16 vsrc1 = __builtin_convertvector(*(u8_16*)(cur_scanline + cur_col + 16), u16_16);

            u16_8 v_r_avg, v_g_avg, v_b_avg;
            u8_8 vR, vG, vB;
            v_r_avg  = __builtin_shufflevector(vsrc0, vsrc1, 0, 3,  6,  9, 12, 15, 18, 21);
            v_g_avg  = __builtin_shufflevector(vsrc0, vsrc1, 1, 4,  7, 10, 13, 16, 19, 22);
            v_b_avg  = __builtin_shufflevector(vsrc0, vsrc1, 2, 5,  8, 11, 14, 17, 20, 23);
            vR = __builtin_convertvector(v_r_avg, u8_8);
            vG = __builtin_convertvector(v_g_avg, u8_8);
            vB = __builtin_convertvector(v_b_avg, u8_8);
#if OPT_PLANE
            *(u8_8*)(plane[0] + y*width + x)= vR;
            *(u8_8*)(plane[1] + y*width + x)= vG;
            *(u8_8*)(plane[2] + y*width + x)= vB;
#endif
            v_r_avg += __builtin_shufflevector(vsrc0, vsrc1, 3, 6,  9, 12, 15, 18, 21, 24);
            v_g_avg += __builtin_shufflevector(vsrc0, vsrc1, 4, 7, 10, 13, 16, 19, 22, 25);
            v_b_avg += __builtin_shufflevector(vsrc0, vsrc1, 5, 8, 11, 14, 17, 20, 23, 26);

            vsrc0 = __builtin_convertvector(*(u8_16*)(next_scanline + cur_col), u16_16);
            vsrc1 = __builtin_convertvector(*(u8_16*)(next_scanline + cur_col + 16), u16_16);
            v_r_avg += __builtin_shufflevector(vsrc0, vsrc1, 0, 3,  6,  9, 12, 15, 18, 21);
            v_g_avg += __builtin_shufflevector(vsrc0, vsrc1, 1, 4,  7, 10, 13, 16, 19, 22);
            v_b_avg += __builtin_shufflevector(vsrc0, vsrc1, 2, 5,  8, 11, 14, 17, 20, 23);
            v_r_avg += __builtin_shufflevector(vsrc0, vsrc1, 3, 6,  9, 12, 15, 18, 21, 24);
            v_g_avg += __builtin_shufflevector(vsrc0, vsrc1, 4, 7, 10, 13, 16, 19, 22, 25);
            v_b_avg += __builtin_shufflevector(vsrc0, vsrc1, 5, 8, 11, 14, 17, 20, 23, 26);
            v_r_avg >>= 2;
            v_g_avg >>= 2;
            v_b_avg >>= 2;
            for(int i = 0; i < 8; i++){
                if (v_r_avg[i] >= 60 && v_g_avg[i] >= 40 && v_b_avg[i] >= 20 && v_r_avg[i] >= v_b_avg[i] && (v_r_avg[i] - v_g_avg[i]) >= 10)
                    if (max3(v_r_avg[i], v_g_avg[i], v_b_avg[i]) - min3(v_r_avg[i], v_g_avg[i], v_b_avg[i]) >= 10)
                        row_sum[y]++;
            }
        }
        /* clang-format on */
#endif
        for (; x < width; x++) {
            int cur_col = x * channels;
            int next_col = min(cur_col + channels, last_col);
            uint8_t *c00 = cur_scanline + cur_col;
            uint8_t *c10 = cur_scanline + next_col;
            uint8_t *c01 = next_scanline + cur_col;
            uint8_t *c11 = next_scanline + next_col;
            int r_avg = ((c00[0] + c10[0] + c01[0] + c11[0])) >> 2;
            int g_avg = ((c00[1] + c10[1] + c01[1] + c11[1])) >> 2;
            int b_avg = ((c00[2] + c10[2] + c01[2] + c11[2])) >> 2;

#if OPT_PLANE
            /* clang-format off */
            plane[0][y*width + x] = c00[0];
            plane[1][y*width + x] = c00[1];
            plane[2][y*width + x] = c00[2];
            /* clang-format on */
#endif
            /* TODO: detect appropriate RGB values */
            if (r_avg >= 60 && g_avg >= 40 && b_avg >= 20 && r_avg >= b_avg &&
                (r_avg - g_avg) >= 10)
                if (max3(r_avg, g_avg, b_avg) - min3(r_avg, g_avg, b_avg) >= 10)
                    row_sum[y]++;
        }
    }

    unsigned int sum = 0;
    for(int y = 0; y < height; y++)
        sum += row_sum[y];
    return sum;
}


void compute_offset(int *out, int len, int left, int right, int step)
{
    assert(out);
    assert((len >= 0) && (left >= 0) && (right >= 0));

    for (int x = -left; x < len + right; x++) {
        int pos = x;
        int len2 = 2 * len;
        if (pos < 0) {
            do {
                pos += len2;
            } while (pos < 0);
        } else if (pos >= len2) {
            do {
                pos -= len2;
            } while (pos >= len2);
        }
        if (pos >= len)
            pos = len2 - 1 - pos;
        out[x + left] = pos * step;
    }
}

void denoise(uint8_t *out,
             uint8_t *in,
             int *smooth_table,
             int width,
             int height,
             int channels,
             int radius)
{
    assert(in && out);
    assert(radius > 0);

    int window_size = (2 * radius + 1) * (2 * radius + 1);

    int *col_pow = malloc(width * channels * sizeof(int));
    int *col_val = malloc(width * channels * sizeof(int));
    int *row_pos = malloc((width + 2 * radius) * channels * sizeof(int));
    int *col_pos = malloc((height + 2 * radius) * channels * sizeof(int));

    int stride = width * channels;

    compute_offset(row_pos, width, radius, radius, channels);
    compute_offset(col_pos, height, radius, radius, stride);

    int *row_off = row_pos + radius;
    int *col_off = col_pos + radius;
    for (int y = 0; y < height; y++) {
        uint8_t *scan_in_line = in + y * stride;
        uint8_t *scan_out_line = out + y * stride;
        if (y == 0) {
            for (int x = 0; x < stride; x += channels) {
                int col_sum[3] = {0};
                int col_sum_pow[3] = {0};
                for (int z = -radius; z <= radius; z++) {
                    uint8_t *sample = in + col_off[z] + x;
                    for (int c = 0; c < channels; ++c) {
                        col_sum[c] += sample[c];
                        col_sum_pow[c] += sample[c] * sample[c];
                    }
                }
                for (int c = 0; c < channels; ++c) {
                    col_val[x + c] = col_sum[c];
                    col_pow[x + c] = col_sum_pow[c];
                }
            }
        } else {
            uint8_t *last_col = in + col_off[y - radius - 1];
            uint8_t *next_col = in + col_off[y + radius];
            for (int x = 0; x < stride; x += channels) {
                for (int c = 0; c < channels; ++c) {
                    col_val[x + c] -= last_col[x + c] - next_col[x + c];
                    col_pow[x + c] -= last_col[x + c] * last_col[x + c] -
                                      next_col[x + c] * next_col[x + c];
                }
            }
        }

        int prev_sum[3] = {0}, prev_sum_pow[3] = {0};
        for (int z = -radius; z <= radius; z++) {
            int index = row_off[z];
            for (int c = 0; c < channels; ++c) {
                prev_sum[c] += col_val[index + c];
                prev_sum_pow[c] += col_pow[index + c];
            }
        }

        for (int c = 0; c < channels; ++c) {
            int mean = prev_sum[c] / window_size;
            int diff = mean - scan_in_line[c];
            int edge = CLAMP2BYTE(diff);
            int masked_edge =
                (edge * scan_in_line[c] + (256 - edge) * mean) >> 8;
            int var = (prev_sum_pow[c] - mean * prev_sum[c]) / window_size;
            int out = masked_edge -
                      diff * var / (var + smooth_table[scan_in_line[c]]);
            scan_out_line[c] = CLAMP2BYTE(out);
        }

        scan_in_line += channels, scan_out_line += channels;
        for (int x = 1; x < width; x++) {
            int last_row = row_off[x - radius - 1];
            int next_row = row_off[x + radius];
            for (int c = 0; c < channels; ++c) {
                prev_sum[c] -= col_val[last_row + c] - col_val[next_row + c];
                prev_sum_pow[c] = prev_sum_pow[c] - col_pow[last_row + c] +
                                  col_pow[next_row + c];
                int mean = prev_sum[c] / window_size;
                int diff = mean - scan_in_line[c];
                int edge = CLAMP2BYTE(diff);
                int masked_edge =
                    (edge * scan_in_line[c] + (256 - edge) * mean) >> 8;
                int var = (prev_sum_pow[c] - mean * prev_sum[c]) / window_size;
                int out = masked_edge -
                          diff * var / (var + smooth_table[scan_in_line[c]]);
                scan_out_line[c] = CLAMP2BYTE(out);
            }

            scan_in_line += channels, scan_out_line += channels;
        }
    }

    free(col_pow);
    free(col_val);
    free(row_pos);
    free(col_pos);
}

/* clang-format off */
void denoise2(
    uint8_t *out,
    uint8_t **planes,
    int *smooth_table,
    int width,
    int height,
    int channels,
    int ch_idx,
    int radius
){
    uint8_t *in = planes[ch_idx];

    assert(in && out);
    assert(radius > 0);

    int window_size = (2*radius + 1) * (2*radius + 1);

    int *col_pow = calloc(width * sizeof(int), 1);
    int *col_val = calloc(width * sizeof(int), 1);
    int *row_pos = malloc((height + 2*radius + 1) * sizeof(int));
    int *col_pos = malloc((width + 2*radius + 1) * sizeof(int));

    compute_offset(row_pos+1, height, radius, radius, width);
    compute_offset(col_pos+1, width, radius, radius, 1);
    row_pos[0] = row_pos[2*radius+1];
    col_pos[0] = col_pos[2*radius+1];

    int *row_off = row_pos + radius+1;
    int *col_off = col_pos + radius+1;

    for (int x = 0; x < width; x ++) {
        for (int z = -radius; z <= radius; z++) {
            uint8_t sample = *(in + row_off[z] + x);
            col_val[x] += sample;
            col_pow[x] += sample * sample;
        }
    }
    for (int y = 0; y < height; y++) {
        uint8_t *scan_in_line = in + y*width;
        uint8_t *scan_out_line = out + y*width*channels;
        for (int x = 0; x < width; x++) {
            uint8_t *last_col = in + row_off[y - radius - 1];
            uint8_t *next_col = in + row_off[y + radius];
            col_val[x] -= last_col[x] - next_col[x];
            col_pow[x] -= last_col[x]*last_col[x] - next_col[x]*next_col[x];
        }

        int prev_sum = 0, prev_sum_pow = 0;
        for (int z = -radius; z <= radius; z++) {
            int index = col_off[z];
            prev_sum += col_val[index];
            prev_sum_pow += col_pow[index];
        }
        for (int x = 0; x < width; x++,scan_in_line++, scan_out_line += channels) {
            int last_row = col_off[x - radius - 1];
            int next_row = col_off[x + radius];
            prev_sum -= col_val[last_row] - col_val[next_row];
            prev_sum_pow = prev_sum_pow - col_pow[last_row] + col_pow[next_row];

            int pix = *scan_in_line;
            int mean = prev_sum / window_size;
            int diff = mean - pix;
            int edge = CLAMP2BYTE(diff);
            int masked_edge = (edge*pix + (256 - edge)*mean) >> 8;
            int var = (prev_sum_pow - mean*prev_sum) / window_size;
            int out = masked_edge - diff*var / (var + smooth_table[pix]);
            scan_out_line[ch_idx] = CLAMP2BYTE(out);
        }
    }

    free(col_pow);
    free(col_val);
    free(row_pos);
    free(col_pos);
}

typedef struct{
    int *smooth_table;
    int *row_pos;
    int *col_pos;
    int roi_x;
    int roi_y;
    int roi_w;
    int roi_h;
} tile_ctx;

void denoise3(
    uint8_t *out,
    uint8_t **planes,
    tile_ctx *tile,
    int width,
    int height,
    int channels,
    int ch_idx,
    int radius
){
    uint8_t *in = planes[ch_idx];

    assert(in && out);
    assert(radius > 0);

    int roi_x = tile->roi_x;
    int roi_y = tile->roi_y;
    int roi_w = tile->roi_w;
    int roi_h = tile->roi_h;
    int *smooth_table = tile->smooth_table;

    int window_size = (2*radius + 1) * (2*radius + 1);

    int *col_pow = calloc(width * sizeof(int), 1);
    int *col_val = calloc(width * sizeof(int), 1);

    int *row_off = tile->row_pos + radius+1;
    int *col_off = tile->col_pos + radius+1;

    int sx = max(0, roi_x - radius - 1);
    int ex = min(width, roi_x + roi_w + radius);
    for (int x = sx; x < ex; x++) {
        for (int z = roi_y - radius - 1; z < roi_y + radius; z++) {
            uint8_t sample = *(in + row_off[z] + x);
            col_val[x] += sample;
            col_pow[x] += sample * sample;
        }
    }
    for (int y = roi_y; y < (roi_y+roi_h); y++) {
        uint8_t *scan_in_line = in + y*width + roi_x;
        uint8_t *scan_out_line = out + (y*width + roi_x)*channels;
        for (int x = sx; x < ex; x++) {
            uint8_t *last_col = in + row_off[y - radius - 1];
            uint8_t *next_col = in + row_off[y + radius];
            col_val[x] -= last_col[x] - next_col[x];
            col_pow[x] -= last_col[x]*last_col[x] - next_col[x]*next_col[x];
        }

        int prev_sum = 0, prev_sum_pow = 0;
        for (int z = roi_x - radius - 1; z < roi_x + radius; z++) {
            int index = col_off[z];
            prev_sum += col_val[index];
            prev_sum_pow += col_pow[index];
        }
#if OPT_VECTOR
        if(roi_x > 0 && (roi_x + roi_w + radius) < width){
            for (int x = roi_x; x < (roi_x+roi_w); x+=16,scan_in_line+=16, scan_out_line += channels*16) {
                int last_col = x - radius - 1;
                int next_col = x + radius;
                s32_16 vcol_val_last = *(s32_16*)(col_val + last_col);
                s32_16 vcol_val_next = *(s32_16*)(col_val + next_col);
                s32_16 vprev_sum = vcol_val_last - vcol_val_next;
                s32_16 vcol_pow_last = *(s32_16*)(col_pow + last_col);
                s32_16 vcol_pow_next = *(s32_16*)(col_pow + next_col);
                s32_16 vprev_sum_pow = vcol_pow_last - vcol_pow_next;
                for(int i = 1; i < 16; i++){
                    vprev_sum[i] += vprev_sum[i-1];
                    vprev_sum_pow[i] += vprev_sum_pow[i-1];
                }
                vprev_sum = prev_sum - vprev_sum;
                vprev_sum_pow = prev_sum_pow - vprev_sum_pow;
                s32_16 vpix = __builtin_convertvector(*(u8_16*)(scan_in_line), s32_16);
                s32_16 vmean = vprev_sum / window_size;
                s32_16 vdiff = vmean - vpix;
                s32_16 vedge = vdiff > 255 ? 255 : (vdiff < 0 ? 0 : vdiff);
                s32_16 vmasked_edge = (vedge*vpix + (256 - vedge)*vmean) >> 8;
                s32_16 vvar = (vprev_sum_pow - vmean*vprev_sum) / window_size;
                s32_16 vsmooth;
                for(int i = 0; i < 16; i++)
                    vsmooth[i] = smooth_table[vpix[i]];
                s32_16 vout_val = vmasked_edge - vdiff*vvar / (vvar + vsmooth);
                vout_val = vout_val > 255 ? 255 : (vout_val < 0 ? 0 : vout_val);
                for(int i = 0; i < 16; i++)
                    *(scan_out_line + i*channels + ch_idx) = vout_val[i];
                prev_sum = vprev_sum[15];
                prev_sum_pow = vprev_sum_pow[15];
            }
        }
        else
#endif
        {
            for (int x = roi_x; x < (roi_x+roi_w); x++,scan_in_line++, scan_out_line += channels) {
                int last_col = col_off[x - radius - 1];
                int next_col = col_off[x + radius];
                prev_sum -= col_val[last_col] - col_val[next_col];
                prev_sum_pow -= col_pow[last_col] - col_pow[next_col];

                int pix = *scan_in_line;
                int mean = prev_sum / window_size;
                int diff = mean - pix;
                int edge = CLAMP2BYTE(diff);
                int masked_edge = (edge*pix + (256 - edge)*mean) >> 8;
                int var = (prev_sum_pow - mean*prev_sum) / window_size;
                int out_val = masked_edge - diff*var / (var + smooth_table[pix]);
                scan_out_line[ch_idx] = CLAMP2BYTE(out_val);
            }
        }
    }

    free(col_pow);
    free(col_val);
}

inline uint64_t time_diff(struct timeval *st, struct timeval *et)
{
    return (et->tv_sec - st->tv_sec)*1000000ULL + (et->tv_usec - st->tv_usec);
}
/* clang-format on */

static void die(char *msg)
{
    fprintf(stderr, "Fatal: %s\n", msg);
    exit(-1);
}

int main(int argc, char *argv[])
{
    if (argc < 2) {
        printf("%s -i INPUT [-o OUTPUT] [-l LEVEL]\n", argv[0]);
        return -1;
    }

    char *ifn = NULL;
    char *ofn = "out.jpg";
    int smoothing_level = 10;

    /* clang-format off */
    int opt;
    while ((opt = getopt (argc, argv, "i:l:o:")) != -1){
        switch(opt){
            case 'i': {
                ifn = optarg;
                break;
            }
            case 'l': {
                smoothing_level = atoi(optarg);
                smoothing_level = (smoothing_level < 1 ? 1 : (smoothing_level > 20 ? 20 : smoothing_level));
                break;
            }
            case 'o': {
                ofn = optarg;
                break;
            }
        }
    }
    printf("ifn:%s ofn:%s level:%d\n", ifn, ofn, smoothing_level);
    /* clang-format on */

    struct timeval stime, etime;

    int width = 0, height = 0, channels = 0;
    uint8_t *in = stbi_load(ifn, &width, &height, &channels, 0);
    if (!in)
        die("Fail to load input file");
    assert(width > 0 && height > 0);
    assert(channels >= 3);

    int dimension = width * height;
    uint8_t *out = malloc(dimension * channels);
    if (!out)
        die("Out of memory");

    uint8_t *in_planes[4] = {NULL};
    for (int i = 0; i < channels; i++)
        in_planes[i] = malloc(dimension);

    /* Separation between skin and non-skin pixels */
    gettimeofday(&stime, NULL);
    float rate = detect(in, in_planes, width, height, channels) /
                 (float) dimension * 100;
    gettimeofday(&etime, NULL);
    printf("detect - %lu us\n", time_diff(&stime, &etime));

    /* Perform edge detection, resulting in an edge map for further denoise */
    /* clang-format off */
    gettimeofday(&stime, NULL);
    int smooth_table[256] = {0};
    float ii = 0.f;
    for (int i = 0; i <= 255; i++, ii -= 1.) {
        smooth_table[i] = (
            expf(ii * (1.0f / (smoothing_level * 255.0f))) +
            (smoothing_level * (i + 1)) + 1
        ) / 2;
        smooth_table[i] = max(smooth_table[i], 1);
    }
#if OPT_PLANE

#if TILING
    int radius = min(width, height)/rate + 1;
    int *row_pos = malloc((height + 2*radius + 2) * sizeof(int));
    int *col_pos = malloc((width + 2*radius + 2) * sizeof(int));

    compute_offset(row_pos, height, radius+1, radius+1, width);
    compute_offset(col_pos, width, radius+1, radius+1, 1);

    #define TILE_W 240
    #define TILE_H 240
    for(int i = 0; i < channels; i++){
        #if 1

        int tx = (width + (TILE_W-1))/TILE_W;
        int ty = (height + (TILE_H-1))/TILE_H;
        #pragma omp parallel for
        for(int idx = 0; idx < tx*ty; idx++){
            int x = (idx % tx) * TILE_W;
            int y = (idx / tx) * TILE_H;
            if(y + TILE_H > height)
                y = height - TILE_H;
            if(x + TILE_W >= width)

            x = width - TILE_W;
            tile_ctx tile;
            tile.roi_x = x;
            tile.roi_y = y;
            tile.roi_w = TILE_W;
            tile.roi_h = TILE_H;
            tile.smooth_table = smooth_table;
            tile.row_pos = row_pos;
            tile.col_pos = col_pos;

            denoise3(out, in_planes, &tile, width, height, channels, i, min(width, height)/rate + 1);
        }

        #else
        for(int _y = 0; _y < (height + (TILE_H-1))/TILE_H; _y++){
            int y = _y * TILE_H;
            if(y + TILE_H > height)
                y = height - TILE_H;

            #pragma omp parallel for
            for(int _x = 0; _x < (width + (TILE_W-1))/TILE_W; _x++){
                int x = _x * TILE_W;
                if(x + TILE_W >= width)
                    x = width - TILE_W;
                tile_ctx tile;
                tile.roi_x = x;
                tile.roi_y = y;
                tile.roi_w = TILE_W;
                tile.roi_h = TILE_H;
                tile.smooth_table = smooth_table;
                tile.row_pos = row_pos;
                tile.col_pos = col_pos;

                denoise3(out, in_planes, &tile, width, height, channels, i, min(width, height)/rate + 1);

            }
        }
        #endif
    }

    free(row_pos);
    free(col_pos);
#else
    #pragma omp parallel for
    for(int i = 0; i < channels; i++)
        denoise2(out, in_planes, smooth_table, width, height, channels, i, min(width, height)/rate + 1);
#endif // TILING

#else
    denoise(out, in, smooth_table, width, height, channels, min(width, height) / rate + 1);
#endif
    gettimeofday(&etime, NULL);
    printf("denoise - %lu us\n", time_diff(&stime, &etime));
    /* clang-format on */

    if (!stbi_write_jpg(ofn, width, height, channels, out, 100))
        die("Fail to generate");

    for (int i = 0; i < channels; i++)
        free(in_planes[i]);

    free(out);
    free(in);
    return 0;
}
