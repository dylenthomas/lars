#ifndef WAV_WRITER_H
#define WAV_WRITER_H

#include <stdint.h>
#include <stdlib.h>

#pragma pack(push, 1)
typedef struct {
    char     riff_id[4];
    uint32_t riff_size;
    char     wave_id[4];
    char     fmt_id[4];
    uint32_t fmt_size;
    uint16_t audio_format;
    uint16_t num_channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample;
    char     data_id[4];
    uint32_t data_size;
} wav_header_t;
#pragma pack(pop)

int write_wav_file(const char* filename, const float* samples, int num_samples, int sample_rate);
void make_wav_filename(char* out, size_t out_size);

#endif
