#ifndef WAV_WRITER_H
#define WAV_WRITER_H

#include <stdio.h>
#include <stdint.h>
#include <string.h>

typedef struct {
    FILE* file;
    uint32_t data_bytes_written;
    int sample_rate;
    int channels;
} WavWriter;

// WAV header fields — written at open, then patched at close once total size is known
static int wav_open(WavWriter* w, const char* path, int sample_rate, int channels) {
    w->file = fopen(path, "wb");
    if (!w->file) return -1;
    w->data_bytes_written = 0;
    w->sample_rate = sample_rate;
    w->channels = channels;

    // Write a placeholder header (44 bytes); will be patched on close
    uint8_t header[44] = {0};
    fwrite(header, 1, 44, w->file);
    return 0;
}

static void wav_patch_header(WavWriter* w) {
    uint32_t data_chunk_size = w->data_bytes_written;
    uint32_t riff_chunk_size = 36 + data_chunk_size;
    uint16_t audio_format    = 1;           // PCM
    uint16_t num_channels    = (uint16_t)w->channels;
    uint32_t sample_rate     = (uint32_t)w->sample_rate;
    uint16_t bits_per_sample = 16;
    uint16_t block_align     = num_channels * (bits_per_sample / 8);
    uint32_t byte_rate       = sample_rate * block_align;

    fseek(w->file, 0, SEEK_SET);

    // RIFF chunk
    fwrite("RIFF", 1, 4, w->file);
    fwrite(&riff_chunk_size, 4, 1, w->file);
    fwrite("WAVE", 1, 4, w->file);

    // fmt sub-chunk
    fwrite("fmt ", 1, 4, w->file);
    uint32_t fmt_size = 16;
    fwrite(&fmt_size,        4, 1, w->file);
    fwrite(&audio_format,    2, 1, w->file);
    fwrite(&num_channels,    2, 1, w->file);
    fwrite(&sample_rate,     4, 1, w->file);
    fwrite(&byte_rate,       4, 1, w->file);
    fwrite(&block_align,     2, 1, w->file);
    fwrite(&bits_per_sample, 2, 1, w->file);

    // data sub-chunk header
    fwrite("data", 1, 4, w->file);
    fwrite(&data_chunk_size, 4, 1, w->file);
}

// Write a buffer of floats (assumed -1.0 to 1.0) as 16-bit PCM samples
static void wav_write_float(WavWriter* w, const float* buffer, int num_samples) {
    for (int i = 0; i < num_samples; i++) {
        float clamped = buffer[i];
        if (clamped >  1.0f) clamped =  1.0f;
        if (clamped < -1.0f) clamped = -1.0f;
        int16_t sample = (int16_t)(clamped * 32767.0f);
        fwrite(&sample, sizeof(int16_t), 1, w->file);
        w->data_bytes_written += sizeof(int16_t);
    }
}

static void wav_close(WavWriter* w) {
    if (!w->file) return;
    wav_patch_header(w);
    fclose(w->file);
    w->file = NULL;
}

#endif // WAV_WRITER_H