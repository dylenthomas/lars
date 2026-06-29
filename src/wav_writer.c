#include "wav_writer.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdio.h>

int write_wav_file(const char* filename, const float* samples, int num_samples, int sample_rate) {
    FILE* f = fopen(filename, "wb");
    if (!f) {
        fprintf(stderr, "Failed to open %s for writing\n", filename);
        return 1;
    }

    int16_t* pcm = malloc(num_samples * sizeof(int16_t));
    if (!pcm) {
        fclose(f);
        return 1;
    }

    for (int i = 0; i < num_samples; i++) {
        float s = samples[i];
        if (s > 1.0f) s = 1.0f;
        else if (s < -1.0f) s = -1.0f;
        pcm[i] = (int16_t)(s * 32767.0f);
    }

    wav_header_t header;
    memcpy(header.riff_id, "RIFF", 4);
    memcpy(header.wave_id, "WAVE", 4);
    memcpy(header.fmt_id, "fmt ", 4);
    memcpy(header.data_id, "data", 4);

    header.fmt_size = 16;
    header.audio_format = 1;
    header.num_channels = 1;
    header.sample_rate = sample_rate;
    header.bits_per_sample = 16;
    header.block_align = header.num_channels * header.bits_per_sample / 8;
    header.byte_rate = header.sample_rate * header.block_align;
    header.data_size = num_samples * sizeof(int16_t);
    header.riff_size = 36 + header.data_size;

    fwrite(&header, sizeof(header), 1, f);
    fwrite(pcm, sizeof(int16_t), num_samples, f);

    free(pcm);
    fclose(f);
    return 0;
}

void make_wav_filename(char* out, size_t out_size) {
    static int counter = 0;
    time_t now = time(NULL);
    struct tm* t = localtime(&now);
    snprintf(out, out_size, "wav_out/audio_%04d%02d%02d_%02d%02d%02d_%03d.wav",
            t->tm_year + 1900, t->tm_mon + 1, t->tm_mday,
            t->tm_hour, t->tm_min, t->tm_sec, counter++);
}
