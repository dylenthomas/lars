#ifndef MICACCESS_H_
#define MIC_ACCESS_H_

#include <alsa/asoundlib.h>

void read_mic(int16_t* buffer, snd_pcm_t* capture_handle, int buffer_samples);
void init_mic(const char* name, snd_pcm_t** capture_handle, int sample_rate, int channels);
void close_mic(snd_pcm_t *capture_handle);

#endif
