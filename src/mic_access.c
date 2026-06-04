#include "mic_access.h"

#include <alsa/asoundlib.h>

void checkErr(int err) {
	if (err < 0) {
		fprintf(stderr, "Error!, %s\n", snd_strerror(err));
		exit(1);
	}
}

void read_mic(int16_t* buffer, snd_pcm_t* capture_handle, int buffer_samples) {
	int err = snd_pcm_readi(capture_handle, buffer, buffer_samples);
	checkErr(err);
	if (err != buffer_samples) {
		fprintf(stderr, "Read from audio interface failed! (%s)\n", snd_strerror(err));
		checkErr(snd_pcm_recover(capture_handle, err, 1));
	}
}

void init_mic(const char* name, snd_pcm_t** capture_handle, int sample_rate, int channels) {
	unsigned int usr = sample_rate;
	snd_pcm_hw_params_t* hw_params;
	
	checkErr(snd_pcm_open(capture_handle, name, SND_PCM_STREAM_CAPTURE, 0));
    checkErr(snd_pcm_hw_params_malloc(&hw_params));
    checkErr(snd_pcm_hw_params_any(*capture_handle, hw_params));
    checkErr(snd_pcm_hw_params_set_access(*capture_handle, hw_params, SND_PCM_ACCESS_RW_INTERLEAVED));
    checkErr(snd_pcm_hw_params_set_format(*capture_handle, hw_params, SND_PCM_FORMAT_S16_LE));
    checkErr(snd_pcm_hw_params_set_rate_near(*capture_handle, hw_params, &usr, 0));
    checkErr(snd_pcm_hw_params_set_channels(*capture_handle, hw_params, channels));
    checkErr(snd_pcm_hw_params(*capture_handle, hw_params));
    snd_pcm_hw_params_free(hw_params);
    checkErr(snd_pcm_prepare(*capture_handle));
}

void close_mic(snd_pcm_t* capture_handle) {
	snd_pcm_close(capture_handle);
}
