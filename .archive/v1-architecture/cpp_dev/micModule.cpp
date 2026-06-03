#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <stdio.h>
#include <cmath>
#include <algorithm>

#include <alsa/asoundlib.h>
#include "VADmodel.h"

// Build command:
// g++ -fPIC -shared -o utils/mic1.so cpp_dev/micModule.cpp -I /home/dylenthomas/networkShare/mnemosyne/calliope/LARS/onnxruntime-linux-x64-1.12.1/include/ -L /home/dylenthomas/networkShare/mnemosyne/calliope/LARS/onnxruntime-linux-x64-1.12.1/lib  -lasound -lonnxruntime -Wl,-rpath,/home/dylenthomas/networkShare/mnemosyne/calliope/LARS/onnxruntime-linux-x64-1.12.1/lib

const char model_path[] = "../.model/silero_vad_16k_op15.onnx";
const int sample_rate = 16000;
const int buffer_samples = 512;
const int channels = 1;

static std::unique_ptr<VAD> vad = std::make_unique<VAD>(
    model_path, sample_rate, buffer_samples
);

static snd_pcm_t *capture_handle;

void checkErr(int err, int check_val) {
    if (err < check_val) {
        fprintf(stderr, "error!, %s\n", snd_strerror(err));
        exit(1);
    }
}

void read_mic(std::vector<float>& predict_buffer) {
    int err;
    std::vector<int16_t> read_buffer(buffer_samples * channels, 0);

    if ((err = snd_pcm_readi(capture_handle, read_buffer.data(), buffer_samples)) != buffer_samples) {
        fprintf(stderr, "read from audio interface failed (%s)\n", snd_strerror(err));
        int recovery = snd_pcm_recover(capture_handle, err, 1);
        if (recovery < 0) {
            printf("Failed to recover stream.\n");
            exit(1);
        }
    }

    // high pass filter

    std::transform(read_buffer.begin(), read_buffer.end(), predict_buffer.begin(),
        [](int16_t sample){ return sample / 32768.0f; }    
    );
}

extern "C" {
    void reset_vad() {
        vad.reset();
    }

    /* free the returned audio data buffer in python from memory since python doesn't own that memory */
    void free_buffer(float* buffer) {
        delete[] buffer;
    }

    // Algorithim: 
    //  Once called wait until speech is detected to start adding audio to return buffer. Then once the person stops talking or max length is reached return the buffer. 

    void init_mic(const char* name) {
        int err;
        unsigned int sr = sample_rate;
        snd_pcm_hw_params_t *hw_params;
        
        checkErr(snd_pcm_open(&capture_handle, name, SND_PCM_STREAM_CAPTURE, 0), 0);
        checkErr(snd_pcm_hw_params_malloc(&hw_params), 0);
        checkErr(snd_pcm_hw_params_any(capture_handle, hw_params), 0);
        checkErr(snd_pcm_hw_params_set_access(capture_handle, hw_params, SND_PCM_ACCESS_RW_INTERLEAVED), 0);
        checkErr(snd_pcm_hw_params_set_format(capture_handle, hw_params, SND_PCM_FORMAT_S16_LE), 0);
        checkErr(snd_pcm_hw_params_set_rate_near(capture_handle, hw_params, &sr, 0), 0);
        checkErr(snd_pcm_hw_params_set_channels(capture_handle, hw_params, channels), 0);
        checkErr(snd_pcm_hw_params(capture_handle, hw_params), 0);
        snd_pcm_hw_params_free(hw_params);
        checkErr(snd_pcm_prepare(capture_handle), 0);
    }

    void close_mic() {
        snd_pcm_close(capture_handle);
    }

    float* get_speech(
        int max_audio_length,
        int* collected_samples, 
        float speech_threshold
    ) {
        int iters = std::round((sample_rate * max_audio_length) / buffer_samples);
        int total_samples = buffer_samples * channels * iters;
        int err;

        float speech_prob;
    
        std::vector<float> predict_buffer(buffer_samples * channels, 0.0f);
        /* allocate array on the heap, so they can be transferred to python since they survive after the function ends 
           returning something like short buf_storage[total_smaples] would return a pointer to dead memory */
        float* buf_storage = new float[total_samples];
        bool speech_detected = false;

        while (!speech_detected) {
            read_mic(predict_buffer);

            speech_prob = vad->process(predict_buffer);
            speech_detected = speech_prob >= speech_threshold;
        }

        for (int i = 0; i < iters; i++) {
            read_mic(predict_buffer);

            speech_prob = vad->process(predict_buffer);
            speech_detected = speech_prob >= speech_threshold;

            std::copy(
                predict_buffer.begin(), 
                predict_buffer.end(), 
                buf_storage + (i * buffer_samples * channels)
            );
            
            if (!speech_detected) { break; }
        }

        *collected_samples = total_samples;
        return buf_storage;
    }
}
