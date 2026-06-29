#ifndef LARS_MAIN_H
#define LARS_MAIN_H

#include <stdio.h>
#include <pthread.h>
#include <stdatomic.h>

#include "onnxruntime_c_api.h"
#include "sherpa-onnx/c-api/c-api.h"
#include "mic_access.h"
#include "fft.h"
#include "wav_writer.h"
#include "audio_and_transcription.h"

#define NUM_THREADS 3 // First (NUM_MICS) threads are reserved for the mics
#define TRANSCRIPT_THREAD_ID (NUM_THREADS - 1)

const char* keyword_config_path = "";

int kbhit(void);

static int max(const float values[], int num_vals);

#endif //LARS_MAIN_H
