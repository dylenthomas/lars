//
// Created by dylenthomas on 2026-06-03.
//

#ifndef LARS_MAIN_H
#define LARS_MAIN_H

#include <stdio.h>
#include <pthread.h>
#include <stdatomic.h>

#include "onnxruntime_c_api.h"
#include "mic_access.h"
#include "sherpa-onnx/c-api/c-api.h"

// Mic characteristics
#define MIC_BUFFER_LEN 512
#define SAMPLE_RATE 16000
#define CHANNELS 1

// VAD tensor dimensions
#define STATE_LEN (2 * 1 * 128)
#define SAMPLE_RATE_DIMS 1
#define INPUT_DIMS 2
#define STATE_DIMS 3

// ORT Required GPU hardware information
#define VENDOR_ID 4318
#define DEVICE_ID 0
#define GPU_ALIGNMENT 0

#define NUM_THREADS 3

// Length of padding to add to the end of ASR
#define TAIL_PADDING_LENGTH 8000

static OrtEnv* ort_env = NULL;
static OrtSessionOptions* ort_session_opts = NULL;
static OrtCUDAProviderOptionsV2* ort_cuda_opts = NULL;
static OrtSession* ort_session = NULL;
static OrtRunOptions* ort_run_opts = NULL;
static OrtMemoryInfo* ort_gpu_mem_info = NULL;
static OrtMemoryInfo* ort_cpu_mem_info = NULL;
static OrtAllocator* ort_alloc = NULL;
static OrtIoBinding* ort_io_binding = NULL;

static const int64_t sample_rate[] = {SAMPLE_RATE};
static const int64_t sample_rate_shape[] = {1};
static const int64_t input_data_shape[] = {1, MIC_BUFFER_LEN}; // input data will be the mic buffer
static const int64_t state_data_shape[] = {2, 1, 128};
static float initial_state[STATE_LEN] = {0};

static float combined_buffer[MIC_BUFFER_LEN] = {0};
static OrtValue* vad_input_tensor = NULL;
static OrtValue* vad_state_tensor = NULL;
static OrtValue* vad_sr_tensor = NULL;

static const char* encoder_filename = "/home/dylenthomas/LiveASRonRPi-4/.models/sherpa-onnx-streaming-zipformer-en-2023-06-21/encoder-epoch-99-avg-1.onnx";
static const char* decoder_filename = "/home/dylenthomas/LiveASRonRPi-4/.models/sherpa-onnx-streaming-zipformer-en-2023-06-21/decoder-epoch-99-avg-1.onnx";
static const char* joiner_filename = "/home/dylenthomas/LiveASRonRPi-4/.models/sherpa-onnx-streaming-zipformer-en-2023-06-21/joiner-epoch-99-avg-1.onnx";
static const char* tokens_filename = "/home/dylenthomas/LiveASRonRPi-4/.models/sherpa-onnx-streaming-zipformer-en-2023-06-21/tokens.txt";
static const char* vad_model_path = "/home/dylenthomas/LiveASRonRPi-4/.models/silero_vad_16k_op15.onnx";
static const char* keyword_config_path = "";
static const char* mic1_name = "plughw:CARD=Snowball,DEV=0";
static const char* mic2_name = "plughw:CARD=Snowball_1,DEV=0";

static volatile int keep_running = 1;

SherpaOnnxOnlineRecognizerConfig recognizer_config;
const SherpaOnnxOnlineRecognizer* sherpa_recognizer;
const SherpaOnnxOnlineStream* sherpa_stream;
SherpaOnnxOnlineSpeechDenoiserConfig denoiser_config;
const SherpaOnnxOnlineSpeechDenoiser* sd;

void intHandler(int dummy);

struct mic_thread_data {
    float* buffer;
    snd_pcm_t* device;
    long* updated;
    pthread_mutex_t* mutex;
};

struct transcribe_thread_data {
    const SherpaOnnxOnlineRecognizer* recognizer;
    const SherpaOnnxOnlineStream* stream;
    const SherpaOnnxOnlineSpeechDenoiser* denoiser;

    int ready;
    float audio[MIC_BUFFER_LEN];
    int flush;
    float chance_of_speech;

    pthread_mutex_t* mutex;
    pthread_cond_t* cond;
};

static atomic_int run_mic_threads = 0;

static int badStatus(OrtStatus* status, const OrtApi* ort);

static const OrtApi* initializeORT();

static float getSpeechProb(OrtValue*** outputs, const OrtApi* ort);

static void* readMicData(void* ptr);

static void* transcribe(void* ptr);

static int initalizeSherpa();

#endif //LARS_MAIN_H
