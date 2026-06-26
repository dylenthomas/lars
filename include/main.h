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
#define MIC1_NAME "plughw:CARD=Snowball,DEV=0"
#define MIC2_NAME "plughw:CARD=Snowball_1,DEV=0"
#define MIC3_NAME "plughw:CARD=Device,DEV=0"
#define MIC4_NAME "plughw:CARD=Device_1,DEV=0"
#define MIC5_NAME "plughw:CARD=Device_2,DEV=0"

// VAD tensor dimensions
#define STATE_LEN (2 * 1 * 128)
#define SAMPLE_RATE_DIMS 1
#define INPUT_DIMS 2
#define STATE_DIMS 3

// ORT Required GPU hardware information (specific to my desktop hardware)
#define VENDOR_ID 4318
#define DEVICE_ID 0
#define GPU_ALIGNMENT 0

#define NUM_THREADS 7 // First (NUM_MICS) threads are reserved for the mics
#define NUM_MICS 5
#define NODE1_THREAD_ID (NUM_THREADS - 2)
#define TRANSCRIPT_THREAD_ID (NUM_THREADS - 1)

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
static const char* denoiser_model_path = "/home/dylenthomas/LiveASRonRPi-4/.models/dpdfnet8.onnx";
static const char* keyword_config_path = "";
const char* mic_names[NUM_MICS] = {MIC1_NAME, MIC2_NAME, MIC3_NAME, MIC4_NAME, MIC5_NAME};

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
    int data_ready;
    pthread_mutex_t* mutex;
    pthread_cond_t* cond;
};

struct node_thread_data {
    struct mic_thread_data** mics; // The mic in the first index is the origin of the node
    float* buffer;
    int data_ready;
    int num_mics;
    pthread_mutex_t* mutex;
    pthread_cond_t* cond;
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
static atomic_int run_node_threads = 0;

/**
 * Check status of any ORT function
 *
 * @param status Status pointer returned by ORT functino
 * @param ort ORT Api pointer
 * @return good or bad status (1 = bad status and function failed, 0 = good status)
 */
static int badStatus(OrtStatus* status, const OrtApi* ort);

/**
 * Initialize the ORT environment and necessary tensors
 *
 * @return ORT Api pointer
 */
static const OrtApi* initializeORT();

/**
 * Get the speech probability form the VAD model outputs
 *
 * @param outputs Outputs from the VAD model
 * @param ort ORT Api pointer
 * @return probability of speech (if < 0.0f then something went wrong)
 */
static float getSpeechProb(OrtValue*** outputs, const OrtApi* ort);

/**
 * Mic thread function that persistently runs throughout program life
 *
 * @param ptr thread struct containing all necessary information
 * @return NULL
 */
static void* readMicData(void* ptr);

/**
 * Unlock all mutexes for a given array of microphones
 *
 * @param mics array containing pointers to structs for all microphones
 * @param num_mics the number of mics to be unlocked
 */
static void unlockMicMutexes(struct mic_thread_data* mics[], int num_mics);

/**
 * Thread to combine an arbitrary number of microphones into a single node
 *
 * @param ptr pointer to struct containing all thread information
 * @return NULL
 */
static void* createNode(void* ptr);

/**
 * Transcribe a given audio buffer into speech
 *
 * @param ptr thread struct containing all necessary information
 * @return NULL
 */
static void* transcribe(void* ptr);

/**
 * Find the location of the maximum value in an array of floats
 *
 * @param values Array of floats
 * @param num_vals The number of values of the array
 * @return The location of the max value
 */
static int max(const float values[], int num_vals);

/**
 * Intiailize the sherpa api
 *
 * @return status of the initialization (1 = good, 0 = bad)
 */
static int initalizeSherpa();

static int findSampleDelay(const float* ref_buffer, const float* other_buffer);

#endif //LARS_MAIN_H
