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

#define NUM_THREADS 3 // First (NUM_MICS) threads are reserved for the mics
#define TRANSCRIPT_THREAD_ID (NUM_THREADS - 1)

// ORT Required GPU hardware information (specific to my desktop hardware)
#define VENDOR_ID 4318
#define DEVICE_ID 0
#define GPU_ALIGNMENT 0

// VAD tensor dimensions
#define STATE_LEN (2 * 1 * 128)
#define SAMPLE_RATE_DIMS 1
#define INPUT_DIMS 2
#define STATE_DIMS 3

// Length of padding to add to the end of ASR
#define TAIL_PADDING_LENGTH 8000

// Mic characteristics
#define MIC_BUFFER_LEN 512
#define SAMPLE_RATE 16000
#define CHANNELS 1
#define NUM_MICS 2
#define MIC1_NAME "plughw:CARD=Snowball,DEV=0"
#define MIC2_NAME "plughw:CARD=Snowball_1,DEV=0"

#define MAX_WAV_TIME 5

#define FLUSH_DELAY_MS 2000 

struct mic_thread_data_struct {
    snd_pcm_t* device;
    pthread_mutex_t* mutex;
    pthread_cond_t* cond;
    int data_ready;
    float* buffer;
    float gain;
};

struct transcription_thread_data_struct {
    const SherpaOnnxOnlineRecognizer* recognizer;
    const SherpaOnnxOnlineStream* stream;
    const SherpaOnnxOnlineSpeechDenoiser* denoiser;
    pthread_mutex_t* mutex;
    pthread_cond_t* cond;
    float audio[MIC_BUFFER_LEN];
    float chance_of_speech;
    int ready;
    int flush;
};

int initialize_mics_and_transcription(void);

void* mic_thread(void* ptr);

void* transcription_thread(void* ptr);

static const OrtApi* initializeORT(void);

static int initializeSherpa(void);

int kbhit(void);

static int max(const float values[], int num_vals);

static int bad_ort_status(OrtStatus* status, const OrtApi* ort);

float chance_of_speech(OrtValue*** outputs, const OrtApi* ort);

static inline int64_t now_ms(void);

// Assignments

const char* keyword_config_path = "";

static OrtEnv* ort_env = NULL;
static OrtSessionOptions* ort_session_opts = NULL;
static OrtCUDAProviderOptionsV2* ort_cuda_opts = NULL;
static OrtSession* ort_session = NULL;
static OrtRunOptions* ort_run_opts = NULL;
static OrtMemoryInfo* ort_gpu_mem_info = NULL; 
static OrtMemoryInfo* ort_cpu_mem_info = NULL;
static OrtAllocator* ort_alloc = NULL;
static OrtIoBinding* ort_io_binding = NULL;

static float combined_buffer[MIC_BUFFER_LEN] = {0};
static OrtValue* vad_input_tensor = NULL;
static OrtValue* vad_state_tensor = NULL;
static OrtValue* vad_sr_tensor = NULL;

static const int64_t sample_rate[] = {SAMPLE_RATE};
static const int64_t sample_rate_shape[] = {1};
static const int64_t input_data_shape[] = {1, MIC_BUFFER_LEN}; // input data will be the mic buffer
static const int64_t state_data_shape[] = {2, 1, 128};
static float initial_state[STATE_LEN] = {0};

static const char* encoder_filename = "/home/dylenthomas/lars/.models/sherpa-onnx-streaming-zipformer-en-2023-06-21/encoder-epoch-99-avg-1.onnx";
static const char* decoder_filename = "/home/dylenthomas/lars/.models/sherpa-onnx-streaming-zipformer-en-2023-06-21/decoder-epoch-99-avg-1.onnx";
static const char* joiner_filename = "/home/dylenthomas/lars/.models/sherpa-onnx-streaming-zipformer-en-2023-06-21/joiner-epoch-99-avg-1.onnx";
static const char* tokens_filename = "/home/dylenthomas/lars/.models/sherpa-onnx-streaming-zipformer-en-2023-06-21/tokens.txt";
static const char* vad_model_path = "/home/dylenthomas/lars/.models/silero_vad_16k_op15.onnx";

SherpaOnnxOnlineRecognizerConfig recognizer_config;
const SherpaOnnxOnlineRecognizer* sherpa_recognizer;
const SherpaOnnxOnlineStream* sherpa_stream;

const char* mic_names[NUM_MICS] = {MIC1_NAME, MIC2_NAME};
snd_pcm_t* mic_chs[NUM_MICS];

struct mic_thread_data_struct mic_thread_data[NUM_MICS];
struct transcription_thread_data_struct transcription_thread_data;

atomic_int run_mic_threads = 0;
atomic_int run_transcription_thread = 0;

pthread_mutex_t transcribe_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t transcribe_cond = PTHREAD_COND_INITIALIZER;

pthread_mutex_t mic1_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t mic1_cond = PTHREAD_COND_INITIALIZER;

pthread_mutex_t mic2_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t mic2_cond = PTHREAD_COND_INITIALIZER;

float mic1_buffer[MIC_BUFFER_LEN] = {0};
float mic2_buffer[MIC_BUFFER_LEN] = {0};

float* mic_buffers[NUM_MICS] = {mic1_buffer, mic2_buffer};
pthread_mutex_t* mic_mutexes[NUM_MICS] = {&mic1_mutex, &mic2_mutex};
pthread_cond_t* mic_conds[NUM_MICS] = {&mic1_cond, &mic2_cond};
float mic_gains[NUM_MICS] = {3.0f, 5.0f};

#endif //LARS_MAIN_H
