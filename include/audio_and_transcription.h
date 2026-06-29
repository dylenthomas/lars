#ifndef AUDIO_AND_TRANSCRIPTION_H
#define AUDIO_AND_TRANSCRIPTION_H

#include <stdatomic.h>
#include <pthread.h>

#include "onnxruntime_c_api.h"
#include "sherpa-onnx/c-api/c-api.h"
#include "mic_access.h"
#include "wav_writer.h"

#define MAX_WAV_TIME 5

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

const char* mic_names[NUM_MICS] = {MIC1_NAME, MIC2_NAME};
snd_pcm_t* mic_chs[NUM_MICS];

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

struct mic_thread_data_struct mic_thread_data[NUM_MICS];
struct transcription_thread_data_struct transcription_thread_data;

atomic_int run_mic_threads = 0;
atomic_int run_transcription_thread = 0;

OrtEnv* ort_env = NULL;
OrtSessionOptions* ort_session_opts = NULL;
OrtCUDAProviderOptionsV2* ort_cuda_opts = NULL;
OrtSession* ort_session = NULL;
OrtRunOptions* ort_run_opts = NULL;
OrtMemoryInfo* ort_gpu_mem_info = NULL;
OrtMemoryInfo* ort_cpu_mem_info = NULL;
OrtAllocator* ort_alloc = NULL;
OrtIoBinding* ort_io_binding = NULL;

const int64_t sample_rate[] = {SAMPLE_RATE};
const int64_t sample_rate_shape[] = {1};
const int64_t input_data_shape[] = {1, MIC_BUFFER_LEN}; // input data will be the mic buffer
const int64_t state_data_shape[] = {2, 1, 128};
float initial_state[STATE_LEN] = {0};

float combined_buffer[MIC_BUFFER_LEN] = {0};
OrtValue* vad_input_tensor = NULL;
OrtValue* vad_state_tensor = NULL;
OrtValue* vad_sr_tensor = NULL;

const char* encoder_filename = "/home/dylenthomas/lars/.models/sherpa-onnx-streaming-zipformer-en-2023-06-21/encoder-epoch-99-avg-1.onnx";
const char* decoder_filename = "/home/dylenthomas/lars/.models/sherpa-onnx-streaming-zipformer-en-2023-06-21/decoder-epoch-99-avg-1.onnx";
const char* joiner_filename = "/home/dylenthomas/lars/.models/sherpa-onnx-streaming-zipformer-en-2023-06-21/joiner-epoch-99-avg-1.onnx";
const char* tokens_filename = "/home/dylenthomas/lars/.models/sherpa-onnx-streaming-zipformer-en-2023-06-21/tokens.txt";
const char* vad_model_path = "/home/dylenthomas/lars/.models/silero_vad_16k_op15.onnx";

SherpaOnnxOnlineRecognizerConfig recognizer_config;
const SherpaOnnxOnlineRecognizer* sherpa_recognizer;
const SherpaOnnxOnlineStream* sherpa_stream;

int initialize_audio_and_transcription();

void* mic_thread(void* ptr);

void* transcription_thread(void* ptr);

static int initializeSherpa();

const OrtApi* initializeORT();

static int bad_ort_status(OrtStatus* status, const OrtApi* ort);

float chance_of_speech(OrtValue*** outputs, const OrtApi* ort);

#endif
