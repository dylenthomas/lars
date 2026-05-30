#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <time.h>
#include <signal.h>
#include <stdatomic.h>

#include "onnxruntime_c_api.h"
#include "mic_access.h"
#include "config_parser.h"
#include "sherpa-onnx/c-api/c-api.h"

#define KEYWORD_CONF "configs/keywords.json"
#define MIC_BUFFER_LEN 512 
#define SAMPLE_RATE 16000
#define CHANNELS 1
#define LONGEST_TRANSCRIPT 30

#define STATE_LEN 2 * 1 * 128
#define SAMPLE_RATE_DIMS 1
#define INPUT_DIMS 2
#define STATE_DIMS 3

#define VENDOR_ID 4318
#define DEVICE_ID 0
#define GPU_ALIGNMENT 0

#define NUM_THREADS 3

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
static const char* mic1_name = "plughw:CARD=Snowball,DEV=0";
static const char* mic2_name = "plughw:CARD=Snowball_1,DEV=0";

static volatile int keep_running = 1;
void intHandler(int dummy) {
    keep_running = 0;
}

struct mic_thread_data {
    float* buffer;
    snd_pcm_t* device;
    atomic_long* updated;
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

static int badStatus(OrtStatus* status, const OrtApi* ort) {
	// Make sure the API was accessed correctly 
	if (status != NULL) {
		const char* err_msg = ort->GetErrorMessage(status);
		fprintf(stderr, "ORT Error: %s\n", err_msg);
		ort->ReleaseStatus(status);
		return 1;
	}
	return 0;
}

static const OrtApi* initializeORT() {
    printf("Initializing ORT...\n");
	const OrtApi* ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
	if (ort == NULL) { fprintf(stderr, "ORT api returned nullptr!\n"); return NULL; }

	if (badStatus(ort->CreateEnv(ORT_LOGGING_LEVEL_VERBOSE, "onnxruntime", &ort_env), ort)) { goto ort_env_fail; }

	if (badStatus(ort->CreateSessionOptions(&ort_session_opts), ort)) { goto ort_session_opts_fail; }
	if (badStatus(ort->SetIntraOpNumThreads(ort_session_opts, 1), ort)) { goto ort_session_opts_fail; }
	if (badStatus(ort->SetInterOpNumThreads(ort_session_opts, 1), ort)) { goto ort_session_opts_fail; }
	if (badStatus(ort->SetSessionGraphOptimizationLevel(ort_session_opts, ORT_ENABLE_ALL), ort)) { goto ort_session_opts_fail; }

    if (badStatus(ort->CreateCUDAProviderOptions(&ort_cuda_opts), ort)) { goto ort_cuda_opts_fail; }
    if (badStatus(ort->SessionOptionsAppendExecutionProvider_CUDA_V2(ort_session_opts, ort_cuda_opts), ort)) { goto ort_cuda_opts_fail; }

	if (badStatus(ort->CreateSession(ort_env, vad_model_path, ort_session_opts, &ort_session), ort)) { goto ort_session_fail; }

	if (badStatus(ort->CreateRunOptions(&ort_run_opts), ort)) { goto ort_session_fail; }

    if (badStatus(ort->CreateMemoryInfo_V2("Cuda", OrtMemoryInfoDeviceType_GPU,
        VENDOR_ID, DEVICE_ID, OrtMemTypeDefault, GPU_ALIGNMENT,
        OrtDeviceAllocator, &ort_gpu_mem_info), ort)) { goto ort_gpu_mem_info_fail; }

	if (badStatus(ort->CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeDefault,
	    &ort_cpu_mem_info), ort)) { goto ort_cpu_mem_info_fail; }

	if (badStatus(ort->GetAllocatorWithDefaultOptions(&ort_alloc), ort)) { goto ort_cpu_mem_info_fail; }

	if (badStatus(ort->CreateTensorWithDataAsOrtValue(
		ort_gpu_mem_info, sample_rate, sizeof(sample_rate), sample_rate_shape, SAMPLE_RATE_DIMS,
		ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &vad_sr_tensor), ort)) { goto ort_sr_tensor_fail; }

	// create initializing state for model of all zeros
	if (badStatus(ort->CreateTensorWithDataAsOrtValue(
		ort_gpu_mem_info, initial_state, sizeof(initial_state), state_data_shape, STATE_DIMS,
		ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &vad_state_tensor), ort)) { goto ort_state_tensor_fail; }

	if (badStatus(ort->CreateTensorWithDataAsOrtValue(
		ort_gpu_mem_info, combined_buffer, sizeof(combined_buffer), input_data_shape, INPUT_DIMS,
		ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &vad_input_tensor), ort)) { goto ort_input_tensor_fail; }

    if (badStatus(ort->CreateIoBinding(ort_session, &ort_io_binding), ort)) { goto ort_io_binding_fail; }

    if (badStatus(ort->BindInput(ort_io_binding, "input", vad_input_tensor), ort)) { goto ort_io_binding_fail; }
    if (badStatus(ort->BindInput(ort_io_binding, "state", vad_state_tensor), ort)) { goto ort_io_binding_fail; }
    if (badStatus(ort->BindInput(ort_io_binding, "sr", vad_sr_tensor), ort)) { goto ort_io_binding_fail; }
    if (badStatus(ort->BindOutputToDevice(ort_io_binding, "output", ort_cpu_mem_info), ort)) { goto ort_io_binding_fail; }
    if (badStatus(ort->BindOutputToDevice(ort_io_binding, "stateN", ort_cpu_mem_info), ort)) { goto ort_io_binding_fail; }

    return ort;

    ort_io_binding_fail:
        ort->ReleaseIoBinding(ort_io_binding);
    ort_input_tensor_fail:
        ort->ReleaseValue(vad_input_tensor);
    ort_state_tensor_fail:
        ort->ReleaseValue(vad_state_tensor);
    ort_sr_tensor_fail:
        ort->ReleaseValue(vad_sr_tensor);
    ort_cpu_mem_info_fail:
        ort->ReleaseMemoryInfo(ort_cpu_mem_info);
    ort_gpu_mem_info_fail:
        ort->ReleaseMemoryInfo(ort_gpu_mem_info);
    ort_session_fail:
        ort->ReleaseSession(ort_session);
    ort_cuda_opts_fail:
        ort->ReleaseCUDAProviderOptions(ort_cuda_opts);
    ort_session_opts_fail:
        ort->ReleaseSessionOptions(ort_session_opts);
    ort_env_fail:
        ort->ReleaseEnv(ort_env);
    printf("Released all ORT bindings.\n");

    return NULL;
}

static float getSpeechProb(OrtValue*** outputs, const OrtApi* ort) {
    size_t output_count = 2;
    if (badStatus(ort->RunWithBinding(ort_session, ort_run_opts, ort_io_binding), ort)) { return -2.0f; }
    if (badStatus(ort->GetBoundOutputValues(ort_io_binding, ort_alloc, outputs, &output_count), ort)) { return -2.0f; }
	
	// Retrieve probability of speech from the model
	OrtValue* ort_speech_prob = (*outputs)[0];
	
	float* prob_data = NULL;
	if (badStatus(ort->GetTensorMutableData(ort_speech_prob, (void**)&prob_data), ort)) { return -1.0f; }
    return prob_data[0];
}

static void* readMicData(void* ptr) {
    const struct mic_thread_data* args = (struct mic_thread_data*)ptr;

    while (run_mic_threads) {
        int16_t tmp_buffer[MIC_BUFFER_LEN] = {0};
	    int i = 0;

        read_mic(tmp_buffer, args->device, MIC_BUFFER_LEN);
        // convert int mic data to float	
        pthread_mutex_lock(args->mutex);
	    while (i < MIC_BUFFER_LEN) { args->buffer[i] = (float)tmp_buffer[i] / 32768.0f; i++; }
        pthread_mutex_unlock(args->mutex);

        // TODO: Replace with mutex
        // avoid running prediction on old data
        atomic_store(args->updated, random());
    }

    return NULL;
}

static void* transcribe(void* ptr) {
    struct transcribe_thread_data* args = ptr;
    const SherpaOnnxOnlineRecognizer* recognizer = args->recognizer;
    const SherpaOnnxOnlineStream* stream = args->stream;
    const SherpaOnnxOnlineSpeechDenoiser* denoiser = args->denoiser;

    while (keep_running) {
        float audio[MIC_BUFFER_LEN] = {0};
        int flush = 0;
        float chance_of_speech = 0.0;

        pthread_mutex_lock(args->mutex);
        while (!args->ready) {
            pthread_cond_wait(args->cond, args->mutex); // sleep until main thread requests prediction
        }
        args->ready = 0;
        memcpy(audio, args->audio, MIC_BUFFER_LEN * sizeof(float));
        flush = args->flush;
        chance_of_speech = args->chance_of_speech;
        pthread_mutex_unlock(args->mutex);

        if (!flush) {
            const SherpaOnnxDenoisedAudio* chunk = SherpaOnnxOnlineSpeechDenoiserRun(denoiser, audio, MIC_BUFFER_LEN, SAMPLE_RATE);

            if (chunk->sample_rate <= 48000 && chunk != NULL) {
                //SherpaOnnxOnlineStreamAcceptWaveform(stream, SAMPLE_RATE, audio, MIC_BUFFER_LEN);
                SherpaOnnxOnlineStreamAcceptWaveform(stream, chunk->sample_rate, chunk->samples, chunk->n);
                while (SherpaOnnxIsOnlineStreamReady(recognizer, stream)) {
                    SherpaOnnxDecodeOnlineStream(recognizer, stream);
                }

                const SherpaOnnxOnlineRecognizerResult* r = SherpaOnnxGetOnlineStreamResult(recognizer, stream);
                printf("\033[1A");
                printf("\033[2K\r[%.2f] %s    \n", chance_of_speech, r->text);
                fflush(stdout);

                if (SherpaOnnxOnlineStreamIsEndpoint(recognizer, stream)) {
                    SherpaOnnxOnlineStreamReset(recognizer, stream);
                }

                SherpaOnnxDestroyOnlineRecognizerResult(r);
            }
            SherpaOnnxDestroyDenoisedAudio(chunk);
        } else {
            const SherpaOnnxDenoisedAudio* tail = SherpaOnnxOnlineSpeechDenoiserFlush(denoiser);
            const float tail_padding[TAIL_PADDING_LENGTH] = {0};

            if (tail) { SherpaOnnxOnlineStreamAcceptWaveform(stream, SAMPLE_RATE, tail->samples, tail->n); }

            SherpaOnnxOnlineStreamAcceptWaveform(stream, SAMPLE_RATE, tail_padding, TAIL_PADDING_LENGTH);
            SherpaOnnxOnlineStreamInputFinished(stream);
            while (SherpaOnnxIsOnlineStreamReady(recognizer, stream)) {
                SherpaOnnxDecodeOnlineStream(recognizer, stream);
            }

            const SherpaOnnxOnlineRecognizerResult* r = SherpaOnnxGetOnlineStreamResult(recognizer, stream);
            printf("\033[1A");
            printf("\033[2K\r[%.2f] %s    \n", chance_of_speech, r->text);
            fflush(stdout);

            SherpaOnnxDestroyOnlineRecognizerResult(r);
            SherpaOnnxOnlineStreamReset(recognizer, stream);
            SherpaOnnxDestroyDenoisedAudio(tail);
        }
    }

    return NULL;
}

int main(int argc, char *argv[]) {
    signal(SIGINT, intHandler);
    srandom(time(NULL));

    SherpaOnnxOnlineRecognizerConfig recognizer_config;
    memset(&recognizer_config, 0, sizeof(recognizer_config));
    recognizer_config.decoding_method = "greedy_search";
    recognizer_config.model_config.debug = 0;
    recognizer_config.model_config.num_threads = 4;
    recognizer_config.model_config.provider = "cuda";
    recognizer_config.model_config.tokens = tokens_filename;
    recognizer_config.model_config.transducer.encoder = encoder_filename;
    recognizer_config.model_config.transducer.decoder = decoder_filename;
    recognizer_config.model_config.transducer.joiner = joiner_filename;
    recognizer_config.enable_endpoint = 1;

    SherpaOnnxOnlineSpeechDenoiserConfig denoiser_config;
    memset(&denoiser_config, 0, sizeof(denoiser_config));
    denoiser_config.model.dpdfnet.model = "/home/dylenthomas/LiveASRonRPi-4/.models/dpdfnet_baseline.onnx";
    denoiser_config.model.num_threads = 2;
    denoiser_config.model.debug = 0;
    denoiser_config.model.provider = "cpu";

    const SherpaOnnxOnlineRecognizer* sherpa_recognizer = SherpaOnnxCreateOnlineRecognizer(&recognizer_config);
    if (sherpa_recognizer == NULL) {
        fprintf(stderr, "Please check your config!\n");
        goto sherpa_recognizer_fail;
    }
    const SherpaOnnxOnlineStream* sherpa_stream = SherpaOnnxCreateOnlineStream(sherpa_recognizer);

    const SherpaOnnxOnlineSpeechDenoiser* sd = SherpaOnnxCreateOnlineSpeechDenoiser(&denoiser_config);
    if (sd == NULL) {
        fprintf(stderr, "Failed to create speech denoiser\n");
        goto sherpa_denoiser_fail;
    }

    struct transcribe_thread_data transcribe_data;
    pthread_mutex_t transcribe_mutex = PTHREAD_MUTEX_INITIALIZER;
    pthread_cond_t transcribe_cond = PTHREAD_COND_INITIALIZER;

    transcribe_data.recognizer = sherpa_recognizer;
    transcribe_data.stream = sherpa_stream;
    transcribe_data.denoiser = sd;
    transcribe_data.cond = &transcribe_cond;
    transcribe_data.mutex = &transcribe_mutex;
    transcribe_data.ready = 0;

    //struct keywordHM keywords = createKeywordHM(KEYWORD_CONF);

    double peak_value = 0.0;
    int iterations_held = 0;
    int iterations_decayed = 0;

    atomic_long updated1 = ATOMIC_VAR_INIT(0);
    atomic_long updated2 = ATOMIC_VAR_INIT(0);
    float mic1_buffer[MIC_BUFFER_LEN] = {0};
    float mic2_buffer[MIC_BUFFER_LEN] = {0};

    int ret;
    long int mic1_last_updated;
    long int mic2_last_updated;

    int vad_previously_triggered = 0;
// Initialize the ORT Api --------------------------------------------------------------------------
    const OrtApi* ort = initializeORT();
    if (ort == NULL) { goto ort_initialize_fail; }
// Initialize Microphone ---------------------------------------------------------------------------
	printf("Initializing microphones...\n");

	snd_pcm_t* mic1_ch;
	init_mic(mic1_name, &mic1_ch, SAMPLE_RATE, CHANNELS, MIC_BUFFER_LEN);

    snd_pcm_t* mic2_ch;
    init_mic(mic2_name, &mic2_ch, SAMPLE_RATE, CHANNELS, MIC_BUFFER_LEN);

    struct mic_thread_data mic_data[2];

    pthread_mutex_t mic_mutex1 = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_t mic_mutex2 = PTHREAD_MUTEX_INITIALIZER;

    mic_data[0].buffer = mic1_buffer;
    mic_data[0].device = mic1_ch;
    mic_data[0].updated = &updated1;
    mic_data[0].mutex = &mic_mutex1;
    mic_data[1].buffer = mic2_buffer;
    mic_data[1].device = mic2_ch;
    mic_data[1].updated = &updated2;
    mic_data[1].mutex = &mic_mutex2;
// Initialize threads ------------------------------------------------------------------------------	
    pthread_t thread[NUM_THREADS];
    
    printf("Starting audio threads.\n");
    run_mic_threads = 1;
    
    if ((ret = pthread_create(&thread[0], NULL, readMicData, &mic_data[0])) != 0) { 
        fprintf(stderr, "Error: failed to create first mic thread %d", ret);
        goto first_mic_thread_fail;
    }
    printf("Started mic1 thread.\n");

    if ((ret = pthread_create(&thread[1], NULL, readMicData, &mic_data[1])) != 0) { 
        fprintf(stderr, "Error: failed to create second mic thread %d", ret);
        goto second_mic_thread_fail;
    }
    printf("Started mic2 thread.\n");

    if ((ret = pthread_create(&thread[2], NULL, transcribe, &transcribe_data)) != 0) {
        fprintf(stderr, "Error: failed to create transcribe thread %d", ret);
        goto second_mic_thread_fail;
    }

    mic1_last_updated = *mic_data[0].updated;
    mic2_last_updated = *mic_data[1].updated;
// -------------------------------------------------------------------------------------------------
    while (keep_running) {
        int hold_iterations = 5;
        const double speech_threshold = 0.7; // trigger threshold to start transcription

        // wait until both buffers are populated
        if (mic1_last_updated == *mic_data[0].updated) { continue; }
        mic1_last_updated = *mic_data[0].updated;

        if (mic2_last_updated == *mic_data[1].updated) { continue; }
        mic2_last_updated = *mic_data[1].updated;

        pthread_mutex_lock(mic_data[0].mutex);
        pthread_mutex_lock(mic_data[1].mutex);

        float rms1 = 0, rms2 = 0;
        for (int i = 0; i < MIC_BUFFER_LEN; i++) {
            rms1 += mic_data[0].buffer[i] * mic_data[0].buffer[i];
            rms2 += mic_data[1].buffer[i] * mic_data[1].buffer[i];
        }
        memcpy(combined_buffer, rms1 > rms2 ? mic_data[0].buffer : mic_data[1].buffer, MIC_BUFFER_LEN * sizeof(float));

        pthread_mutex_unlock(mic_data[0].mutex);
        pthread_mutex_unlock(mic_data[1].mutex);

		float speech_prob;
        OrtValue** outputs = NULL;

		speech_prob = getSpeechProb(&outputs, ort);
        if (speech_prob == -2.0f) { continue; }
        if (speech_prob == -1.0f) { goto end_of_loop;; }
        
        if (speech_prob > peak_value) { // increase
            peak_value = speech_prob;
            iterations_decayed = 0;
        }
        else if (iterations_held <= hold_iterations) { // hold
            iterations_held++;
        }
        else { // decay
            const double decay_rate = 0.25f;
            iterations_held = 0;
            peak_value *= exp(-1 * decay_rate * iterations_decayed);
            iterations_decayed++;
        }

        if (peak_value >= speech_threshold) {
            pthread_mutex_lock(transcribe_data.mutex);

            transcribe_data.ready = 1;
            transcribe_data.flush = 0;
            transcribe_data.chance_of_speech = (float)peak_value;
            memcpy(transcribe_data.audio, combined_buffer, MIC_BUFFER_LEN * sizeof(float));
            pthread_cond_signal(transcribe_data.cond);

            pthread_mutex_unlock(transcribe_data.mutex);
            vad_previously_triggered = 1;
        } else if (vad_previously_triggered) {
            pthread_mutex_lock(transcribe_data.mutex);

            transcribe_data.ready = 1;
            transcribe_data.flush = 1;
            transcribe_data.chance_of_speech = (float)peak_value;
            pthread_cond_signal(transcribe_data.cond);

            pthread_mutex_unlock(transcribe_data.mutex);
            vad_previously_triggered = 0;
            peak_value = 0.0;
        }
        
        // Cleanup iteration
end_of_loop:
        ort->ReleaseValue(outputs[0]);
        ort->ReleaseValue(vad_state_tensor);
        vad_state_tensor = outputs[1];
        outputs[1] = NULL;
	}
	
// Cleanup for program exit ------------------------------------------------------------------------
second_mic_thread_fail:
    printf("Cleaning up second mic.\n");
    run_mic_threads = 0;
    pthread_join(thread[1], NULL);
    close_mic(mic2_ch);

first_mic_thread_fail:
    printf("Cleaning up first mic.\n");
    run_mic_threads = 0;
    pthread_join(thread[0], NULL);
	close_mic(mic1_ch);

    keep_running = 0;

    printf("Cleaning up ORT.\n");
    ort->ReleaseIoBinding(ort_io_binding);
    ort->ReleaseValue(vad_input_tensor);
    ort->ReleaseValue(vad_state_tensor);
    ort->ReleaseValue(vad_sr_tensor);
	ort->ReleaseMemoryInfo(ort_cpu_mem_info);
    ort->ReleaseMemoryInfo(ort_gpu_mem_info);
	ort->ReleaseSession(ort_session);
    ort->ReleaseCUDAProviderOptions(ort_cuda_opts);
	ort->ReleaseSessionOptions(ort_session_opts);
	ort->ReleaseEnv(ort_env);
    printf("Released all ORT bindings.\n");

ort_initialize_fail:
sherpa_denoiser_fail:
    SherpaOnnxDestroyOnlineSpeechDenoiser(sd);

sherpa_recognizer_fail:
    SherpaOnnxDestroyOnlineRecognizer(sherpa_recognizer);
    SherpaOnnxDestroyOnlineStream(sherpa_stream);

    printf("Exiting program.\n");
	return 0;
}