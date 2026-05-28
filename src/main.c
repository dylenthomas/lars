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
#include "fft.h"
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

#define NUM_THREADS 2

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

atomic_int run_mic_threads = 0;

const char* encoder_filename = "/home/dylenthomas/LiveASRonRPi-4/.model/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17/encoder-epoch-99-avg-1.onnx";
const char* decoder_filename = "/home/dylenthomas/LiveASRonRPi-4/.model/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17/decoder-epoch-99-avg-1.onnx";
const char* joiner_filename = "/home/dylenthomas/LiveASRonRPi-4/.model/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17/joiner-epoch-99-avg-1.onnx";
const char* tokens_filename = "/home/dylenthomas/LiveASRonRPi-4/.model/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17/tokens.txt";
const char* provider = "cpu"; 

int badStatus(OrtStatus* status, const OrtApi* ort) {
	// Make sure the API was accessed correctly 
	if (status != NULL) {
		const char* err_msg = ort->GetErrorMessage(status);
		fprintf(stderr, "ORT Error: %s\n", err_msg);
		ort->ReleaseStatus(status);
		return 1;
	}
	return 0;
}

float getSpeechProb(
        OrtValue*** outputs,
        size_t output_count,
		const OrtApi* ort,
		OrtSession* session,
		OrtRunOptions* run_opts,
        OrtIoBinding* io_binding,
        OrtAllocator* alloc
		) {
    if (badStatus(ort->RunWithBinding(session, run_opts, io_binding), ort)) { return -1.0f; }
    if (badStatus(ort->GetBoundOutputValues(io_binding, alloc, outputs, &output_count), ort)) { return -1.0f; }
	
	// Retrieve probability of speech from the model
	OrtValue* ort_speech_prob = (*outputs)[0];
	
	float* prob_data = NULL;
	if (badStatus(ort->GetTensorMutableData(ort_speech_prob, (void**)&prob_data), ort)) { return -1.0f; }
    return prob_data[0];
}

void* readMicData(void* ptr) {
    struct mic_thread_data* args = (struct mic_thread_data*)ptr;

    while (run_mic_threads) {
        int16_t tmp_buffer[MIC_BUFFER_LEN] = {0};
	    int i = 0;
	
        read_mic(tmp_buffer, args->device, MIC_BUFFER_LEN);
        // convert int mic data to float	
        pthread_mutex_lock(args->mutex);
	    while (i < MIC_BUFFER_LEN) { args->buffer[i] = (float)tmp_buffer[i] / 32768.0f; i++; }
        pthread_mutex_unlock(args->mutex);
        
        // avoid running prediction on old data
        atomic_store(args->updated, random());
    }

    return NULL;
}

int findSampleDelay(float* ref_buffer, float* other_buffer)  {
    // Find sample delay using frequency domain formula
    int i = 0;
    int n = MIC_BUFFER_LEN;
    int m_delay = 0;

    complex x[n], y[n], tmp[n], Rxy[n];

    while (i < n) {
        x[i].Re = ref_buffer[i];
        x[i].Im = 0.0f;
        y[i].Re = other_buffer[i];
        y[i].Im = 0.0f;

        i++;
    }

    fft(x, n, tmp);
    fft(y, n, tmp);

    i = 0;
    while (i < n) { 
        // compute correlation
        // the total fomula is the dot product between x and complex conjugate of y
        //  normalized by the magnitude
        float a = x[i].Re;
        float b = x[i].Im;
        float c = y[i].Re;
        float d = y[i].Im;

        double m1 = pow((double)(a*c + b*d), 2);
        double m2 = pow((double)(b*c - a*d), 2);
        double mag = sqrt(m1 + m2);

        Rxy[i].Re = (a*c + b*d)/mag;
        Rxy[i].Im = (b*c - a*d)/mag;

        i++;
    }

    ifft(Rxy, n, tmp);
    // The index of the max value represents our delay
    i = 0;
    while (i < n) {
        // compare absolute values
        float a = (Rxy[i].Re < 0.0) ? -Rxy[i].Re : Rxy[i].Re;
        float b = (Rxy[m_delay].Re < 0.0) ? -Rxy[m_delay].Re : Rxy[m_delay].Re;
        m_delay = (a > b) ? i : m_delay;

        i++;
    }
    
    return (m_delay > n/2) ? m_delay - n : m_delay;
}

int main(int argc, char *argv[]) {
    signal(SIGINT, intHandler);
    srandom(time(NULL));

    SherpaOnnxOnlineRecognizerConfig recognizer_config;
    memset(&recognizer_config, 0, sizeof(recognizer_config));
    recognizer_config.decoding_method = "greedy_search";
    recognizer_config.model_config.debug = 1;
    recognizer_config.model_config.num_threads = 1;
    recognizer_config.model_config.provider = provider;
    recognizer_config.model_config.tokens = tokens_filename;
    recognizer_config.model_config.transducer.encoder = encoder_filename;
    recognizer_config.model_config.transducer.decoder = decoder_filename;
    recognizer_config.model_config.transducer.joiner = joiner_filename;
    recognizer_config.enable_endpoint = 1;

    SherpaOnnxOnlineSpeechDenoiserConfig denoiser_config;
    memset(&denoiser_config, 0, sizeof(denoiser_config));
    denoiser_config.model.dpdfnet.model = "/home/dylenthomas/LiveASRonRPi-4/.model/dpdfnet8.onnx";
    denoiser_config.model.num_threads = 1;
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

    const SherpaOnnxDisplay *display = SherpaOnnxCreateDisplay(50);
    int32_t segment_id = 0;

    if (argc != 4) { printf("Args should be: VAD Model Path, Mic1 Name, Mic2 Name\n"); return 0; }
	const char* vad_model_path = argv[1];
	const char* mic1_name = argv[2];
    const char* mic2_name = argv[3];

    //struct keywordHM keywords = createKeywordHM(KEYWORD_CONF);

	const float speech_threshold = 0.5; // trigger threshold to start transcription
    size_t num_outputs = 2;
    
    double peak_value = 0.0f;
    int hold_iterations = 5;
    int iterations_held = 0;
    double decay_rate = 0.25f;
    int iterations_decayed = 0;

    int transcript_buffers = 0;
	
    int64_t sample_rate[] = {SAMPLE_RATE};
	const int64_t sample_rate_shape[] = {1};

	const int64_t input_data_shape[] = {1, MIC_BUFFER_LEN}; // input data will be the mic buffer
	const int64_t state_data_shape[] = {2, 1, 128};	

    atomic_long updated1 = ATOMIC_VAR_INIT(0);
    atomic_long updated2 = ATOMIC_VAR_INIT(0);
    float mic1_buffer[MIC_BUFFER_LEN] = {0};
    float mic2_buffer[MIC_BUFFER_LEN] = {0};
    float combined_buffer[MIC_BUFFER_LEN] = {0};
    float long_buffer[SAMPLE_RATE * LONGEST_TRANSCRIPT] = {0};
	float initial_state[STATE_LEN] = {0};

    int long_buffer_size = 0;
    int long_buffer_capacity = SAMPLE_RATE * LONGEST_TRANSCRIPT;

	OrtValue* input_tensor = NULL;
	OrtValue* state_tensor = NULL;
	OrtValue* sr_tensor = NULL;

    int ret;
    long int mic1_last_updated;
    long int mic2_last_updated;

    complex X[MIC_BUFFER_LEN], Y[MIC_BUFFER_LEN], combined[MIC_BUFFER_LEN], tmp[MIC_BUFFER_LEN];
// Initialize the ORT Api --------------------------------------------------------------------------
	printf("Initializing ORT...\n");
	const OrtApi* ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
	if (ort == NULL) { fprintf(stderr, "ORT api returned nullptr!\n"); return 1; }

	OrtEnv* env = NULL;
	if (badStatus(ort->CreateEnv(ORT_LOGGING_LEVEL_VERBOSE, "onnxruntime", &env), ort)) { goto ort_env_fail; }

	OrtSessionOptions* session_opts = NULL;
	if (badStatus(ort->CreateSessionOptions(&session_opts), ort)) { goto ort_session_opts_fail; }
	if (badStatus(ort->SetIntraOpNumThreads(session_opts, 1), ort)) { goto ort_session_opts_fail; }
	if (badStatus(ort->SetInterOpNumThreads(session_opts, 1), ort)) { goto ort_session_opts_fail; }
	if (badStatus(ort->SetSessionGraphOptimizationLevel(session_opts, ORT_ENABLE_ALL), ort)) { goto ort_session_opts_fail; }

    OrtCUDAProviderOptionsV2* cuda_opts = NULL;
    if (badStatus(ort->CreateCUDAProviderOptions(&cuda_opts), ort)) { goto ort_cuda_opts_fail; }
    if (badStatus(ort->SessionOptionsAppendExecutionProvider_CUDA_V2(session_opts, cuda_opts), ort)) { goto ort_cuda_opts_fail; }

	OrtSession* ort_session = NULL;
	if (badStatus(ort->CreateSession(env, vad_model_path, session_opts, &ort_session), ort)) { goto ort_session_fail; }

	OrtRunOptions* ort_run_opts = NULL;
	if (badStatus(ort->CreateRunOptions(&ort_run_opts), ort)) { goto ort_session_fail; }

    OrtMemoryInfo* gpu_mem_info = NULL;
    if (badStatus(ort->CreateMemoryInfo_V2("Cuda", OrtMemoryInfoDeviceType_GPU, VENDOR_ID, DEVICE_ID,
            OrtMemTypeDefault, GPU_ALIGNMENT, OrtDeviceAllocator, &gpu_mem_info), ort)) { goto ort_gpu_mem_info_fail; }

	OrtMemoryInfo* cpu_mem_info = NULL;
	if (badStatus(ort->CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeDefault, &cpu_mem_info), ort)) { goto ort_cpu_mem_info_fail; }

	OrtAllocator* alloc = NULL;
	if (badStatus(ort->GetAllocatorWithDefaultOptions(&alloc), ort)) { goto ort_cpu_mem_info_fail; }
	
	if (badStatus(ort->CreateTensorWithDataAsOrtValue(
		gpu_mem_info, sample_rate, sizeof(sample_rate), sample_rate_shape, SAMPLE_RATE_DIMS,
		ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &sr_tensor), ort)) { goto ort_sr_tensor_fail; }

	// create initializing state for model of all zeros 
	if (badStatus(ort->CreateTensorWithDataAsOrtValue(
		gpu_mem_info, initial_state, sizeof(initial_state), state_data_shape, STATE_DIMS,
		ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &state_tensor), ort)) { goto ort_state_tensor_fail; }

	if (badStatus(ort->CreateTensorWithDataAsOrtValue(
		gpu_mem_info, combined_buffer, sizeof(combined_buffer), input_data_shape, INPUT_DIMS, 
		ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor), ort)) { goto ort_input_tensor_fail; }

    OrtIoBinding* io_binding = NULL;
    if (badStatus(ort->CreateIoBinding(ort_session, &io_binding), ort)) { goto ort_io_binding_fail; }

    if (badStatus(ort->BindInput(io_binding, "input", input_tensor), ort)) { goto ort_io_binding_fail; }
    if (badStatus(ort->BindInput(io_binding, "state", state_tensor), ort)) { goto ort_io_binding_fail; }
    if (badStatus(ort->BindInput(io_binding, "sr", sr_tensor), ort)) { goto ort_io_binding_fail; }
    if (badStatus(ort->BindOutputToDevice(io_binding, "output", cpu_mem_info), ort)) { goto ort_io_binding_fail; }
    if (badStatus(ort->BindOutputToDevice(io_binding, "stateN", cpu_mem_info), ort)) { goto ort_io_binding_fail; }
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

    mic1_last_updated = *mic_data[0].updated;
    mic2_last_updated = *mic_data[1].updated;
// -------------------------------------------------------------------------------------------------
    int i;

    while (keep_running) {
        // wait until both buffers are populated
        if (mic1_last_updated == *mic_data[0].updated) { continue; }
        mic1_last_updated = *mic_data[0].updated;

        if (mic2_last_updated == *mic_data[1].updated) { continue; }
        mic2_last_updated = *mic_data[1].updated;

        // get the delay of mic2 relative to mic1
        pthread_mutex_lock(mic_data[0].mutex);
        pthread_mutex_lock(mic_data[1].mutex);
        for (int i = 0; i < MIC_BUFFER_LEN; i++) {
            X[i].Re = mic_data[0].buffer[i]; X[i].Im = 0.0f;
            Y[i].Re = mic_data[1].buffer[i]; Y[i].Im = 0.0f;
        }

        int delay = findSampleDelay(mic_data[0].buffer, mic_data[1].buffer);

        // Audio reached mic 2 first
        if (delay < 0) { 
            memcpy(combined_buffer, mic_data[1].buffer, MIC_BUFFER_LEN * sizeof(float));
        }
        // Audio reached mic 1 first
        else { 
            memcpy(combined_buffer, mic_data[0].buffer, MIC_BUFFER_LEN * sizeof(float));     
        }

        /*
        fft(X, MIC_BUFFER_LEN, tmp);
        fft(Y, MIC_BUFFER_LEN, tmp);

        for (int f = 0; f < MIC_BUFFER_LEN; f++){
            double shift_angle = -2 * PI * f * delay/MIC_BUFFER_LEN;
            complex Y_shifted;

            Y_shifted.Re = Y[f].Re * cos(shift_angle) - Y[f].Im * sin(shift_angle);
            Y_shifted.Im = Y[f].Re * sin(shift_angle) + Y[f].Im * cos(shift_angle);

            combined[f].Re = (X[f].Re + Y_shifted.Re) * 0.5f;
            combined[f].Im = (X[f].Im + Y_shifted.Im) * 0.5f;
        }

        ifft(combined, MIC_BUFFER_LEN, tmp);
        memset(combined_buffer, 0, sizeof(combined_buffer));

        for (int i = 0; i < MIC_BUFFER_LEN; i++) {
            combined_buffer[i] = combined[i].Re / MIC_BUFFER_LEN;
        }
        */

        pthread_mutex_unlock(mic_data[0].mutex);
        pthread_mutex_unlock(mic_data[1].mutex);

		float speech_prob;
        OrtValue** outputs = NULL;
            
        const SherpaOnnxDenoisedAudio* chunk = SherpaOnnxOnlineSpeechDenoiserRun(sd, combined_buffer, MIC_BUFFER_LEN, SAMPLE_RATE);
        if (chunk == NULL) { goto end_loop; }
        //memcpy(combined_buffer, chunk->samples, chunk->n * sizeof(float));
        //if (chunk->n < MIC_BUFFER_LEN) {
        //    memset(combined_buffer + chunk->n, 0, (MIC_BUFFER_LEN - chunk->n) * sizeof(float));
        //}

		speech_prob = getSpeechProb(&outputs, num_outputs, ort, ort_session, ort_run_opts, io_binding, alloc);
        if (speech_prob == -1.0f) { goto end_loop; } // if VAD failed just skip the iteration 
        
        if (speech_prob > peak_value) {
        // Increase
            peak_value = speech_prob;
            iterations_decayed = 0;
        }
        else if (iterations_held <= hold_iterations) {
        // Hold
            iterations_held++;
        }
        else { 
        // Decay
            iterations_held = 0;
            peak_value *= exp(-1 * decay_rate * iterations_decayed);
            iterations_decayed++;
        }
        
        printf("%.2f\n", peak_value);

        if (peak_value >= speech_threshold) {
            int samples_to_copy = chunk->n;
            if (long_buffer_size + samples_to_copy > long_buffer_capacity) {
                samples_to_copy = long_buffer_capacity - long_buffer_size;
            }
            memcpy(long_buffer + long_buffer_size, chunk->samples, samples_to_copy * sizeof(float));
            long_buffer_size += samples_to_copy;


            SherpaOnnxOnlineStreamAcceptWaveform(sherpa_stream, chunk->sample_rate, chunk->samples, chunk->n);
            while (SherpaOnnxIsOnlineStreamReady(sherpa_recognizer, sherpa_stream)) {
                SherpaOnnxDecodeOnlineStream(sherpa_recognizer, sherpa_stream);
            }

            const SherpaOnnxOnlineRecognizerResult* r = SherpaOnnxGetOnlineStreamResult(sherpa_recognizer, sherpa_stream);
            if (strlen(r->text)) { printf("%s\n", r->text); }

            if (SherpaOnnxOnlineStreamIsEndpoint(sherpa_recognizer, sherpa_stream)) {
                if (strlen(r->text)) { ++segment_id; }
                SherpaOnnxOnlineStreamReset(sherpa_recognizer, sherpa_stream);
            }
            SherpaOnnxDestroyOnlineRecognizerResult(r);
            
            if (long_buffer_size >= long_buffer_capacity) {
                printf("Reached capacity\n");
                SherpaOnnxWriteWave(long_buffer, long_buffer_size, SAMPLE_RATE, "/home/dylenthomas/LiveASRonRPi-4/wavs/transcript.wav");
                long_buffer_size = 0;
                peak_value = 0.0;
            }
        }
        else { // TODO: Make this only run if we previously had speech.f
            const SherpaOnnxDenoisedAudio* tail = SherpaOnnxOnlineSpeechDenoiserFlush(sd);
            if (tail) { 
                int samples_to_copy = tail->n;
                if (long_buffer_size + samples_to_copy > long_buffer_capacity) {
                    samples_to_copy = long_buffer_capacity - long_buffer_size;
                }
                memcpy(long_buffer + long_buffer_size, tail->samples, samples_to_copy * sizeof(float));
                long_buffer_size += samples_to_copy;

                SherpaOnnxOnlineStreamAcceptWaveform(sherpa_stream, tail->sample_rate, tail->samples, tail->n);
                SherpaOnnxDestroyDenoisedAudio(tail); 
            }
            
            // Only write if we actually accumulated something
            if (long_buffer_size > 0) {
                printf("Reached end of speech and logged audio\n");
                SherpaOnnxWriteWave(long_buffer, long_buffer_size, SAMPLE_RATE, "/home/dylenthomas/LiveASRonRPi-4/wavs/transcript.wav");
                long_buffer_size = 0;
            }
            
            float tail_padding[4800] = {0};
            SherpaOnnxOnlineStreamAcceptWaveform(sherpa_stream, SAMPLE_RATE, tail_padding, 4800);
            SherpaOnnxOnlineStreamInputFinished(sherpa_stream);
            while (SherpaOnnxIsOnlineStreamReady(sherpa_recognizer, sherpa_stream)) {
                SherpaOnnxDecodeOnlineStream(sherpa_recognizer, sherpa_stream);
            }

            const SherpaOnnxOnlineRecognizerResult* r = SherpaOnnxGetOnlineStreamResult(sherpa_recognizer, sherpa_stream);
            if (strlen(r->text)) { printf("%s\n", r->text); }

            segment_id = 0;
            peak_value = 0.0;
            SherpaOnnxDestroyOnlineRecognizerResult(r);
            SherpaOnnxOnlineStreamReset(sherpa_recognizer, sherpa_stream);
        }
        
        // Cleanup iteration
end_loop:
        ort->ReleaseValue(state_tensor);
        ort->ReleaseValue(outputs[0]);
        state_tensor = outputs[1];
        outputs[1] = NULL;
        SherpaOnnxDestroyDenoisedAudio(chunk);
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
ort_io_binding_fail:
    ort->ReleaseIoBinding(io_binding);
ort_input_tensor_fail:
    ort->ReleaseValue(input_tensor);
ort_state_tensor_fail:
    ort->ReleaseValue(state_tensor);
ort_sr_tensor_fail:
    ort->ReleaseValue(sr_tensor);
ort_cpu_mem_info_fail:
	ort->ReleaseMemoryInfo(cpu_mem_info);
ort_gpu_mem_info_fail:
    ort->ReleaseMemoryInfo(gpu_mem_info);
ort_session_fail:
	ort->ReleaseSession(ort_session);
ort_cuda_opts_fail:
    ort->ReleaseCUDAProviderOptions(cuda_opts);
ort_session_opts_fail:
	ort->ReleaseSessionOptions(session_opts);
ort_env_fail:
	ort->ReleaseEnv(env);
    printf("Released all ORT bindings.\n");

sherpa_denoiser_fail:
    SherpaOnnxDestroyOnlineSpeechDenoiser(sd);

sherpa_recognizer_fail:
    SherpaOnnxDestroyOnlineRecognizer(sherpa_recognizer);
    SherpaOnnxDestroyOnlineStream(sherpa_stream);
    SherpaOnnxDestroyDisplay(display);
    
    printf("Exiting program.\n");
	return 0;
}