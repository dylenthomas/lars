#include "main.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <time.h>
#include <signal.h>

#include "onnxruntime_c_api.h"
#include "mic_access.h"
#include "sherpa-onnx/c-api/c-api.h"

// =====================================================================================================================
// - 1 Corinthians 10:31
// =====================================================================================================================

// TODO: Add new mics
// TODO: Setup unit tests for the queue
// TODO: Start thinking about thread structure to put the queue before transcription and have it manage that
//  I think that the main thread can just update the queue, and the transcription thread is
//  checking the queue constantly for new data to transcribe
// TODO: BEFORE I start implementing the queue: Get transcription working with the new microphone setup
// TODO: Rethink the mic updated functionality

// TODO: For communication with the RaspberryPi think about using mosquitto (or another MQTT tool)
// TODO: On the RaspberryPi use pinctrl (or other similar cli) to control hardware states

void intHandler(int dummy) {
    keep_running = 0;
}

/**
 * Check status of any ORT function
 *
 * @param status Status pointer returned by ORT functino
 * @param ort ORT Api pointer
 * @return good or bad status (1 = bad status and function failed, 0 = good status)
 */
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

/**
 * Initialize the ORT environment and necessary tensors
 *
 * @return ORT Api pointer
 */
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
		ort_gpu_mem_info, (void*)sample_rate, sizeof(sample_rate), sample_rate_shape, SAMPLE_RATE_DIMS,
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

/**
 * Get the speech probability form the VAD model outputs
 *
 * @param outputs Outputs from the VAD model
 * @param ort ORT Api pointer
 * @return probability of speech (if < 0.0f then something went wrong)
 */
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

/**
 * Mic thread function that persistently runs throughout program life
 *
 * @param ptr thread struct containing all necessary information
 * @return NULL
 */
static void* readMicData(void* ptr) {
    struct mic_thread_data* args = (struct mic_thread_data*)ptr;

    while (run_mic_threads) {
        int16_t tmp_buffer[MIC_BUFFER_LEN] = {0};
	    int i = 0;

        read_mic(tmp_buffer, args->device, MIC_BUFFER_LEN);
        // convert int mic data to float	
        pthread_mutex_lock(args->mutex);
	    while (i < MIC_BUFFER_LEN) { args->buffer[i] = (float)tmp_buffer[i] / 32768.0f; i++; }

        // avoid predicting on old values
        long newUpdated = random();
        args->updated = &newUpdated;
        pthread_mutex_unlock(args->mutex);
    }

    return NULL;
}

/**
 * Transcribe a given audio buffer into speech
 *
 * @param ptr thread struct containing all necessary information
 * @return NULL
 */
static void* transcribe(void* ptr) {
    struct transcribe_thread_data* args = ptr;
    const SherpaOnnxOnlineRecognizer* recognizer = args->recognizer;
    const SherpaOnnxOnlineStream* stream = args->stream;
    const SherpaOnnxOnlineSpeechDenoiser* denoiser = args->denoiser;

    while (keep_running) {
        float audio[MIC_BUFFER_LEN] = {0};
        int flush = 0;
        float chance_of_speech = 0.0f;

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

static int initializeSherpa() {
    // Configure the ASR model
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

    sherpa_recognizer = SherpaOnnxCreateOnlineRecognizer(&recognizer_config);
    if (sherpa_recognizer == NULL) {
        fprintf(stderr, "Please check your config!\n");
        SherpaOnnxDestroyOnlineRecognizer(sherpa_recognizer);
        SherpaOnnxDestroyOnlineStream(sherpa_stream);

        return 1;
    }
    sherpa_stream = SherpaOnnxCreateOnlineStream(sherpa_recognizer);

    // Configure the speech denoiser model
    memset(&denoiser_config, 0, sizeof(denoiser_config));
    denoiser_config.model.dpdfnet.model = denoiser_model_path;
    denoiser_config.model.num_threads = 2;
    denoiser_config.model.debug = 0;
    denoiser_config.model.provider = "cpu";

    sd = SherpaOnnxCreateOnlineSpeechDenoiser(&denoiser_config);
    if (sd == NULL) {
        fprintf(stderr, "Failed to create speech denoiser\n");
        SherpaOnnxDestroyOnlineRecognizer(sherpa_recognizer);
        SherpaOnnxDestroyOnlineStream(sherpa_stream);
        SherpaOnnxDestroyOnlineSpeechDenoiser(sd);

        return 1;
    }

    return 0;
}

int main() {
    signal(SIGINT, intHandler); // Allow safe exit from program with Ctrl-C

    srandom(time(NULL)); // Initialize the random function

    double peak_value = 0.0;
    int iterations_held = 0;
    int iterations_decayed = 0;

    float mic1_buffer[MIC_BUFFER_LEN] = {0};
    float mic2_buffer[MIC_BUFFER_LEN] = {0};
    float mic3_buffer[MIC_BUFFER_LEN] = {0};
    float mic4_buffer[MIC_BUFFER_LEN] = {0};
    float mic5_buffer[MIC_BUFFER_LEN] = {0};

    int ret;

    snd_pcm_t* mic_chs[NUM_MICS];
    long mic_updates[NUM_MICS];

    int vad_previously_triggered = 0;

    // Initialize Sherpa models ========================================================================================

    if (initializeSherpa()) { return 1; }

    // Initialize the ORT Api ==========================================================================================

    const OrtApi* ort = initializeORT();
    if (ort == NULL) { goto ort_initialize_fail; }

    // Initialize Microphones ==========================================================================================

    // TODO: Move all the microphone variables to arrays to make modifying all of them at once easier.

    printf("Initializing microphones...\n");

    for (int i = 0; i < NUM_MICS; i++) {
        init_mic(mic1_name, &mic_chs[i], SAMPLE_RATE, CHANNELS );
    }

    // Setup threads ===================================================================================================

    struct transcribe_thread_data transcribe_data;
    pthread_mutex_t transcribe_mutex = PTHREAD_MUTEX_INITIALIZER;
    pthread_cond_t transcribe_cond = PTHREAD_COND_INITIALIZER;
    transcribe_data.recognizer = sherpa_recognizer;
    transcribe_data.stream = sherpa_stream;
    transcribe_data.denoiser = sd;
    transcribe_data.cond = &transcribe_cond;
    transcribe_data.mutex = &transcribe_mutex;
    transcribe_data.ready = 0;

    struct mic_thread_data mic_data[NUM_MICS];
    pthread_mutex_t mic1_mutex = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_t mic2_mutex = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_t mic3_mutex = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_t mic4_mutex = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_t mic5_mutex = PTHREAD_MUTEX_INITIALIZER;

    mic_data[0].buffer = mic1_buffer;
    mic_data[0].device = mic_chs[0];
    mic_data[0].updated = &mic_updates[0];
    mic_data[0].mutex = &mic1_mutex;

    mic_data[1].buffer = mic2_buffer;
    mic_data[1].device = mic_chs[1];
    mic_data[1].updated = &mic_updates[1];
    mic_data[1].mutex = &mic2_mutex;

    mic_data[2].buffer = mic3_buffer;
    mic_data[2].device = mic_chs[2];
    mic_data[2].updated = &mic_updates[2];
    mic_data[2].mutex = &mic3_mutex;

    mic_data[3].buffer = mic4_buffer;
    mic_data[3].device = mic_chs[3];
    mic_data[3].updated = &mic_updates[3];
    mic_data[3].mutex = &mic4_mutex;

    mic_data[4].buffer = mic5_buffer;
    mic_data[4].device = mic_chs[4];
    mic_data[4].updated = &mic_updates[4];
    mic_data[4].mutex = &mic5_mutex;

    // Start threads ===================================================================================================

    pthread_t thread[NUM_THREADS];
    
    printf("Starting audio threads.\n");
    run_mic_threads = 1;

    int num_mics_initialized = 0;
    while (num_mics_initialized < NUM_MICS) {
        if ((ret = pthread_create(&thread[num_mics_initialized], NULL, readMicData, &mic_data[num_mics_initialized])) != 0) {
            fprintf(stderr, "Error: failed to create mic %d thread. Code: %d", num_mics_initialized, ret);
            goto mic_threads_fail;
        }

        num_mics_initialized++;
    }

    // Initialize the local mic update fields
    for (int i = 0; i < NUM_MICS; i++) {
        mic_updates[i] = *mic_data[i].updated;
    }


    if ((ret = pthread_create(&thread[NUM_THREADS - 1], NULL, transcribe, &transcribe_data)) != 0) {
        fprintf(stderr, "Error: failed to create transcription thread. Code: %d", ret);
        goto transcribe_thread_fail;
    }

    // Main Loop =======================================================================================================
    while (keep_running) {
        int hold_iterations = 5;
        const double speech_threshold = 0.7; // trigger threshold to start transcription

        float speech_prob;
        OrtValue** outputs = NULL;

        float rms1 = 0, rms2 = 0;

        // Accessing mic data ==========================================================================================

        // TODO: I am not sure if this method of accessing mic data is the best, maybe it should be processed in another
        //  thread because of how many mics there are.
        //  Maybe I should have a thread for combining mics 3-5 together and a thread that determines which node to use.

        for (int i = 0; i < NUM_MICS; i++) {
            pthread_mutex_lock(mic_data[i].mutex);
        }

        // TODO: I don't think that waiting for all five mics to be updated at the same time is necessary or smart.
        for (int i = 0; i < NUM_MICS; i++) {
            if (mic_updates[i] == *mic_data[i].updated) {
                for (int j = 0; i < NUM_MICS; j++) {
                    pthread_mutex_unlock(mic_data[j].mutex);
                }
                goto end_of_loop;
            }
        }

        // TODO: Combine the last three mics into a single stream using fft stuff.

        for (int i = 0; i < MIC_BUFFER_LEN; i++) {
            rms1 += mic_data[0].buffer[i] * mic_data[0].buffer[i];
            rms2 += mic_data[1].buffer[i] * mic_data[1].buffer[i];
        }
        memcpy(combined_buffer, rms1 > rms2 ? mic_data[0].buffer : mic_data[1].buffer, MIC_BUFFER_LEN * sizeof(float));

        for (int i = 0; i < NUM_MICS; i++) {
            pthread_mutex_unlock(mic_data[i].mutex);
        }
        // =============================================================================================================

		speech_prob = getSpeechProb(&outputs, ort);
        if (speech_prob == -2.0f) { continue; }
        if (speech_prob == -1.0f) { goto end_of_loop; }
        
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
clean_up_loop:
        ort->ReleaseValue(outputs[0]);
        ort->ReleaseValue(vad_state_tensor);
        vad_state_tensor = outputs[1];
        outputs[1] = NULL;
end_of_loop:
	}
	
// Cleanup for program exit ------------------------------------------------------------------------
transcribe_thread_fail:
    pthread_join(thread[NUM_THREADS - 1], NULL);

mic_threads_fail:
    run_mic_threads = 0;
    for (int i = 0; i < num_mics_initialized; i++) {
        fprintf(stdout, "Cleaning up mic %d", i + 1);
        pthread_join(thread[i], NULL);
        close_mic(mic_ch_list[i]);
    }

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
    SherpaOnnxDestroyOnlineSpeechDenoiser(sd);
    SherpaOnnxDestroyOnlineRecognizer(sherpa_recognizer);
    SherpaOnnxDestroyOnlineStream(sherpa_stream);

    printf("Exiting program.\n");
	return 0;
}