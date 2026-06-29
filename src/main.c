#include "main.h"

#include <stdio.h>
#include <math.h>
#include <pthread.h>
#include <signal.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

// =====================================================================================================================
// - 1 Corinthians 10:31
// =====================================================================================================================

// TODO: Add a flush delay of a few seconds
// TODO: Start working on keyword identification
// TODO: Include more robust error handling
// TODO: Add "write to wav" flag

// TODO: For communication with the RaspberryPi think about using mosquitto (or another MQTT tool)
// TODO: On the RaspberryPi use pinctrl (or other similar cli) to control hardware states

// TODO: Consider the scenario where a mic is not updating for multiple iterations and how to handle that

int main() {
    double peak_value = 0.0;
    int iterations_held = 0;
    int iterations_decayed = 0;
    int ret;
    int vad_previously_triggered = 0;

    // Initialize Sherpa models ========================================================================================
    if (initializeSherpa() < 0) { return -1; }

    // Initialize the ORT Api ==========================================================================================
    const OrtApi* ort = initializeORT();
    if (ort == NULL) { goto ort_fail; }

    const int a_and_t_status = initialize_mics_and_transcription();
    if (a_and_t_status == -1) {
        goto sherpa_fail;
    }
    
    // Start threads ===================================================================================================
    pthread_t thread[NUM_THREADS];
    
    printf("Starting audio threads.\n");
    run_mic_threads = 1;
    int num_mics_initialized = 0;
    while (num_mics_initialized < NUM_MICS) {
        if ((ret = pthread_create(&thread[num_mics_initialized], NULL, mic_thread, &mic_thread_data[num_mics_initialized])) != 0) {
            fprintf(stderr, "Error: failed to create mic %d thread. Code: %d", num_mics_initialized, ret);
            goto mic_threads_fail; 
        }

        printf("Started mic %d\n", num_mics_initialized);
        num_mics_initialized++;
    }
   
    run_transcription_thread = 1;
    if ((ret = pthread_create(&thread[TRANSCRIPT_THREAD_ID], NULL, transcription_thread, &transcription_thread_data)) != 0) {
        fprintf(stderr, "Error: failed to create transcription thread. Code: %d", ret);
        goto transcription_thread_fail; 
    }
    printf("Started transcription thread\n");


    // Main Loop =======================================================================================================
    while (!kbhit()) {
        int hold_iterations = 5;
        const double speech_threshold = 0.7; // trigger threshold to start transcription

        float speech_prob;
        OrtValue** outputs = NULL;

        float rms[NUM_MICS] = {0};

        // Accessing mic data ==========================================================================================

        int ready_mics = 0;
        for (int i = 0; i < NUM_MICS; i++) {
            pthread_mutex_lock(mic_thread_data[i].mutex);
            ready_mics += mic_thread_data[i].data_ready;
        }
      
        if (ready_mics == NUM_MICS) {
            for (int i = 0; i < NUM_MICS; i++) {
                for (int j = 0; j < MIC_BUFFER_LEN; j++) {
                    rms[i] += mic_thread_data[i].buffer[j] * mic_thread_data[i].buffer[j];
                }
            }

            const int loudest_mic = max(rms, NUM_MICS);
            memcpy(combined_buffer, mic_thread_data[loudest_mic].buffer, MIC_BUFFER_LEN * sizeof(float));

            for (int i = 0; i < NUM_MICS; i++) {
                mic_thread_data[i].data_ready = 0;
                pthread_cond_signal(mic_thread_data[i].cond);
                pthread_mutex_unlock(mic_thread_data[i].mutex);
            } 
        } else {
            // Don't want to run transcription
            for (int i = 0; i < NUM_MICS; i++) {
                pthread_mutex_unlock(mic_thread_data[i].mutex);
            }
            usleep(1000);
            continue;
        }

        // =============================================================================================================

		speech_prob = chance_of_speech(&outputs, ort);
        if (speech_prob == -2.0f) { continue; }
        if (speech_prob == -1.0f) { goto clean_up_loop; }
        
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
            printf("rms = [ %.3f, %.3f, %.3f]\n", rms[0], rms[1], rms[2]);

            pthread_mutex_lock(transcription_thread_data.mutex);

            transcription_thread_data.ready = 1;
            transcription_thread_data.flush = 0;
            transcription_thread_data.chance_of_speech = (float)peak_value;
            memcpy(transcription_thread_data.audio, combined_buffer, MIC_BUFFER_LEN * sizeof(float));
            pthread_cond_signal(transcription_thread_data.cond);

            pthread_mutex_unlock(transcription_thread_data.mutex);

            vad_previously_triggered = 1;
        } else if (vad_previously_triggered) {
            pthread_mutex_lock(transcription_thread_data.mutex);

            transcription_thread_data.ready = 1;
            transcription_thread_data.flush = 1;
            transcription_thread_data.chance_of_speech = (float)peak_value;
            pthread_cond_signal(transcription_thread_data.cond);

            pthread_mutex_unlock(transcription_thread_data.mutex);

            vad_previously_triggered = 0;
            peak_value = 0.0;
        }
        
clean_up_loop:
        ort->ReleaseValue(outputs[0]);
        ort->ReleaseValue(vad_state_tensor);
        vad_state_tensor = outputs[1];
        outputs[1] = NULL;
	}
	
    // Cleanup for program exit ----------------------------------------------------------------------
transcription_thread_fail:
    run_transcription_thread = 0;
    pthread_join(thread[NUM_THREADS - 1], NULL);

mic_threads_fail:
    run_mic_threads = 0;
    for (int i = 0; i < num_mics_initialized; i++) {
        fprintf(stdout, "Cleaning up mic %d", i + 1);
        pthread_join(thread[i], NULL);
        close_mic(mic_chs[i]);
    }

sherpa_fail:
    printf("Cleaning up ORT.\n");
    ort->ReleaseIoBinding(ort_io_binding);
    ort->ReleaseValue(vad_input_tensor);
    ort->ReleaseValue(vad_sr_tensor);
    ort->ReleaseMemoryInfo(ort_gpu_mem_info);
	ort->ReleaseSession(ort_session);
    ort->ReleaseCUDAProviderOptions(ort_cuda_opts);
	ort->ReleaseSessionOptions(ort_session_opts);
	ort->ReleaseEnv(ort_env);
    printf("Released all ORT bindings.\n");

ort_fail:
    SherpaOnnxDestroyOnlineRecognizer(sherpa_recognizer);
    SherpaOnnxDestroyOnlineStream(sherpa_stream);
    printf("Exiting program.\n");
	return 0;
}

int kbhit(void) {
    int k;
    ioctl(STDIN_FILENO, FIONREAD, &k);
    return k;
}

static int max(const float values[], const int num_vals) {
    int max_loc = 0;
    float max_val = -INFINITY;
    for (int i = 0; i < num_vals; i++) {
        if (values[i] >  max_val) {
            max_val = values[i];
            max_loc = i;
        }
    }

    return max_loc;
}

static const OrtApi* initializeORT() {
    printf("Initializing ORT...\n");
	const OrtApi* ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
	if (ort == NULL) { fprintf(stderr, "ORT api returned nullptr!\n"); return NULL; }

	if (bad_ort_status(ort->CreateEnv(ORT_LOGGING_LEVEL_VERBOSE, "onnxruntime", &ort_env), ort)) { goto ort_env_fail; }

	if (bad_ort_status(ort->CreateSessionOptions(&ort_session_opts), ort)) { goto ort_session_opts_fail; }
	if (bad_ort_status(ort->SetIntraOpNumThreads(ort_session_opts, 1), ort)) { goto ort_session_opts_fail; }
	if (bad_ort_status(ort->SetInterOpNumThreads(ort_session_opts, 1), ort)) { goto ort_session_opts_fail; }
	if (bad_ort_status(ort->SetSessionGraphOptimizationLevel(ort_session_opts, ORT_ENABLE_ALL), ort)) { goto ort_session_opts_fail; }

    if (bad_ort_status(ort->CreateCUDAProviderOptions(&ort_cuda_opts), ort)) { goto ort_cuda_opts_fail; }
    if (bad_ort_status(ort->SessionOptionsAppendExecutionProvider_CUDA_V2(ort_session_opts, ort_cuda_opts), ort)) { goto ort_cuda_opts_fail; }

	if (bad_ort_status(ort->CreateSession(ort_env, vad_model_path, ort_session_opts, &ort_session), ort)) { goto ort_session_fail; }

	if (bad_ort_status(ort->CreateRunOptions(&ort_run_opts), ort)) { goto ort_session_fail; }

    if (bad_ort_status(ort->CreateMemoryInfo_V2("Cuda", OrtMemoryInfoDeviceType_GPU,
        VENDOR_ID, DEVICE_ID, OrtMemTypeDefault, GPU_ALIGNMENT,
        OrtDeviceAllocator, &ort_gpu_mem_info), ort)) { goto ort_gpu_mem_info_fail; }

	if (bad_ort_status(ort->CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeDefault,
	    &ort_cpu_mem_info), ort)) { goto ort_cpu_mem_info_fail; }

	if (bad_ort_status(ort->GetAllocatorWithDefaultOptions(&ort_alloc), ort)) { goto ort_cpu_mem_info_fail; }

	if (bad_ort_status(ort->CreateTensorWithDataAsOrtValue(
		ort_gpu_mem_info, (void*)sample_rate, sizeof(sample_rate), sample_rate_shape, SAMPLE_RATE_DIMS,
		ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &vad_sr_tensor), ort)) { goto ort_sr_tensor_fail; }

	// create initializing state for model of all zeros
	if (bad_ort_status(ort->CreateTensorWithDataAsOrtValue(
		ort_gpu_mem_info, initial_state, sizeof(initial_state), state_data_shape, STATE_DIMS,
		ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &vad_state_tensor), ort)) { goto ort_state_tensor_fail; }

	if (bad_ort_status(ort->CreateTensorWithDataAsOrtValue(
		ort_gpu_mem_info, combined_buffer, sizeof(combined_buffer), input_data_shape, INPUT_DIMS,
		ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &vad_input_tensor), ort)) { goto ort_input_tensor_fail; }

    if (bad_ort_status(ort->CreateIoBinding(ort_session, &ort_io_binding), ort)) { goto ort_io_binding_fail; }

    if (bad_ort_status(ort->BindInput(ort_io_binding, "input", vad_input_tensor), ort)) { goto ort_io_binding_fail; }
    if (bad_ort_status(ort->BindInput(ort_io_binding, "state", vad_state_tensor), ort)) { goto ort_io_binding_fail; }
    if (bad_ort_status(ort->BindInput(ort_io_binding, "sr", vad_sr_tensor), ort)) { goto ort_io_binding_fail; }
    if (bad_ort_status(ort->BindOutputToDevice(ort_io_binding, "output", ort_cpu_mem_info), ort)) { goto ort_io_binding_fail; }
    if (bad_ort_status(ort->BindOutputToDevice(ort_io_binding, "stateN", ort_cpu_mem_info), ort)) { goto ort_io_binding_fail; }

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
    printf("ORT initialize failed, released all ORT bindings.\n");

    return NULL;
}

static int bad_ort_status(OrtStatus* status, const OrtApi* ort) {
	// Make sure the API was accessed correctly 
	if (status != NULL) {
		const char* err_msg = ort->GetErrorMessage(status);
		fprintf(stderr, "ORT Error: %s\n", err_msg);
		ort->ReleaseStatus(status);
		return 1;
	}
	return 0;
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

        return -1;
    }
    sherpa_stream = SherpaOnnxCreateOnlineStream(sherpa_recognizer);
    
    return 0;
}

float chance_of_speech(OrtValue*** outputs, const OrtApi* ort) {
    size_t output_count = 2;
    if (bad_ort_status(ort->RunWithBinding(ort_session, ort_run_opts, ort_io_binding), ort)) { return -2.0f; }
    if (bad_ort_status(ort->GetBoundOutputValues(ort_io_binding, ort_alloc, outputs, &output_count), ort)) { return -2.0f; }
	
	// Retrieve probability of speech from the model
	OrtValue* ort_speech_prob = (*outputs)[0];
	
	float* prob_data = NULL;
	if (bad_ort_status(ort->GetTensorMutableData(ort_speech_prob, (void**)&prob_data), ort)) { return -1.0f; }
    return prob_data[0];
}

int initialize_mics_and_transcription() {
    // Initialize Microphones ==========================================================================================
    printf("Initializing microphones...\n");

    for (int i = 0; i < NUM_MICS; i++) {
        init_mic(mic_names[i], &mic_chs[i], SAMPLE_RATE, CHANNELS );
    } 

    // Setup threads ===================================================================================================
    pthread_mutex_t transcribe_mutex = PTHREAD_MUTEX_INITIALIZER;
    pthread_cond_t transcribe_cond = PTHREAD_COND_INITIALIZER;
    transcription_thread_data.recognizer = sherpa_recognizer;
    transcription_thread_data.stream = sherpa_stream;
    transcription_thread_data.cond = &transcribe_cond;
    transcription_thread_data.mutex = &transcribe_mutex;
    transcription_thread_data.ready = 0;

    float mic1_buffer[MIC_BUFFER_LEN] = {0};
    float mic2_buffer[MIC_BUFFER_LEN] = {0};
    pthread_mutex_t mic1_mutex = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_t mic2_mutex = PTHREAD_MUTEX_INITIALIZER;
    pthread_cond_t mic1_cond = PTHREAD_COND_INITIALIZER;
    pthread_cond_t mic2_cond = PTHREAD_COND_INITIALIZER;

    float* mic_buffers[NUM_MICS] = {mic1_buffer, mic2_buffer};
    pthread_mutex_t* mic_mutexes[NUM_MICS] = {&mic1_mutex, &mic2_mutex};
    pthread_cond_t* mic_conds[NUM_MICS] = {&mic1_cond, &mic2_cond};
    float mic_gains[NUM_MICS] = {3.0f, 5.0f};

    for (int i = 0; i < NUM_MICS; i++) {
        mic_thread_data[i].buffer = mic_buffers[i];
        mic_thread_data[i].device = mic_chs[i];
        mic_thread_data[i].data_ready = 0;
        mic_thread_data[i].mutex = mic_mutexes[i];
        mic_thread_data[i].cond = mic_conds[i];
        mic_thread_data[i].gain = mic_gains[i];
    }

    return 0;
}

void* mic_thread(void* ptr) {
    struct mic_thread_data_struct* args = ptr;

    while (run_mic_threads) {
        pthread_mutex_lock(args->mutex);
        while (args->data_ready) {
            pthread_cond_wait(args->cond, args->mutex);
        }
        pthread_mutex_unlock(args->mutex);

        int16_t tmp_buffer[MIC_BUFFER_LEN] = {0};
        read_mic(tmp_buffer, args->device, MIC_BUFFER_LEN);

        pthread_mutex_lock(args->mutex);
	    for (int i = 0; i < MIC_BUFFER_LEN; i++) { 
            float sample = (float)tmp_buffer[i] / 32768.0f * args->gain; 
            if (sample > 1.0f) sample = 1.0f;
            else if (sample < -1.0f) sample = -1.0f;
            args->buffer[i] = sample;
        }
        
        args->data_ready = 1;
        pthread_mutex_unlock(args->mutex);
    }

    return NULL;
}

void* transcription_thread(void* ptr) {
    struct transcription_thread_data_struct* args = ptr;
    const SherpaOnnxOnlineRecognizer* recognizer = args->recognizer;
    const SherpaOnnxOnlineStream* stream = args->stream;

    static float saved_audio[MAX_WAV_TIME * SAMPLE_RATE] = {0};
    static int saved_samples = 0;
    static char wav_filename[64];

    while (run_transcription_thread) {
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

        if (saved_samples == MAX_WAV_TIME * SAMPLE_RATE) {
            make_wav_filename(wav_filename, sizeof(wav_filename));
            write_wav_file(wav_filename, saved_audio, saved_samples, SAMPLE_RATE);
            saved_samples = 0;
            memset(saved_audio, 0, sizeof(saved_audio));
        }

        if (!flush) {
            SherpaOnnxOnlineStreamAcceptWaveform(stream, SAMPLE_RATE, audio, MIC_BUFFER_LEN);
            while (SherpaOnnxIsOnlineStreamReady(recognizer, stream)) {
                SherpaOnnxDecodeOnlineStream(recognizer, stream);
            }

            const SherpaOnnxOnlineRecognizerResult* r = SherpaOnnxGetOnlineStreamResult(recognizer, stream);
            printf("[%.2f] %s\n", chance_of_speech, r->text);

            if (SherpaOnnxOnlineStreamIsEndpoint(recognizer, stream)) {
                SherpaOnnxOnlineStreamReset(recognizer, stream);
            }

            SherpaOnnxDestroyOnlineRecognizerResult(r);
                
            if (MIC_BUFFER_LEN + saved_samples > MAX_WAV_TIME * SAMPLE_RATE) {
                int write_samples = MAX_WAV_TIME * SAMPLE_RATE - saved_samples;
                memcpy(&saved_audio[saved_samples], audio, write_samples * sizeof(float));
                saved_samples += write_samples;
            } else {
                memcpy(&saved_audio[saved_samples], audio, MIC_BUFFER_LEN * sizeof(float));
                saved_samples += MIC_BUFFER_LEN;
            }
        } else {
            const float tail_padding[TAIL_PADDING_LENGTH] = {0};

            SherpaOnnxOnlineStreamAcceptWaveform(stream, SAMPLE_RATE, tail_padding, TAIL_PADDING_LENGTH);
            SherpaOnnxOnlineStreamInputFinished(stream);
            while (SherpaOnnxIsOnlineStreamReady(recognizer, stream)) {
                SherpaOnnxDecodeOnlineStream(recognizer, stream);
            }

            const SherpaOnnxOnlineRecognizerResult* r = SherpaOnnxGetOnlineStreamResult(recognizer, stream);
            printf("[%.2f] %s\n", chance_of_speech, r->text);
           
            make_wav_filename(wav_filename, sizeof(wav_filename));
            write_wav_file(wav_filename, saved_audio, saved_samples, SAMPLE_RATE);
            saved_samples = 0;
            memset(saved_audio, 0, sizeof(saved_audio));

            SherpaOnnxDestroyOnlineRecognizerResult(r);
            SherpaOnnxOnlineStreamReset(recognizer, stream);
        }
    }

    return NULL;
}
