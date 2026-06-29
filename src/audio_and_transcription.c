#include "audio_and_transcription.h"

int initialize_audio_and_transcription() {
   // Initialize Sherpa models ========================================================================================
    if (initializeSherpa() < 0) { return -1; }

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

const OrtApi* initializeORT() {
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
