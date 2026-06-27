#include "main.h"

#include <stdio.h>
#include <math.h>
#include <pthread.h>
#include <signal.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include "onnxruntime_c_api.h"
#include "mic_access.h"
#include "sherpa-onnx/c-api/c-api.h"
#include "fft.h"

// =====================================================================================================================
// - 1 Corinthians 10:31
// =====================================================================================================================

// TODO: For communication with the RaspberryPi think about using mosquitto (or another MQTT tool)
// TODO: On the RaspberryPi use pinctrl (or other similar cli) to control hardware states

// TODO: Consider the scenario where a mic is not updating for multiple iterations and how to handle that

int kbhit(void) {
    int k;
    ioctl(STDIN_FILENO, FIONREAD, &k);
    return k;
}

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
    struct mic_thread_data* args = ptr;

    while (run_mic_threads) {
        pthread_mutex_lock(args->mutex);
        while (args->data_ready) {
            pthread_cond_wait(args->cond, args->mutex);
        }
        pthread_mutex_unlock(args->mutex);

        int16_t tmp_buffer[MIC_BUFFER_LEN] = {0};
	    int i = 0;
        read_mic(tmp_buffer, args->device, MIC_BUFFER_LEN);

        pthread_mutex_lock(args->mutex);
	    while (i < MIC_BUFFER_LEN) { args->buffer[i] = (float)tmp_buffer[i] / 32768.0f; i++; }
        args->data_ready = 1;
        pthread_mutex_unlock(args->mutex);
    }

    return NULL;
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

static int findSampleDelay(const complex origin[MIC_BUFFER_LEN], const complex other[MIC_BUFFER_LEN]) {
    // Find sample delay using frequency domain formula
    int i = 0;
    const int n = MIC_BUFFER_LEN;
    int m_delay = 0;

    complex tmp[n], Rxy[n];

    i = 0;
    while (i < n) {
        // compute correlation
        // the total formula is the dot product between x and complex conjugate of y
        //  normalized by the magnitude
        const double a = origin[i].Re;
        const double b = origin[i].Im;
        const double c = other[i].Re;
        const double d = other[i].Im;

        const double m1 = pow(a*c + b*d, 2);
        const double m2 = pow(b*c - a*d, 2);
        const double mag = sqrt(m1 + m2);

        Rxy[i].Re = (a*c + b*d)/mag;
        Rxy[i].Im = (b*c - a*d)/mag;

        i++;
    }

    ifft(Rxy, n, tmp);
    // The index of the max value represents our delay
    i = 0;
    while (i < n) {
        // compare absolute values
        const double a = (Rxy[i].Re < 0.0) ? -Rxy[i].Re : Rxy[i].Re;
        const double b = (Rxy[m_delay].Re < 0.0) ? -Rxy[m_delay].Re : Rxy[m_delay].Re;
        m_delay = (a > b) ? i : m_delay;

        i++;
    }

    return (m_delay > n/2) ? m_delay - n : m_delay;
}

static void* createNode(void* ptr) {
    struct node_thread_data* args = ptr;

    while (run_node_threads) {
        pthread_mutex_lock(args->mutex);
        while (args->data_ready) {
            pthread_cond_wait(args->cond, args->mutex);
        }
        pthread_mutex_unlock(args->mutex);

        int num_ready = 0;
        for (int i = 0; i < args->num_mics; i++) {
            int mi = args->mic_indexes[i];
            pthread_mutex_lock(mic_data[mi].mutex);
            num_ready += mic_data[mi].data_ready;
        }

        if (num_ready != args->num_mics) {
            for (int i = 0; i < args->num_mics; i++) {
                int mi = args->mic_indexes[i];
                pthread_mutex_unlock(mic_data[mi].mutex);
            }
            usleep(1000); // Give the mic threads a chance to update the data
        } else {
            complex other[MIC_BUFFER_LEN];
            complex origin[MIC_BUFFER_LEN];
            complex combined[MIC_BUFFER_LEN];
            complex tmp[MIC_BUFFER_LEN];
            float rms[args->num_mics];

            memset(rms, 0, sizeof(rms));
            // Find the mic with the highest power and set that as the origin
            for (int i = 0; i < args->num_mics; i++) {
                int mi = args->mic_indexes[i];
                for (int j = 0; j < MIC_BUFFER_LEN; j++) {
                    rms[i] += mic_data[mi].buffer[j] * mic_data[mi].buffer[j];
                }
            }
            const int origin_mic = max(rms, args->num_mics);

            fprintf(stdout, "origin_mic = %d\n", origin_mic);

            for (int i = 0; i < MIC_BUFFER_LEN; i++) {
                origin[i].Re = mic_data[origin_mic].buffer[i];
                origin[i].Im = 0.0f;
            }

            fft(origin, MIC_BUFFER_LEN, tmp);
            memcpy(combined, origin, sizeof(origin)); // Start off by setting the combined buffer to the origin

            for (int i = 0; i < args->num_mics; i++) {
                if (i == origin_mic) { continue; }
                int mi = args->mic_indexes[i];

                for (int j = 0; j < MIC_BUFFER_LEN; j++) {
                    other[j].Re = mic_data[mi].buffer[j];
                    other[j].Im = 0.0f;
                }
                fft(other, MIC_BUFFER_LEN, tmp);
                const int delay = findSampleDelay(origin, other);

                // shift the other microphone so it is temporally synced with the origin mic
                for (int k = 0; k < MIC_BUFFER_LEN; k++) {
                    const double shift_angle = -2 * PI * k * delay/MIC_BUFFER_LEN;
                    combined[k].Re += other[k].Re * cos(shift_angle) - other[k].Im * sin(shift_angle);
                    combined[k].Im += other[k].Re * sin(shift_angle) + other[k].Im * cos(shift_angle);
                }
            }

            fprintf(stdout, "Combined all FFT signals\n");

            ifft(combined, MIC_BUFFER_LEN, tmp);
            pthread_mutex_lock(args->mutex);
            memset(args->buffer, 0, MIC_BUFFER_LEN * sizeof(float));
            for (int j = 0; j < MIC_BUFFER_LEN; j++) {
                args->buffer[j] += (float) (combined[j].Re / MIC_BUFFER_LEN / args->num_mics);
            }
            args->data_ready = 1;
            
            fprintf(stdout, "Set data_ready\n");

            for (int i = 0; i < args->num_mics; i++) { 
                int mi = args->mic_indexes[i];
                mic_data[mi].data_ready = 0;
                pthread_cond_signal(mic_data[mi].cond);
                pthread_mutex_unlock(mic_data[mi].mutex);
            }
            pthread_mutex_unlock(args->mutex);
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
    double peak_value = 0.0;
    int iterations_held = 0;
    int iterations_decayed = 0;
    int ret;
    int vad_previously_triggered = 0;

    // Initialize Sherpa models ========================================================================================

    if (initializeSherpa()) { return 1; }

    // Initialize the ORT Api ==========================================================================================

    const OrtApi* ort = initializeORT();
    if (ort == NULL) { goto ort_initialize_fail; }

    // Initialize Microphones ==========================================================================================
    printf("Initializing microphones...\n");

    snd_pcm_t* mic_chs[NUM_MICS];
    for (int i = 0; i < NUM_MICS; i++) {
        init_mic(mic_names[i], &mic_chs[i], SAMPLE_RATE, CHANNELS );
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

    float mic1_buffer[MIC_BUFFER_LEN] = {0};
    float mic2_buffer[MIC_BUFFER_LEN] = {0};
    float mic3_buffer[MIC_BUFFER_LEN] = {0};
    float mic4_buffer[MIC_BUFFER_LEN] = {0};
    float mic5_buffer[MIC_BUFFER_LEN] = {0};
    pthread_mutex_t mic1_mutex = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_t mic2_mutex = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_t mic3_mutex = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_t mic4_mutex = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_t mic5_mutex = PTHREAD_MUTEX_INITIALIZER;
    pthread_cond_t mic1_cond = PTHREAD_COND_INITIALIZER;
    pthread_cond_t mic2_cond = PTHREAD_COND_INITIALIZER;
    pthread_cond_t mic3_cond = PTHREAD_COND_INITIALIZER;
    pthread_cond_t mic4_cond = PTHREAD_COND_INITIALIZER;
    pthread_cond_t mic5_cond = PTHREAD_COND_INITIALIZER;

    float* mic_buffers[NUM_MICS] = {mic1_buffer, mic2_buffer, mic3_buffer, mic4_buffer, mic5_buffer};
    pthread_mutex_t* mic_mutexes[NUM_MICS] = {&mic1_mutex, &mic2_mutex, &mic3_mutex, &mic4_mutex, &mic5_mutex};
    pthread_cond_t* mic_conds[NUM_MICS] = {&mic1_cond, &mic2_cond, &mic3_cond, &mic4_cond, &mic5_cond};

    for (int i = 0; i < NUM_MICS; i++) {
        mic_data[i].buffer = mic_buffers[i];
        mic_data[i].device = mic_chs[i];
        mic_data[i].data_ready = 0;
        mic_data[i].mutex = mic_mutexes[i];
        mic_data[i].cond = mic_conds[i];
    }

    struct node_thread_data node_1;
    int mic_indexes[3] = {2, 3, 4};
    float node1_buffer[MIC_BUFFER_LEN] = {0};
    pthread_mutex_t node_1_mutex = PTHREAD_MUTEX_INITIALIZER;
    pthread_cond_t node_1_cond = PTHREAD_COND_INITIALIZER;
    node_1.mic_indexes = mic_indexes;
    node_1.buffer = node1_buffer;
    node_1.data_ready = 0;
    node_1.num_mics = 3;
    node_1.mutex = &node_1_mutex;
    node_1.cond = &node_1_cond;

    // Create alias for the first two mics to keep node terminology consistent
    struct mic_thread_data* node_2 = &mic_data[0];
    struct mic_thread_data* node_3 = &mic_data[1];

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

        fprintf(stdout, "Started mic %d\n", num_mics_initialized);
        num_mics_initialized++;
    }

    run_node_threads = 1;
    if ((ret = pthread_create(&thread[NODE1_THREAD_ID], NULL, createNode, &node_1)) != 0) {
        fprintf(stderr, "Error: failed to create node 1 thread. Code: %d", ret);
        goto node_1_thread_fail;
    }
    fprintf(stdout, "Started node 1 thread\n");

    if ((ret = pthread_create(&thread[TRANSCRIPT_THREAD_ID], NULL, transcribe, &transcribe_data)) != 0) {
        fprintf(stderr, "Error: failed to create transcription thread. Code: %d", ret);
        goto transcribe_thread_fail;
    }
    fprintf(stdout, "Started transcription thread\n");

    // Main Loop =======================================================================================================
    while (!kbhit()) {
        int hold_iterations = 5;
        const double speech_threshold = 0.7; // trigger threshold to start transcription

        float speech_prob;
        OrtValue** outputs = NULL;

        float rms[3] = {0};

        // Accessing mic data ==========================================================================================

        pthread_mutex_lock(node_1.mutex);
        pthread_mutex_lock(node_2->mutex);
        pthread_mutex_lock(node_3->mutex);
        
        pthread_mutex_lock(mic_data[2].mutex);
        pthread_mutex_lock(mic_data[3].mutex);
        pthread_mutex_lock(mic_data[4].mutex);
        fprintf(stdout, "mic1: %d, mic2: %d, node1: %d, mic3: %d, mic4: %d, mic5: %d\n",
                node_2->data_ready, node_3->data_ready, node_1.data_ready, mic_data[2].data_ready, mic_data[3].data_ready, mic_data[4].data_ready
        );
        pthread_mutex_unlock(mic_data[4].mutex);
        pthread_mutex_unlock(mic_data[3].mutex);
        pthread_mutex_unlock(mic_data[2].mutex);
        
        if (node_1.data_ready && node_2->data_ready && node_3->data_ready) {
            for (int i = 0; i < MIC_BUFFER_LEN; i++) {
                rms[0] += node_1.buffer[i] * node_1.buffer[i];
                rms[1] += node_2->buffer[i] * node_2->buffer[i];
                rms[2] += node_3->buffer[i] * node_3->buffer[i];
            }

            int highest_power = max(rms, 3);
            switch (highest_power) {
                case 0:
                    memcpy(combined_buffer, node_1.buffer, MIC_BUFFER_LEN * sizeof(float));
                    break;
                case 1:
                    memcpy(combined_buffer, node_2->buffer, MIC_BUFFER_LEN * sizeof(float));
                    break;
                case 2:
                    memcpy(combined_buffer, node_3->buffer, MIC_BUFFER_LEN * sizeof(float));
                    break;
                default:
                    break;
            }

            node_1.data_ready = 0;
            node_2->data_ready = 0;
            node_3->data_ready = 0;
            pthread_cond_signal(node_1.cond);
            pthread_cond_signal(node_2->cond);
            pthread_cond_signal(node_3->cond);
            pthread_mutex_unlock(node_1.mutex);
            pthread_mutex_unlock(node_2->mutex);
            pthread_mutex_unlock(node_3->mutex);
        } else {
            // Don't want to run transcription
            pthread_mutex_unlock(node_1.mutex);
            pthread_mutex_unlock(node_2->mutex);
            pthread_mutex_unlock(node_3->mutex);
            usleep(1000);
            continue;
        }

        // =============================================================================================================

		speech_prob = getSpeechProb(&outputs, ort);
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
        
clean_up_loop:
        ort->ReleaseValue(outputs[0]);
        ort->ReleaseValue(vad_state_tensor);
        vad_state_tensor = outputs[1];
        outputs[1] = NULL;
	}
	
// Cleanup for program exit ------------------------------------------------------------------------
node_1_thread_fail:
    run_node_threads = 0;
    pthread_join(thread[NUM_THREADS - 1], NULL);

transcribe_thread_fail:
    pthread_join(thread[NUM_THREADS - 2], NULL);

mic_threads_fail:
    run_mic_threads = 0;
    for (int i = 0; i < num_mics_initialized; i++) {
        fprintf(stdout, "Cleaning up mic %d", i + 1);
        pthread_join(thread[i], NULL);
        close_mic(mic_chs[i]);
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
