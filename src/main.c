#include <Python.h>
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

    // Initialize Python 
    PyStatus status;
    PyConfig config;
    PyConfig_InitPythonConfig(&config);

    status = PyConfig_SetString(&config, &config.pythonpath_env, L"/home/dylenthomas/LiveASRonRPi-4/src:/home/dylenthomas/LiveASRonRPi-4/src/cbuild/venv/lib/python3.14/site-packages");
    if (PyStatus_Exception(status)) { Py_ExitStatusException(status); } 
    status = Py_InitializeFromConfig(&config);
    if (PyStatus_Exception(status)) { Py_ExitStatusException(status); }
    
    PyConfig_Clear(&config);

    //PyRun_SimpleString("import sys; print('sys.path:', sys.path)");

    PyObject* pModule = PyImport_Import(PyUnicode_DecodeFSDefault("transcribe"));
    if (pModule == NULL) {
        if (PyErr_Occurred()) { PyErr_Print(); }
        fprintf(stderr, "ERROR: Failed to import transcripter module.\n");
        return -1;
    }

    PyObject* pTranscribeFunc = PyObject_GetAttrString(pModule, "main");
    if (pTranscribeFunc == NULL) {
        if (PyErr_Occurred()) { PyErr_Print(); }
        fprintf(stderr, "ERROR: Failed to get transcribe function.\n");
        return -1;
    }

	if (argc != 4) { printf("Args should be: VAD Model Path, Mic1 Name, Mic2 Name\n"); return 0; }
	const char* vad_model_path = argv[1];
	const char* mic1_name = argv[2];
    const char* mic2_name = argv[3];

    //struct keywordHM keywords = createKeywordHM(KEYWORD_CONF);

	const float speech_threshold = 0.7; // trigger threshold to start transcription
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
	if (badStatus(ort->CreateEnv(ORT_LOGGING_LEVEL_VERBOSE, "onnxruntime", &env), ort)) { return 1; }

	OrtSessionOptions* session_opts = NULL;
	if (badStatus(ort->CreateSessionOptions(&session_opts), ort)) { return 1; }
	if (badStatus(ort->SetIntraOpNumThreads(session_opts, 1), ort)) { return 1; }
	if (badStatus(ort->SetInterOpNumThreads(session_opts, 1), ort)) { return 1; }
	if (badStatus(ort->SetSessionGraphOptimizationLevel(session_opts, ORT_ENABLE_ALL), ort)) { return 1; }

    OrtCUDAProviderOptionsV2* cuda_opts = NULL;
    if (badStatus(ort->CreateCUDAProviderOptions(&cuda_opts), ort)) { return 1; }
    if (badStatus(ort->SessionOptionsAppendExecutionProvider_CUDA_V2(session_opts, cuda_opts), ort)) { return 1; }

	OrtSession* ort_session = NULL;
	if (badStatus(ort->CreateSession(env, vad_model_path, session_opts, &ort_session), ort)) { return 1; }

	OrtRunOptions* ort_run_opts = NULL;
	if (badStatus(ort->CreateRunOptions(&ort_run_opts), ort)) { return 1; }

    OrtMemoryInfo* gpu_mem_info = NULL;
    if (badStatus(ort->CreateMemoryInfo_V2("Cuda", OrtMemoryInfoDeviceType_GPU, VENDOR_ID, DEVICE_ID,
            OrtMemTypeDefault, GPU_ALIGNMENT, OrtDeviceAllocator, &gpu_mem_info), ort)) { return 1; }

	OrtMemoryInfo* cpu_mem_info = NULL;
	if (badStatus(ort->CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeDefault, &cpu_mem_info), ort)) { return 1; }

	OrtAllocator* alloc = NULL;
	if (badStatus(ort->GetAllocatorWithDefaultOptions(&alloc), ort)) { return 1; }
	
	if (badStatus(ort->CreateTensorWithDataAsOrtValue(
		gpu_mem_info, sample_rate, sizeof(sample_rate), sample_rate_shape, SAMPLE_RATE_DIMS,
		ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &sr_tensor), ort)) { return 1; }

	// create initializing state for model of all zeros 
	if (badStatus(ort->CreateTensorWithDataAsOrtValue(
		gpu_mem_info, initial_state, sizeof(initial_state), state_data_shape, STATE_DIMS,
		ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &state_tensor), ort)) { return 1; }

	if (badStatus(ort->CreateTensorWithDataAsOrtValue(
		gpu_mem_info, combined_buffer, sizeof(combined_buffer), input_data_shape, INPUT_DIMS, 
		ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor), ort)) { return 1; }

    OrtIoBinding* io_binding = NULL;
    if (badStatus(ort->CreateIoBinding(ort_session, &io_binding), ort)) { return 1; }

    if (badStatus(ort->BindInput(io_binding, "input", input_tensor), ort)) { return 1; }
    if (badStatus(ort->BindInput(io_binding, "state", state_tensor), ort)) { return 1; }
    if (badStatus(ort->BindInput(io_binding, "sr", sr_tensor), ort)) { return 1; }
    if (badStatus(ort->BindOutputToDevice(io_binding, "output", cpu_mem_info), ort)) { return 1; }
    if (badStatus(ort->BindOutputToDevice(io_binding, "stateN", cpu_mem_info), ort)) { return 1; }
// -------------------------------------------------------------------------------------------------
// Initialize Microphone ---------------------------------------------------------------------------
    int i;

	printf("Initializing microphones...\n");

	snd_pcm_t* mic1_ch;
	init_mic(mic1_name, &mic1_ch, SAMPLE_RATE, CHANNELS, MIC_BUFFER_LEN);

    snd_pcm_t* mic2_ch;
    init_mic(mic2_name, &mic2_ch, SAMPLE_RATE, CHANNELS, MIC_BUFFER_LEN);

    struct mic_thread_data mic_data[2];

    pthread_mutex_t mutex1 = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_t mutex2 = PTHREAD_MUTEX_INITIALIZER;

    mic_data[0].buffer = mic1_buffer;
    mic_data[0].device = mic1_ch;
    mic_data[0].updated = &updated1;
    mic_data[0].mutex = &mutex1;
    mic_data[1].buffer = mic2_buffer;
    mic_data[1].device = mic2_ch;
    mic_data[1].updated = &updated2;
    mic_data[1].mutex = &mutex2;
// -------------------------------------------------------------------------------------------------
	printf("Starting audio collection.\n");

    pthread_t thread[NUM_THREADS];
    run_mic_threads = 1;

    if ((ret = pthread_create(&thread[0], NULL, readMicData, &mic_data[0])) != 0) { 
        fprintf(stderr, "Error: failed to create thread 1 %d", ret);
        return 1; 
    }
    printf("Started mic1 thread.\n");

    if ((ret = pthread_create(&thread[1], NULL, readMicData, &mic_data[1])) != 0) { 
        fprintf(stderr, "Error: failed to create thread 2 %d", ret);
        return 1; 
    }
    printf("Started mic2 thread.\n");

    mic1_last_updated = *mic_data[0].updated;
    mic2_last_updated = *mic_data[1].updated;
    
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

        pthread_mutex_unlock(mic_data[0].mutex);
        pthread_mutex_unlock(mic_data[1].mutex);

		float speech_prob;
        OrtValue** outputs = NULL;

		speech_prob = getSpeechProb(&outputs, num_outputs, ort, ort_session, ort_run_opts, io_binding, alloc);
        if (speech_prob == -1.0f) { continue; } // if VAD failed just skip the iteration 
        
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
            i = 0;
            while (i < MIC_BUFFER_LEN) { 
                long_buffer[i + MIC_BUFFER_LEN * transcript_buffers] = combined_buffer[i]; 
                i++; 
            }
            transcript_buffers++;
        }
        else if (transcript_buffers) {
            PyObject *pArgs = PyTuple_New(transcript_buffers * MIC_BUFFER_LEN);
            if (pArgs == NULL && PyErr_Occurred()) { PyErr_Print(); continue; }

            i = 0;
            while (i < transcript_buffers * MIC_BUFFER_LEN) {
                PyTuple_SetItem(pArgs, i, PyFloat_FromDouble(long_buffer[i]));
                i++;
            }
            
            PyObject *pCallArgs = PyTuple_Pack(1, pArgs);
            if (pArgs == NULL && PyErr_Occurred()) { PyErr_Print(); continue; }
            PyObject *pResult = PyObject_Call(pTranscribeFunc, pCallArgs, NULL);
            if (pArgs == NULL && PyErr_Occurred()) { PyErr_Print(); continue; }

            Py_DECREF(pArgs);
            Py_DECREF(pCallArgs);
            Py_DECREF(pResult);
            memset(long_buffer, 0, sizeof(long_buffer));
            transcript_buffers = 0;
        }

        // Cleanup iteration
        ort->ReleaseValue(state_tensor);
        ort->ReleaseValue(outputs[0]);
        state_tensor = outputs[1];
        outputs[1] = NULL;

	}
	
// Cleanup for program exit ------------------------------------------------------------------------
    printf("cleaning up..\n");

    run_mic_threads = 0;
    keep_running = 0;

	ort->ReleaseMemoryInfo(cpu_mem_info);
    ort->ReleaseMemoryInfo(gpu_mem_info);
    ort->ReleaseIoBinding(io_binding);
	ort->ReleaseSession(ort_session);
    ort->ReleaseCUDAProviderOptions(cuda_opts);
	ort->ReleaseSessionOptions(session_opts);
	ort->ReleaseEnv(env);
    printf("Released all ORT bindings.\n");

    pthread_join(thread[0], NULL);
    pthread_join(thread[1], NULL);
    printf("Joined threads.\n");

	close_mic(mic1_ch);
    close_mic(mic2_ch);
    printf("Closed mics.\n");

    Py_DECREF(pTranscribeFunc);
    Py_DECREF(pModule);
    Py_Finalize();

	return 0;
}
