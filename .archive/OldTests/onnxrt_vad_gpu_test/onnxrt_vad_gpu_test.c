#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

#include "onnxruntime_c_api.h"
#include "mic_access.h"
//#include "cuda_provider_factory.h"

#define VENDOR_ID 4318 
#define DEVICE_ID 0
#define GPU_ALIGNMENT 0

int badStatus(OrtStatus* status, const OrtApi* ort) {
	// Make sure the Api was accessed correctly 
	if (status != NULL) {
		const char* err_msg = ort->GetErrorMessage(status);
		fprintf(stderr, "ORT Error: %s\n", err_msg);
		ort->ReleaseStatus(status);
		return 1;
	}
	return 0;
}

double timedifference_msec(struct timeval t0, struct timeval t1) {
    return (t1.tv_sec - t0.tv_sec) * 1000.0f + (t1.tv_usec - t0.tv_usec) / 1000.0f;
}

int main(int argc, char *argv[]) {
	if (argc != 3) { printf("There should be only two args, VAD model path then mic name.\n"); return 0; }
	const char* vad_model_path = argv[1];
	const char* mic1_name = argv[2];

    struct timeval t0, t1;

	int64_t sample_rate[] = {16000};
	const int64_t sample_rate_shape[] = {1};

	int64_t input_data_shape[] = {1, 512}; // input data will be the mic buffer
	int64_t state_data_shape[] = {2, 1, 128};	

	int16_t tmp_buffer[512] = {0};
	float buffer[512] = {0};
	float initial_state[2 * 1 * 128] = {0};

	//const char* input_names[] = {"input", "state", "sr"};
	//const char* const* input_names = &input_names_arr;
	//const char* output_names[] = {"output", "stateN"};
	//const char* const* output_names = &output_names_arr;
	
	OrtValue* input_tensor = NULL;
	OrtValue* state_tensor = NULL;
	OrtValue* sr_tensor = NULL;
// Initialize the ORT Api --------------------------------------------------------------------------
	printf("Initializing ORT...\n");
	const OrtApi* ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
	if (ort == NULL) { fprintf(stderr, "ORT api returned nullptr!\n"); return 1; }

	OrtEnv* env = NULL;
	if (badStatus(ort->CreateEnv(ORT_LOGGING_LEVEL_VERBOSE, "onnxruntime", &env), ort)) { return 1; }
    printf("Created Environment\n");

	OrtSessionOptions* session_opts = NULL;
	if (badStatus(ort->CreateSessionOptions(&session_opts), ort)) { return 1; }
	if (badStatus(ort->SetIntraOpNumThreads(session_opts, 1), ort)) { return 1; }
	if (badStatus(ort->SetInterOpNumThreads(session_opts, 1), ort)) { return 1; }
	if (badStatus(ort->SetSessionGraphOptimizationLevel(session_opts, ORT_ENABLE_ALL), ort)) { return 1; }
    if (badStatus(ort->SetSessionLogSeverityLevel(session_opts, 0), ort)) { return 1; } // 0 is Verbose
    printf("Created Session Options\n");

    OrtCUDAProviderOptionsV2* cuda_opts = NULL;
    if (badStatus(ort->CreateCUDAProviderOptions(&cuda_opts), ort)) { return 1; }
    if (badStatus(ort->SessionOptionsAppendExecutionProvider_CUDA_V2(session_opts, cuda_opts), ort)) { return 1; }
    printf("Appended Cuda Provider Options\n");

	OrtSession* session = NULL;
	if (badStatus(ort->CreateSession(env, vad_model_path, session_opts, &session), ort)) { return 1; }
    printf("Created Session\n");

	OrtRunOptions* ort_run_opts = NULL;
	if (badStatus(ort->CreateRunOptions(&ort_run_opts), ort)) { return 1; }
    printf("Created Run Options\n");

	OrtMemoryInfo* gpu_mem_info = NULL;
	if (badStatus(ort->CreateMemoryInfo_V2("Cuda", OrtMemoryInfoDeviceType_GPU, VENDOR_ID, DEVICE_ID, 
                OrtMemTypeDefault, GPU_ALIGNMENT, OrtDeviceAllocator, &gpu_mem_info), ort)) { return 1; }
    printf("Created GPU Memory Info\n");

    OrtMemoryInfo* cpu_mem_info = NULL;
	if (badStatus(ort->CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeDefault, &cpu_mem_info), ort)) { return 1; }
    printf("Created CPU Memory Info\n");

	OrtAllocator* alloc = NULL;
	if (badStatus(ort->GetAllocatorWithDefaultOptions(&alloc), ort)) { return 1; }
    printf("Created memory allocator\n");
	
	if (badStatus(ort->CreateTensorWithDataAsOrtValue(
		gpu_mem_info, sample_rate, sizeof(sample_rate), sample_rate_shape, 1,
		ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &sr_tensor), ort)) { return 1; }
    printf("Created sample rate tensor\n");

	// create initializing state for model of all zeros 
	if (badStatus(ort->CreateTensorWithDataAsOrtValue(
		gpu_mem_info, initial_state, sizeof(initial_state), state_data_shape, 3,
		ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &state_tensor), ort)) { return 1; }
    printf("Created state tensor\n");

	if (badStatus(ort->CreateTensorWithDataAsOrtValue(
		gpu_mem_info, buffer, sizeof(buffer), input_data_shape, 2, 
		ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor), ort)) { return 1; }
    printf("Created input tensor\n");

    OrtIoBinding* io_binding = NULL;
    if (badStatus(ort->CreateIoBinding(session, &io_binding), ort)) { return 1; }
    printf("Created IO Binding\n");

    if (badStatus(ort->BindInput(io_binding, "input", input_tensor), ort)) { return 1; }
    printf("input is bound\n");
    if (badStatus(ort->BindInput(io_binding, "state", state_tensor), ort)) { return 1; }
    printf("state is bound\n");
    if (badStatus(ort->BindInput(io_binding, "sr", sr_tensor), ort)) { return 1; }
    printf("sample rate is bound\n");
    if (badStatus(ort->BindOutputToDevice(io_binding, "output", cpu_mem_info), ort)) { return 1; }
    printf("output is bound\n");
    if (badStatus(ort->BindOutputToDevice(io_binding, "stateN", cpu_mem_info), ort)) { return 1; }
    printf("state is bound\n");

// -------------------------------------------------------------------------------------------------
// Initialize Microphone ---------------------------------------------------------------------------
	printf("Initializing microphone...\n");
	snd_pcm_t* mic1_ch;
	
	init_mic(mic1_name, &mic1_ch, sample_rate[0], 1, 512);
// -------------------------------------------------------------------------------------------------
	printf("Starting audio collection.\n");
    double peak_value = 0.0f;
    int hold_iterations = 5;
    int iterations_held = 0;
    double decay_rate = 0.25f;
    int decay_iterations = 0;

	while (1) {
        gettimeofday(&t0, NULL); 
        size_t output_count = 2;
		OrtValue** outputs = NULL;
		read_mic(tmp_buffer, mic1_ch, 512); // read 512 buffer samples

		int i = 0;
		while (i < 512) { buffer[i] = (float)tmp_buffer[i] / 32768.0f; i++; }
       
        if (badStatus(ort->RunWithBinding(session, ort_run_opts, io_binding), ort)) { return 1; } 
        if (badStatus(ort->GetBoundOutputValues(io_binding, alloc, &outputs, &output_count), ort)) { return 1; }

		// Retrieve probability of speech from the model
		OrtValue* ort_speech_prob = outputs[0];
		if (ort_speech_prob == NULL) {
			fprintf(stderr, "Error: Ouput tensor is NULL\n");
			return 1;
		}
		float* speech_prob = NULL;
		if (badStatus(ort->GetTensorMutableData(ort_speech_prob, (void**)&speech_prob), ort)) { return 1; }

		// release old OrtValues
		ort->ReleaseValue(state_tensor);
		ort->ReleaseValue(outputs[0]);
		// Save the previous state for the next run.
		state_tensor = outputs[1];
		outputs[1] = NULL;

        if (speech_prob[0] > peak_value) { // Increase
            peak_value = speech_prob[0];
            decay_iterations = 0; 
        }
        else if (iterations_held <= hold_iterations) { // Hold
            iterations_held++; 
        }
        else { // Decay
            iterations_held = 0;
            peak_value *= exp(-1 * decay_rate * decay_iterations);
            decay_iterations++;
        }

        gettimeofday(&t1, NULL);
		printf("\033[2J\033[H");
		printf("[%.3f] The probability of speech is: %f\n", timedifference_msec(t0, t1), peak_value); 	
    }
	
// Cleanup for program exit ------------------------------------------------------------------------
	ort->ReleaseMemoryInfo(cpu_mem_info);
    ort->ReleaseMemoryInfo(gpu_mem_info);
    ort->ReleaseIoBinding(io_binding);
	ort->ReleaseSession(session);
    ort->ReleaseCUDAProviderOptions(cuda_opts);
	ort->ReleaseSessionOptions(session_opts);
    ort->ReleaseIoBinding(io_binding);
	ort->ReleaseEnv(env);

	close_mic(mic1_ch);

	return 0;
}
