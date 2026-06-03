#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include "onnxruntime_c_api.h"
#include "mic_access.h"

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

	const char* input_names[] = {"input", "state", "sr"};
	//const char* const* input_names = &input_names_arr;
	const char* output_names[] = {"output", "stateN"};
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

	OrtSessionOptions* session_opts = NULL;
	if (badStatus(ort->CreateSessionOptions(&session_opts), ort)) { return 1; }
	if (badStatus(ort->SetIntraOpNumThreads(session_opts, 5), ort)) { return 1; }
	if (badStatus(ort->SetInterOpNumThreads(session_opts, 5), ort)) { return 1; }
	if (badStatus(ort->SetSessionGraphOptimizationLevel(session_opts, ORT_ENABLE_ALL), ort)) { return 1; }

	OrtSession* session = NULL;
	if (badStatus(ort->CreateSession(env, vad_model_path, session_opts, &session), ort)) { return 1; }

	OrtRunOptions* ort_run_opts = NULL;
	if (badStatus(ort->CreateRunOptions(&ort_run_opts), ort)) { return 1; }

	OrtMemoryInfo* mem_info = NULL;
	if (badStatus(ort->CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeDefault, &mem_info), ort)) { return 1; }

	OrtAllocator* alloc = NULL;
	if (badStatus(ort->GetAllocatorWithDefaultOptions(&alloc), ort)) { return 1; }
	
	if (badStatus(ort->CreateTensorWithDataAsOrtValue(
		mem_info, sample_rate, sizeof(sample_rate), sample_rate_shape, 1,
		ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &sr_tensor), ort)) { return 1; }

	// create initializing state for model of all zeros 
	if (badStatus(ort->CreateTensorWithDataAsOrtValue(
		mem_info, initial_state, sizeof(initial_state), state_data_shape, 3,
		ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &state_tensor), ort)) { return 1; }

	if (badStatus(ort->CreateTensorWithDataAsOrtValue(
		mem_info, buffer, sizeof(buffer), input_data_shape, 2, 
		ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor), ort)) { return 1; }
	
// -------------------------------------------------------------------------------------------------
// Initialize Microphone ---------------------------------------------------------------------------
	printf("Initializing microphone...\n");
	snd_pcm_t* mic1_ch;
	
	init_mic(mic1_name, &mic1_ch, sample_rate[0], 1, 512);
// -------------------------------------------------------------------------------------------------
	printf("Starting audio collection.\n");
	//uint64_t start = get_current_time_ms();
	while (1) {
        gettimeofday(&t0, NULL); 
		OrtValue* outputs[2] = {NULL, NULL};
		read_mic(tmp_buffer, mic1_ch, 512); // read 512 buffer samples

		int i = 0;
		while (i < 512) { buffer[i] = (float)tmp_buffer[i] / 32768.0f; i++; }

		const OrtValue* const inputs[3] = {input_tensor, state_tensor, sr_tensor};

		if (badStatus(ort->Run(
			session, ort_run_opts, input_names, inputs, 3,
			output_names, 2, outputs), ort)) { return 1; }

		// Retrieve probability of speech from the model
		OrtValue* ort_speech_prob = outputs[0];
		float* prob_data = NULL;
		if (badStatus(ort->GetTensorMutableData(ort_speech_prob, (void**)&prob_data), ort)) { return 1; }
		float speech_prob = prob_data[0];

		// release old OrtValues
		ort->ReleaseValue(state_tensor);
		ort->ReleaseValue(outputs[0]);
		// Save the previous state for the next run.
		state_tensor = outputs[1];
		outputs[1] = NULL;

        gettimeofday(&t1, NULL);
		printf("\033[2J\033[H");
		printf("[%f] The probability of speech is: %f\n", timedifference_msec(t0, t1) * 1000, speech_prob); 	
    }
	
// Cleanup for program exit ------------------------------------------------------------------------
	ort->ReleaseMemoryInfo(mem_info);
	ort->ReleaseSession(session);
	ort->ReleaseSessionOptions(session_opts);
	ort->ReleaseEnv(env);

	close_mic(mic1_ch);

	return 0;
}
