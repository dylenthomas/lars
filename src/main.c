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

// TODO: For communication with the RaspberryPi think about using mosquitto (or another MQTT tool)
// TODO: On the RaspberryPi use pinctrl (or other similar cli) to control hardware states

// TODO: Consider the scenario where a mic is not updating for multiple iterations and how to handle that

int main() {
    double peak_value = 0.0;
    int iterations_held = 0;
    int iterations_decayed = 0;
    int ret;
    int vad_previously_triggered = 0;

    // Initialize the ORT Api ==========================================================================================
    const OrtApi* ort = initializeORT();
    if (ort == NULL) { return -1; }

    const int a_and_t_status = initialize_audio_and_transcription();
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
