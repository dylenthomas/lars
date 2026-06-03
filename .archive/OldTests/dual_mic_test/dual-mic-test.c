#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <time.h>
#include <signal.h>
#include <stdatomic.h>

// Reference: https://wiki.sei.cmu.edu/confluence/spaces/c/pages/87152076/CON43-C.+Do+not+allow+data+races+in+multithreaded+code

#include "mic_access.h"
#include "fft.h"
#include "wav_writer.h"

#define MIC_BUFFER_LEN 1024 
#define SAMPLE_RATE 16000
#define CHANNELS 1
#define LONGEST_TRANSCRIPT 30
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

void* readMicData(void* ptr) {
    printf("Starting mic thread.\n");

    struct mic_thread_data* args = (struct mic_thread_data*)ptr;

    while (run_mic_threads) {
	    int16_t tmp_buffer[MIC_BUFFER_LEN] = {0};
	    int i = 0;
	
        read_mic(tmp_buffer, args->device, MIC_BUFFER_LEN);

        // convert int mic data to float
        pthread_mutex_lock(args->mutex);
	    while (i < MIC_BUFFER_LEN) { args->buffer[i] = (float)tmp_buffer[i] / 32768.0f; i++; }
        pthread_mutex_unlock(args->mutex);

        // tell the main loop the mic buffer is updated by changing updated to random number
        // want to avoid race condition of both threads updating value at the same time
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

	if (argc != 3) { printf("Args should be: Mic1 Name, Mic2 Name\n"); return 0; }
	const char* mic1_name = argv[1];
    const char* mic2_name = argv[2];

    double peak_value = 0.0f;
    int hold_iterations = 5;
    int iterations_held = 0;
    double decay_rate = 0.25f;
    int iterations_decayed = 0;

    atomic_long updated1 = ATOMIC_VAR_INIT(0);
    atomic_long updated2 = ATOMIC_VAR_INIT(0);
    float mic1_buffer[MIC_BUFFER_LEN] = {0};
    float mic2_buffer[MIC_BUFFER_LEN] = {0};
    float combined_buffer[MIC_BUFFER_LEN] = {0};

    int ret;
    long int mic1_last_updated;
    long int mic2_last_updated;

    complex X[MIC_BUFFER_LEN], Y[MIC_BUFFER_LEN], tmp[MIC_BUFFER_LEN], combined[MIC_BUFFER_LEN];

// Initialize WAV output ---------------------------------------------------------------------------
    // NEW: build a timestamped filename and open the WAV writer
    char wav_filename[64];
    time_t now = time(NULL);
    strftime(wav_filename, sizeof(wav_filename), "recording_%Y%m%d_%H%M%S.wav", localtime(&now));

    WavWriter wav;
    if (wav_open(&wav, wav_filename, SAMPLE_RATE, CHANNELS) != 0) {
        fprintf(stderr, "Error: could not open WAV file %s for writing\n", wav_filename);
        return 1;
    }
    printf("Recording to %s\n", wav_filename);
// -------------------------------------------------------------------------------------------------

// Initialize Microphone ---------------------------------------------------------------------------
    int i;

	printf("Initializing microphones...\n");

	snd_pcm_t* mic1_ch;
	init_mic(mic1_name, &mic1_ch, SAMPLE_RATE, CHANNELS, MIC_BUFFER_LEN);

    snd_pcm_t* mic2_ch;
    init_mic(mic2_name, &mic2_ch, SAMPLE_RATE, CHANNELS, MIC_BUFFER_LEN);

    printf("Initialized both microphones\n");

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

    if ((ret = pthread_create(&thread[1], NULL, readMicData, &mic_data[1])) != 0) { 
        fprintf(stderr, "Error: failed to create thread 2 %d", ret);
        return 1; 
    }

    mic1_last_updated = atomic_load(mic_data[0].updated);
    mic2_last_updated = atomic_load(mic_data[1].updated);

    while (keep_running) {
        // wait until both buffers are populated
        if (mic1_last_updated == atomic_load(mic_data[0].updated)) { continue; }
        mic1_last_updated = atomic_load(mic_data[0].updated);

        if (mic2_last_updated == atomic_load(mic_data[1].updated)) { continue; }
        mic2_last_updated = atomic_load(mic_data[1].updated);

        // get the delay of mic2 relative to mic1
        pthread_mutex_lock(mic_data[0].mutex);
        pthread_mutex_lock(mic_data[1].mutex);

        for (int i = 0; i < MIC_BUFFER_LEN; i++) {
            X[i].Re = mic_data[0].buffer[i]; X[i].Im = 0.0f;
            Y[i].Re = mic_data[1].buffer[i]; Y[i].Im = 0.0f;
        }

        int delay = findSampleDelay(mic_data[0].buffer, mic_data[1].buffer);
        printf("delay %d\n", delay);

        fft(X, MIC_BUFFER_LEN, tmp);
        fft(Y, MIC_BUFFER_LEN, tmp);


        for (int f = 0; f < MIC_BUFFER_LEN; f++) {
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

        wav_write_float(&wav, combined_buffer, MIC_BUFFER_LEN);
	}
	
// Cleanup for program exit ------------------------------------------------------------------------
    printf("cleaning up..\n");

    run_mic_threads = 0;
    pthread_join(thread[0], NULL);
    pthread_join(thread[1], NULL);
    close_mic(mic1_ch);
    close_mic(mic2_ch);

    // NEW: finalise and close the WAV file (patches the header with correct sizes)
    wav_close(&wav);
    printf("Saved recording to %s\n", wav_filename);
	
    return 0;
}