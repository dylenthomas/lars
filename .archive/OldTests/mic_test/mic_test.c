#include <stdio.h>
#include <signal.h>
#include <stdlib.h>

#include "mic_access.h"

snd_pcm_t* mic1_ch;

void sigint_handler(int sig) {
    printf("\nCtrl+C pressed! Cleaning up...\n");
	close_mic(mic1_ch);
    exit(0);
}

int main(int argc, char *argv[]) {
	if (argc != 2) { printf("There should be one argument, the name of the microphone.\n"); return 0;}
	const char* mic1_name = argv[1];

	signal(SIGINT, sigint_handler);

	int16_t buffer[16000 * 3] = {0};

	printf("Initializing microphone...\n");
	init_mic(mic1_name, &mic1_ch, 16000, 1, 512);

	printf("Collecting audio...\n");
	int i = 0;
	while (i < (16000 * 3)/ 512) {
		int16_t chunk[512] = {0};
		read_mic(chunk, mic1_ch, 512);			

		int x = 0;
		while (x < 512) {
			buffer[(512 * i) + x] = chunk[x];
			x++;
		}
		printf("%d\n", i);
		i++;
	}

	printf("Writing file\n");
    FILE *file = fopen("output.wav", "wb");

    // Constants for 16,000 samples at 16-bit
    uint32_t numSamples = 16000 * 3;
    uint32_t dataSize = numSamples * 2; // 2 bytes per sample
    uint32_t totalChunkSize = 36 + dataSize;

    // RIFF header
    fwrite("RIFF", 4, 1, file);
    fwrite(&totalChunkSize, 4, 1, file);
    fwrite("WAVE", 4, 1, file);

    // fmt chunk
    fwrite("fmt ", 4, 1, file);
    uint32_t fmtSize = 16;
    fwrite(&fmtSize, 4, 1, file);
    uint16_t audioFormat = 1; // PCM
    fwrite(&audioFormat, 2, 1, file);
    uint16_t numChannels = 1;
    fwrite(&numChannels, 2, 1, file);
    uint32_t sampleRate = 16000;
    fwrite(&sampleRate, 4, 1, file);
    uint32_t byteRate = sampleRate * 2; 
    fwrite(&byteRate, 4, 1, file);
    uint16_t blockAlign = 2;
    fwrite(&blockAlign, 2, 1, file);
    uint16_t bitsPerSample = 16;
    fwrite(&bitsPerSample, 2, 1, file);

    // data chunk
    fwrite("data", 4, 1, file);
    fwrite(&dataSize, 4, 1, file);

    // Convert float buffer to 16-bit PCM and write
    for (int j = 0; j < numSamples; j++) {
        // Clamp and scale float (-1.0 to 1.0) to short (-32768 to 32767)
        int sample = buffer[j];
        //if (sample > 1.0f) sample = 1.0f;
        //if (sample < -1.0f) sample = -1.0f;
        
        //int16_t pcm_sample = (int16_t)(sample * 32767.0f);
        fwrite(&sample, 2, 1, file);
    }

    fclose(file);

	close_mic(mic1_ch);

	return 0;
}
