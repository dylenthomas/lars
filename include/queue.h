#ifndef LARS_QUEUE_H
#define LARS_QUEUE_H

#include <stdio.h>
#include <stdlib.h>

#define MAX_TRANSCRIPT_QUEUE_LENGTH 10

typedef struct TranscriptQueue {
    // Array will store pointers to audio buffers that are ready to be transcribed
    float** backingArray;
    int queueSize;

    float* (*retrieveFront)(struct TranscriptQueue* self);
    void (*addToQueue)(struct TranscriptQueue* self, float* buffer);
} transcript_queue;

float* retrieveFront(transcript_queue* self);
void addToQueue(transcript_queue* self, float* buffer);

transcript_queue* initialize();
void destroy(transcript_queue* self);

#endif //LARS_QUEUE_H
