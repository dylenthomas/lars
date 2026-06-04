//
// Created by dylenthomas on 2026-06-03.
//

#ifndef LARS_QUEUE_H
#define LARS_QUEUE_H

#include <stdio.h>
#include <stdlib.h>

#include "main.h"

#define STARTING_LENGTH 10

typedef struct TranscriptQueue {
    // Array will store pointers to audio buffers that are ready to be transcribed
    float** backingArray;
    int queueSize;
    int backingArraySize;

    float* (*retrieveFront)(struct TranscriptQueue* self);
    void (*addToQueue)(struct TranscriptQueue* self, float* buffer);
} transcript_queue;

float* retrieveFront(transcript_queue* self);
void addToQueue(transcript_queue* self, float* buffer);

transcript_queue* initialize();

#endif //LARS_QUEUE_H
