//
// Created by dylenthomas on 2026-06-03.
//

#include "queue.h"

/**
 * Create new instance of transcript_queue
 *
 * @return new instance of transcript_queue
 */
transcript_queue* initialize() {
   // TODO: For some reason I have heard malloc is bad, check that
   // TODO: Check if this is the right way to instantiate
   transcript_queue* self = malloc(sizeof(transcript_queue));
   self->backingArray = malloc(sizeof(float**) * STARTING_LENGTH);
   self->queueSize = 0;
   self->backingArraySize = STARTING_LENGTH;

   return self;
}

/**
 * Get the next item in the queue and remove it
 *
 * @param self Current instance of the queue
 * @return Audio buffer at the front of the queue
 */
float* retrieveFront(transcript_queue* self) {
   float* front = self->backingArray[0];
   int newSize = self->queueSize;

   if (newSize < self->backingArraySize / 2 && self->backingArraySize > STARTING_LENGTH) { newSize /= 2; }
   newSize -= 1; // Account for removed item

   float* tmp[newSize];
   // TODO: Check if this is the right way to skip the first item
   memcpy(tmp, (char*)self->backingArray + sizeof(float*), newSize * sizeof(float*));

   self->backingArray = tmp;
   self->queueSize--;

   return front;
}

/**
 * Add an audio buffer to the back of the queue
 *
 * @param self Current instance of the queue
 * @param buffer Audio buffer to add to the queue
 */
void addToQueue(transcript_queue* self, float* buffer) {
   if (self->queueSize + 1 >= self->backingArraySize) {
      int newSize = self->queueSize * 2;
      newSize += 1; // Account for new item

      float* tmp[newSize];
      memcpy(tmp, self->backingArray, (newSize) * sizeof(float*));
      tmp[self->queueSize] = buffer;

      self->queueSize++;
      self->backingArray = tmp;
   } else {
      self->backingArray[self->queueSize] = buffer;
      self->queueSize++;
   }
}