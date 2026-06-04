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
   transcript_queue* self = malloc(sizeof(transcript_queue));
   if (!self) { return NULL; }

   self->backingArray = malloc(sizeof(float*) * MAX_TRANSCRIPT_QUEUE_LENGTH);
   if (!self->backingArray) {
      free(self);
      return NULL;
   }

   self->queueSize = 0;

   return self;
}

/**
 * Destroy the transcript queue
 *
 * @param self current instance of the transcript_queue
 */
void destroy(transcript_queue* self) {
   if (!self) { return; }
   free(self->backingArray);
   free(self);
}

/**
 * Get the next item in the queue and remove it
 *
 * @param self Current instance of the queue
 * @return Audio buffer at the front of the queue
 */
float* retrieveFront(transcript_queue* self) {
   float* front = self->backingArray[0];

   // Shift all elements down one
   memmove(&self->backingArray[0], &self->backingArray[1], (self->queueSize - 1) * sizeof(float*));

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
   if (self->queueSize + 1 > MAX_TRANSCRIPT_QUEUE_LENGTH) {
      // TODO: Setup error state here
   }
   self->backingArray[self->queueSize] = buffer;
   self->queueSize++;
}