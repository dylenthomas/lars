import numpy as np
import os
import wave
import torch
import threading
import re
import datetime

from utils.faster_whisper import WhisperModel
from utils.ML_utils import offlineWhisperProcessor, onnxWraper
from utils.LARS_utils import TCPCommunication
from ctypes import * 
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from sentence_transformers import SentenceTransformer, util

### setup relevant helper classes ------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device: {}".format(device))

print("Starting whisper processor...")
processor = offlineWhisperProcessor(config_path="utils/configs/preprocessor_config.json",
                                    special_tokens_path="utils/configs/tokenizer_config.json",
                                    vocab_path="utils/configs/vocab.json", device=device
                                    )
print("Starting sentence transformer...")
word_vector_generator = SentenceTransformer("all-MiniLM-L6-v2")
word_vector_generator.to(device)
print("Starting VAD model...")
vad_model = onnxWraper(".model/silero_vad_16k_op15.onnx", force_cpu=False)
print("Starting Whisper...")
model = WhisperModel("small", device=device, compute_type="float32")
tcpCommunicator = TCPCommunication()
### -------------------------------------------------------------------------------------------------

### define c++ functions ----------------------------------------------------------------------------
clib_mic1 = CDLL("utils/mic1.so")
clib_mic2 = CDLL("utils/mic2.so")

clib_mic1.reset_vad.argtypes = None
clib_mic1.reset_vad.restype = None
clib_mic1.freeBuffer.argtypes = [POINTER(c_float)]
clib_mic1.freeBuffer.restype = None
clib_mic1.init_mic.argtypes = [c_char_p]
clib_mic1.init_mic.restype = None
clib_mic1.close_mic.argtypes = None
clib_mic1.close_mic.restype = None
clib_mic1.get_speech.argtypes = [c_int, POINTER(c_int), c_float]
clib_mic1.get_speech.restype = [POINTER(c_float)]

clib_mic2.reset_vad.argtypes = None
clib_mic2.reset_vad.restype = None
clib_mic2.freeBuffer.argtypes = [POINTER(c_float)]
clib_mic2.freeBuffer.restype = None
clib_mic2.init_mic.argtypes = [c_char_p]
clib_mic2.init_mic.restype = None
clib_mic2.close_mic.argtypes = None
clib_mic2.close_mic.restype = None
clib_mic2.get_speech.argtypes = [c_int, POINTER(c_int), c_float]
clib_mic2.get_speech.restype = [POINTER(c_float)]
### -------------------------------------------------------------------------------------------------

### SET VARIABLES -------------------------------------------------------------------------
running = True

mic_name1 = b"plughw:CARD=Snowball"
mic_name2 = b"plughw:CARD=Snowball_1"

prediction_que = []
whisper_max_buffer = 30

threadpool_workers = 10

mics = [[], []]

VA_threshold = 0.7

ONE = ["lights", "mute", "unmute"]
TWO = ["lights on", "lights off", "volume down", "volume up"]
THREE = ["overhead lamp off","overhead lamp on", "desk lights off", "desk lights on", "set aux audio", "set phono audio"]
keywords = [THREE, TWO, ONE]
keyword_embeddings = np.concatenate([word_vector_generator.encode(keyword_group) for keyword_group in keywords])
flattened_keywords = [x for sub in keywords for x in sub]
vector_similarity_thres = 0.8
### ----------------------------------------------------------------------------------------

def audio_loop1():
    print("Starting mic1...")
    buffer = np.zeros(whisper_max_buffer * sample_rate, dtype=np.float32)
    collected_samples = c_int()

    while running:
        prt = clib_mic1.get_speech(whisper_max_buffer, byref(collected_samples), VA_threshold)
       
        buffer = buffer[:collected_samples.value]
        buffer = np.ctypeslib.as_array(ptr, (collected_samples.value,))
        clib_mic.freeBuffer(ptr)

        mics[0].append(buffer)

        clib_mic1.reset_vad()

    clib_mic1.close_mic()

def audio_loop2():
    print("Starting mic2...")
    buffer = np.zeros(whisper_max_buffer * sample_rate, dtype=np.float32)
    collected_samples = c_int()

    while running:
        prt = clib_mic2.get_speech(whisper_max_buffer, byref(collected_samples), VA_threshold)
       
        buffer = buffer[:collected_samples.value]
        buffer = np.ctypeslib.as_array(ptr, (collected_samples.value,))
        clib_mic.freeBuffer(ptr)

        mics[1].append(buffer)

        clib_mic2.reset_vad()

    clib_mic2.close_mic()

def transcribe(prediction_que):
    transcription = []
    
    pred_array = np.concatenate(prediction_que)
    que_len = len(prediction_que)
    features = processor.extract_features(torch.from_numpy(pred_array).to(device))
    features = features.cpu()
    segments = model.transcribe(features, beam_size=5, language="en")

    for segment in segments:
        cleaned_text = segment.text.lstrip()
        print("[%.2fs -> %.2fs]-%s" % (segment.start, segment.end, cleaned_text))
        transcription.append(cleaned_text)

    return transcription

def parse_transcript(transcript):
    packet = ''

    transcript = [" ".join(transcript)]
    transcript_embedding = word_vector_generator.encode(transcript)
    transcript_embedding = np.transpose(transcript_embedding)

    similarities = np.matmul(keyword_embeddings, transcript_embedding)
    max_similarity_ind = np.argmax(similarities).item()
    max_similarity = similarities[max_similarity_ind]

    if max_similarity > vector_similarity_thres:
        keyword = flattened_keywords[max_similarity_ind]
        packet += f'{keyword},'

    return packet

def save_audio(audio, name):
    window = audio * (2**15)
    window = window.astype(dtype=np.int16)
    wav = wave.open("{}.wav".format(name), "wb")
    wav.setnchannels(1)
    wav.setsampwidth(2)  # 16-bit
    wav.setframerate(16000)
    wav.writeframes(window)
    wav.close()

if __name__ == "__main__":    
    print("Waiting to connect to end device...")
    #tcpCommunicator.connectClient()
    print("Connected to end device.")

    executor = ThreadPoolExecutor(max_workers=threadpool_workers)

    audio_thread1 = threading.Thread(target=audio_loop1)
    audio_thread1.start()

    audio_thread2 = threading.Thread(target=audio_loop2)
    audio_thread2.start()

    try:
        while running:
            if len(mics[0]) == 0 and len(mics[1]) == 0:
                continue

            elif len(mics[0]) != 0 and len(mics[1]) == 0:
                mic1_buffer = mics[0][0]
                prediction_que.append(mic1_buffer)

                del mics[0][0]

            elif len(mics[0]) == 0 and len(mics[1]) != 0:
                mic2_buffer = mics[1][0]
                prediction_que.append(mic2_buffer)

                del mics[1][0]

            else:
                mic1_buffer = mics[0][0]
                mic2_buffer = mics[1][0]

                correlation = np.correlate(mic1_buffer, mic2_buffer, 'full')
                delay = np.argmax(correlation) - (len(mic2_buffer) - 1)
                if delay > 0:
                    mic2_buffer = np.concatenate([np.zeros(delay), mic2_buffer])
                else:
                    mic2_buffer = mic2_buffer[-delay:]

                prediction_que.append(0.5 * (mic1_buffer + mic2_buffer))

                del mics[0][0]
                del mics[1][0]

            if len(prediction_que) > 0:
                transcript = transcribe(prediction_que[0])
                packet = parse_transcript(transcript)
                executor.submit(tcpCommunicator.sendToServer, packet)

                del prediction_que[0]

    except KeyboardInterrupt:
        print("\nStopping...")
        running = False
        #executor.shutdown(cancel_futures=True)
        audio_thread1.join()
        audio_thread2.join()
        tcpCommunicator.closeClientConnection()