from typing import List

import numpy as np
import torch

from libs.python.faster_whisper import WhisperModel
from libs.python.ML_utils import offlineWhisperProcessor, onnxWraper

### start up --------------------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device: {}".format(device))

processor = offlineWhisperProcessor(config_path="configs/preprocessor_config.json",
                                    special_tokens_path="configs/tokenizer_config.json",
                                    vocab_path="configs/vocab.json", device=device
                                    )
whisper_model = WhisperModel("small", device=device, compute_type="float32")
### ----------------------------------------------------------------------------------------

cdef api str transcribe(audio_buffer: List[float]):
    transcription = []
  
    numpy_buffer = np.array(audio_buffer, dtype=np.float32)
    features = processor.extract_features(torch.from_numpy(numpy_buffer).to(device))
    #features = features.cpu()
    segments = whisper_model.transcribe(features, beam_size=5, language="en")

    for segment in segments:
        cleaned_text = segment.text.lstrip()
        print("[%.2fs -> %.2fs]-%s" % (segment.start, segment.end, cleaned_text))
        transcription.append(cleaned_text)

    return transcription
