import numpy as np
import torch
from typing import List

from python_utils.faster_whisper import WhisperModel
from python_utils.ML_utils import offlineWhisperProcessor

#device = "cuda" if torch.cuda.is_available() else "cpu"
#print("Using device: {}".format(device))

print("Starting Whisper processor...")
#processor = offlineWhisperProcessor(config_path="utils/configs/preprocessor_config.json",
#                                    special_tokens_path="utils/configs/tokenizer_config.json",
#                                    vocab_path="utils/configs/vocab.json", device=device
#                                    )
print("Starting Whisper...")
#model = WhisperModel("small", device=device, compute_type="float32")

def main(audio_buffer: List[float]) -> List[str]:
    print("test.")
    return []
#    transcript = []
    
#    numpy_buffer = np.array(audio_buffer, dtype=np.float32)
    #features = processor.extract_features(torch.from_numpy(numpy_buffer).to(device))
    #segments = model.transcribe(features, beam_size=5, language="en")

    #for segment in segments:
    #    cleaned = segment.text.lstrip()
    #    print("[%.2fs -> %.2fs]-%s" % (segment.start, segment.end, cleaned))
    #    transcript.append(cleaned)

#    return transcript