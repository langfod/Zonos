
import traceback
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from faster_whisper import WhisperModel
import logging

from .torch_utils import (audio_tensor_to_numpy, load_audio_to_np, resample_audio_tensor)



#WHISPER_MODEL = "sorendal/skyrim-whisper-base-int8"
#WHISPER_MODEL = "Numbat/faster-skyrim-whisper-base.en"
WHISPER_MODEL = "distil-whisper/distil-large-v3.5-ct2"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WHISPER_ENGINE: Optional[WhisperModel]= None

def initialize_whisper_model():
    global  WHISPER_ENGINE
    
    if WHISPER_ENGINE is None:
            # Initialize Whisper model
            #logging.info("Loading Whisper model...")
            WHISPER_ENGINE = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="float32")
            #WHISPER_ENGINE = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
            #WHISPER_ENGINE = WhisperModel(WHISPER_MODEL, device=DEVICE, compute_type="int8" if DEVICE=="cuda" else "int8")
            #logging.info("Whisper model loaded successfully.")
    return WHISPER_ENGINE


def transcribe_audio_with_whisper(ref_audio_torch: torch.Tensor, sr: int) -> str:
    """Transcribe audio using local whispercpp server with caching."""



    # Ensure model is initialized
    if WHISPER_ENGINE is None:
        initialize_whisper_model()



    resample, resample_sr = resample_audio_tensor(ref_audio_torch, sr, 16000)
    ref_audio_np = audio_tensor_to_numpy(resample, mono=True, copy=False)

    try:
        texts = []
        words = []
        segments, info = WHISPER_ENGINE.transcribe(
            audio=ref_audio_np,
            beam_size=8,
            vad_filter=False,
            without_timestamps=False,
            word_timestamps=True,
            multilingual=False,
            language="en",
        )
        for segment in segments:
            for word in segment.words:
                #print("[%.4fs -> %.4fs] %s" % (word.start, word.end, word.word))
                words.append(word)
            #print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
            texts.append(segment.text.strip())

        transcription = ' '.join(texts).strip()

        return words

    except Exception as e:
        print(traceback.format_exc())
        logging.error(f"Failed to transcribe audio: {str(e)}")
        return ""
