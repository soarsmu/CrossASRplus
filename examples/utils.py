import random
import numpy as np

import torch

from tts.rv import ResponsiveVoice
from tts.google import Google
from tts.espeak import Espeak
from tts.festival import Festival

from asr.deepspeech import DeepSpeech
from asr.deepspeech2 import DeepSpeech2
from asr.wav2letter import Wav2Letter
from asr.wit import Wit
from asr.wav2vec2 import Wav2Vec2

from estimator.huggingface import HuggingFaceTransformer

def set_seed(seed: int) :
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def create_tts_by_name(name: str):
    return {
        "google" : Google(),
        "rv" : ResponsiveVoice(),
        "espeak" : Espeak(),
        "festival" : Festival()
    }[name]

def create_asr_by_name(name: str):
    return {
        "deepspeech" : DeepSpeech(),
        "deepspeech2" : DeepSpeech2(),
        "wav2letter" : Wav2Letter(),
        "wit" : Wit(),
        "wav2vec2": Wav2Vec2()
    }[name]


def create_huggingface_estimator_by_name(name: str):
    # https://huggingface.co/transformers/custom_datasets.html
    return HuggingFaceTransformer(name=name)


