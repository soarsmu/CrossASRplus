import random
import numpy as np
import json 
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

from pool import asr_pool, tts_pool
from crossasr.text import Text

def getTTS(tts_name: str):
    for tts in tts_pool :
        if tts_name == tts.getName() :
            return tts
    raise NotImplementedError("There is a TTS name problem")


def getASRS(asr_names: [str]):
    asrs = []
    for asr in asr_pool:
        for asr_name in asr_names :
            if asr_name == asr.getName():
                asrs.append(asr)
    if len(asr_names) == len(asrs) :
        return asrs
    raise NotImplementedError("There is an ASR name problem")

def set_seed(seed: int) :
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def readJson(config_path: str):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def readCorpus(corpus_fpath: str) :
    file = open(corpus_fpath)
    corpus = file.readlines()
    texts = []
    i = 1
    for text in corpus:
        texts.append(Text(i, text[:-1]))
        i += 1
    return texts
    
def parseConfig(config):
    conf = {}
    for k,v in config.items() :
        if k != "tts" and k!= "asrs" and k != "corpus_fpath":
            conf[k] = v
    return conf


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


