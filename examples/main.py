import os
import numpy as np
import json

from utils import set_seed, read_json

from tts import create_tts_by_name
from asr import create_asr_by_name
from crossasr import CrossASR


if __name__ == "__main__":

    json_config_path = "config.json"
    config = read_json(json_config_path)

    set_seed(config["seed"])

    tts = create_tts_by_name(config["tts"])
    asrs = []
    for asr_name in config["asrs"]:
        asrs.append(create_asr_by_name(asr_name))

    crossasr = CrossASR(tts=tts, asrs=asrs, output_dir=config["output_dir"])

    corpus_path = config["input_corpus"]
    file = open(corpus_path)
    corpus = file.readlines()
    texts = []
    for text in corpus:
        texts.append(text[:-1])
    
    recompute = bool(config["recompute"])
    crossasr.processCorpus(texts=texts, recompute=recompute)

