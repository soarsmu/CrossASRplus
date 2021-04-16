import numpy as np
import json

from utils import set_seed, read_json

from tts import create_tts_by_name
from asr import create_asr_by_name



if __name__ == "__main__":

    json_config_path="config.json"
    config = read_json(json_config_path)

    print(config)

    set_seed(config["seed"])
    tts = createTTSbyName(config["tts"])
    asrs = []
    for asr_name in config["asrs"] :
        asrs.append(createASRbyName(asrs))
