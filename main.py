import numpy as np
import json

from utils import set_seed, read_json

from tts import create_tts_by_name
from asr import create_asr_by_name



if __name__ == "__main__":

    json_config_path = "config.json"
    config = read_json(json_config_path)

    set_seed(config["seed"])
    
    tts = create_tts_by_name(config["tts"])
    asrs = []
    for asr_name in config["asrs"] :
        asrs.append(create_asr_by_name(asrs))

    file = open(corpus_path)
    corpus = file.readlines()
    
    audio_dir = os.path.join(data_dir, "audio")
    execution_time_dir = os.path.join(execution_time_dir, "audio", tts_name)

    i = 0
    # for i in range(len(corpus)) :
    for i in range(10) :
        text = corpus[i][:-1]
        if config["recompute"] :
            i
            ## generate audio
            ## save execution time
        ## load execution time for generating audio

        transcriptions = {}
        for asr in asrs :
            if config["recompute"] :
                ## recognize audio
                ## save transcription
                ## save execution time
            ## load transcription
            ## postprocess transcriptions
            ## load execution time for recognizing audio
        
        
            
            
            



