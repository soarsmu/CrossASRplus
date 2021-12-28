import os 
import time
import random

from crossasr.constant import DATA_DIR, AUDIO_DIR, EXECUTION_TIME_DIR
from crossasr.utils import save_execution_time, make_dir, read_json, read_corpus

from utils import create_tts_by_name, set_seed

def generate(tts_name: str, corpus_path: str, data_dir: str, execution_time_dir:str):
    tts = create_tts_by_name(tts_name)
    
    audio_dir = os.path.join(data_dir, AUDIO_DIR, tts.getName())
    os.makedirs(audio_dir, exist_ok=True)
    execution_time_dir = os.path.join(
        execution_time_dir, AUDIO_DIR, tts.getName())
    make_dir(execution_time_dir)

    corpus = read_corpus(corpus_fpath=corpus_path)
    print(f"Corpus file path: {corpus_path}")
    print(f"Length: {len(corpus)}")

    for i in range(0, len(corpus)) :
        c = corpus[i]
        text = c.getText()
        filename = c.getId()
        start = time.time()
        audio_fpath = os.path.join(audio_dir, f"{filename}.wav")
        if not os.path.exists(audio_fpath) :
            tts.generateAudio(text=text, audio_fpath=audio_fpath)
            end = time.time()
            execution_time = end - start
            fpath = os.path.join(execution_time_dir, f"{filename}.txt")
            save_execution_time(fpath=fpath, execution_time=execution_time)
            print(f"Generate {i}")
            i += 1
            if tts_name in ["google"]:
                random_number = float(random.randint(15, 40))/10.
                time.sleep(random_number)

if __name__ == "__main__" :

    for json_config_path in ["config-test-clean.json",
                             "config-test-other.json",
                             "config-dev-clean.json",
                             "config-dev-other.json"] :
        
        
        config = read_json(json_config_path)

        set_seed(config["seed"])

        corpus_path = os.path.join(config["output_dir"], config["corpus_fpath"])
        output_dir = config["output_dir"]
        data_dir = os.path.join(output_dir, DATA_DIR)
        execution_time_dir = os.path.join(output_dir, EXECUTION_TIME_DIR)

        tts_name = config["tts"]
        generate(tts_name, corpus_path, data_dir, execution_time_dir)

