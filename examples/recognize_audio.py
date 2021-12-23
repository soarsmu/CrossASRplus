import os
import time
import random

import crossasr
from crossasr.utils import save_execution_time, make_dir, preprocess_text, read_json
from crossasr.constant import DATA_DIR, AUDIO_DIR, TRANSCRIPTION_DIR, EXECUTION_TIME_DIR

from utils import set_seed, create_asr_by_name

def is_empty_file(fpath:str) -> bool:
    file = open(fpath)
    line = file.readline()
    line = crossasr.utils.preprocess_text(line)
    file.close()
    if line == "":
        return True
    return False


def recognize(tts_name: str, asr_name: str, data_dir: str, execution_time_dir: str):
    asr = create_asr_by_name(asr_name)
    audio_dir = os.path.join(data_dir, AUDIO_DIR, tts_name)
    transcription_dir = os.path.join(data_dir, TRANSCRIPTION_DIR, tts_name)
    execution_time_dir = os.path.join(execution_time_dir, TRANSCRIPTION_DIR, tts_name, asr_name)
    make_dir(execution_time_dir)
    make_dir(os.path.join(transcription_dir, asr.getName()))

    for i in range(0, 2620):
        filename = f"{i}"

        print(f"Processing {i}")

        audio_fpath = os.path.join(audio_dir, filename + ".wav")
        execution_time_fpath = os.path.join(execution_time_dir, filename + ".txt")
        
        transcription_fpath = os.path.join(
            transcription_dir, asr.getName(),  filename + ".txt")

        if is_empty_file(transcription_fpath) :

            start = time.time()
            transcription = asr.recognizeAudio(audio_fpath=audio_fpath)
            asr.setTranscription(transcription)
            asr.saveTranscription(
                transcription_dir=transcription_dir, filename=filename)
            end = time.time()
            execution_time = end - start
            
            save_execution_time(fpath=execution_time_fpath,
                                execution_time=execution_time)

            if asr_name in ["wit"]:
                random_number = float(random.randint(9, 47))/10.
                time.sleep(random_number)


if __name__ == "__main__":
    
    json_config_path = "config-dev-other.json"
    # json_config_path = "config.json"
    config = read_json(json_config_path)

    set_seed(config["seed"])

    corpus_path = os.path.join(config["output_dir"], config["corpus_fpath"])
    output_dir = config["output_dir"]
    data_dir = os.path.join(output_dir, DATA_DIR)
    execution_time_dir = os.path.join(output_dir, EXECUTION_TIME_DIR)

    # tts_name = "google"
    
    tts_name = config["tts"]
    asr_name = "deepspeech"

    
    recognize(tts_name, asr_name, data_dir, execution_time_dir)

    # for asr_name in ["deepspeech", "deepspeech2", "wav2letter", "wit", "wav2vec2"] :
    # for asr_name in ["wav2vec2"]:
    #     recognize(tts_name, asr_name, data_dir, execution_time_dir)
