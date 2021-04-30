import os
import time
import random

from utils import save_execution_time, make_dir

from utils import preprocess_text, read_json, set_seed

import constant

## constant for TTS
from constant import GOOGLE, RV, ESPEAK, FESTIVAL

## constant for ASR
from constant import DS, DS2, W2L, WIT, W2V

from constant import DATA_DIR, AUDIO_DIR, TRANSRCRIPTION_DIR, EXECUTION_TIME_DIR

from asr import create_asr_by_name

def recognize(tts_name: str, asr_name: str, data_dir: str, execution_time_dir: str):
    asr = create_asr_by_name(asr_name)
    audio_dir = os.path.join(data_dir, AUDIO_DIR)
    transcription_dir = os.path.join(data_dir, TRANSRCRIPTION_DIR, tts_name)
    execution_time_dir = os.path.join(execution_time_dir, TRANSRCRIPTION_DIR, tts_name, asr_name)
    make_dir(execution_time_dir)

    for i in range(0, 20001):
    # for i in range(0, 1):
        filename = f"{i}"

        print(f"Processing {i}")

        audio_path = os.path.join(audio_dir, tts_name, filename + ".wav")
        start = time.time()
        asr.recognizeAudio(audio_path=audio_path)
        asr.saveTranscription(
            transcription_dir=transcription_dir, filename=filename)
        end = time.time()
        execution_time = end - start
        fpath = os.path.join(execution_time_dir, filename + ".txt")
        save_execution_time(fpath=fpath, execution_time=execution_time)

        if asr_name in [WIT]:
            random_number = float(random.randint(9, 47))/10.
            time.sleep(random_number)


if __name__ == "__main__":
    
    json_config_path = "config.json"
    config = read_json(json_config_path)

    set_seed(config["seed"])

    corpus_path = os.path.join(config["output_dir"], constant.CORPUS_PATH)
    output_dir = config["output_dir"]
    data_dir = os.path.join(output_dir, DATA_DIR)
    execution_time_dir = os.path.join(output_dir, EXECUTION_TIME_DIR)

    tts_name = RV
    
    # for asr_name in [WIT, W2L, DS, DS2] :
    for asr_name in [W2V]:
        recognize(tts_name, asr_name, data_dir, execution_time_dir)
