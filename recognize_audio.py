import os
import time
import random

from utils import save_execution_time, make_dir

from utils import preprocess_text

## constant for TTS
from constant import GOOGLE, RV, ESPEAK, FESTIVAL

## constant for ASR
from constant import DS, DS2, W2L, WIT

from asr import create_asr_by_name

def recognize(tts_name: str, asr_name: str, data_dir: str, execution_time_dir: str):
    asr = create_asr_by_name(asr_name)
    audio_dir = os.path.join(data_dir, "audio")
    transcription_dir = os.path.join(data_dir, "transcription", tts_name)
    execution_time_dir = os.path.join(execution_time_dir, "transcription", tts_name, asr_name)
    make_dir(execution_time_dir)

    for i in range(13229, 20000):
    # for i in range(1, 3):
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
    tts_name = RV
    asr_name = WIT
    data_dir = "data/"
    execution_time_dir = "execution_time/"

    recognize(tts_name, asr_name, data_dir, execution_time_dir)
