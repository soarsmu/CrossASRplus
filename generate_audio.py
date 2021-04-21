import os 
import time
import random


from constant import GOOGLE, RV, FESTIVAL, ESPEAK
from tts import create_tts_by_name
from utils import create_filename_from_text, save_execution_time, make_dir


def generate(tts_name: str, corpus_path: str, data_dir: str, execution_time_dir:str):
    tts = create_tts_by_name(tts_name)
    file = open(corpus_path)
    corpus = file.readlines()
    audio_dir = os.path.join(data_dir, "audio")
    execution_time_dir = os.path.join(execution_time_dir, "audio", tts_name)
    make_dir(execution_time_dir)

    i = 1
    # for text in corpus :
    for i in range(1, 20001) :
        text = corpus[i]
        text = text[:-1]
        filename = f"{i}"
        start = time.time()
        tts.generateAudio(text=text, audio_dir=audio_dir, filename=filename)
        end = time.time()
        execution_time = end - start
        fpath = os.path.join(execution_time_dir, filename + ".txt")
        save_execution_time(fpath=fpath, execution_time=execution_time)
        i += 1
        print(f"Generate {i}")
        if tts_name in [GOOGLE]:
            random_number = float(random.randint(1, 10))/10.
            time.sleep(random_number)

    file.close()


if __name__ == "__main__" :
    tts_name = FESTIVAL
    corpus_path = "corpus/europarl-20000.txt"
    data_dir = "data/"
    execution_time_dir = "execution_time/"

    generate(tts_name, corpus_path, data_dir, execution_time_dir)

