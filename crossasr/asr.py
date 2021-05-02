import os
from crossasr.utils import make_dir

# import torch
# import soundfile as sf
# from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
# import os, gc
# import sys
# import subprocess

# ## constant for TTS
# from constant import GOOGLE, RV, ESPEAK, FESTIVAL

# ## constant for ASR
# from constant import DS, DS2, W2L, WIT, W2V


# from wit import Wit as WitAPI

# WIT_ACCESS_TOKEN = os.getenv("WIT_ACCESS_TOKEN")
# wit_client = WitAPI(WIT_ACCESS_TOKEN)

class ASR:
    def __init__(self, name):
        self.name = name
        self.transcription = ""

    def getName(self) :
        return self.name

    def setName(self, name:str):
        self.name = name

    def getTranscription(self):
        return self.transcription

    def setTranscription(self, transcription: str):
        self.transcription = transcription

    def recognizeAudio(self, audio_path: str) -> str:
        # abstract function need to be implemented by the child class
        raise NotImplementedError()
    
    def saveTranscription(self, transcription_dir: str, filename: str):
        transcription_dir = os.path.join(transcription_dir, self.getName())
        make_dir(transcription_dir)
        transcription_path = os.path.join(transcription_dir, filename + ".txt")
        with open(transcription_path, "w+") as f :
            f.write(self.getTranscription())
    
    def loadTranscription(self, transcription_dir: str, filename: str):
        transcription_dir = os.path.join(transcription_dir, self.getName())
        transcription_path = os.path.join(transcription_dir, filename + ".txt")
        f = open(transcription_path, "r") 
        lines = f.readlines()
        if len(lines) == 0 : return ""
        transcription = lines[0]
        f.close()

        return transcription

# def create_asr_by_name(name: str):
#     return {
#         DS: DeepSpeech(),
#         DS2: DeepSpeech2(),
#         W2L: Wav2Letter(),
#         WIT: Wit(),
#         W2V: Wav2Vec2()
#     }[name]

# def test():
#     audio_dir = "data/audio/"
#     transcription_dir = "data/transcription/"

#     tts_name = GOOGLE
#     filename = "hello_world"

#     audio_path = os.path.join(audio_dir, tts_name, filename + ".wav")
#     transcription_dir = os.path.join(transcription_dir, tts_name)

#     # ds = create_asr_by_name(DS)
#     # ds.recognizeAudio(audio_path=audio_path)
#     # ds.saveTranscription(transcription_dir=transcription_dir, filename=filename)

#     # ds2 = create_asr_by_name(DS2)
#     # ds2.recognizeAudio(audio_path=audio_path)
#     # ds2.saveTranscription(transcription_dir=transcription_dir, filename=filename)
    
#     # wit = create_asr_by_name(WIT)
#     # wit.recognizeAudio(audio_path=audio_path)
#     # wit.saveTranscription(transcription_dir=transcription_dir, filename=filename)

#     w2l = create_asr_by_name(W2L)
#     w2l.recognizeAudio(audio_path=audio_path)
#     w2l.saveTranscription(transcription_dir=transcription_dir, filename=filename)

# if __name__ == "__main__":
#     test()
