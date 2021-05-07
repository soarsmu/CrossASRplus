import os
from crossasr.tts import TTS
import utils

class Google(TTS):

    def __init__(self, name="google"):
        TTS.__init__(self, name=name)

    def generateAudio(self, text: str, audio_fpath: str):
        utils.googleGenerateAudio(text, audio_fpath)
        


