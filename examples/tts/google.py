import os
from crossasr.tts import TTS
import utils

class Google(TTS):

    def __init__(self):
        TTS.__init__(self, name="google")

    def generateAudio(self, text: str, audio_fpath: str):
        utils.googleGenerateAudio(text, audio_fpath)
        


