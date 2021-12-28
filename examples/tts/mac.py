import os
from crossasr.tts import TTS
import utils


class Mac(TTS):

    def __init__(self, name="mac", voice="Alex"):
        TTS.__init__(self, name=name)
        self.voice = voice

    def getVoice(self) -> str:
        return self.voice

    def setVoice(self, voice:str) -> str:
        self.voice = voice

    def generateAudio(self, text: str, audio_fpath: str):
        utils.macGenerateAudio(text, audio_fpath, self.getVoice())
