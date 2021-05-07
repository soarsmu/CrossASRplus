from crossasr.tts import TTS
import utils

class Espeak(TTS):

    def __init__(self, name="espeak"):
        TTS.__init__(self, name=name)

    def generateAudio(self, text: str, audio_fpath: str):
        utils.espeakGenerateAudio(text, audio_fpath)
