from crossasr.tts import TTS
import utils

class Festival(TTS):

    def __init__(self, name="festival"):
        TTS.__init__(self, name=name)

    def generateAudio(self, text: str, audio_fpath: str):
        utils.festivalGenerateAudio(text=text, audio_fpath=audio_fpath)
