from crossasr.tts import TTS
import utils

class ResponsiveVoice(TTS):

    def __init__(self):
        TTS.__init__(self, name="rv")

    def generateAudio(self, text: str, audio_fpath: str):
        utils.rvGenerateAudio(text=text, audio_fpath=audio_fpath)
