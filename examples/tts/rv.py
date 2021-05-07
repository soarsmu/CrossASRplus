from crossasr.tts import TTS
import utils


class ResponsiveVoice(TTS):

    def __init__(self, name="rv"):
        TTS.__init__(self, name = name)

    def generateAudio(self, text: str, audio_fpath: str):
        utils.rvGenerateAudio(text, audio_fpath)