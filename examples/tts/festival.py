import os
from crossasr.tts import TTS
from crossasr.utils import make_dir


class Festival(TTS):

    def __init__(self):
        TTS.__init__(self, name="festival")

    def generateAudio(self, text: str, audio_fpath: str):
        cmd = "festival -b \"(utt.save.wave (SayText \\\"" + \
            text + "\\\") \\\"" + audio_fpath + "\\\" 'riff)\""
        os.system(cmd)