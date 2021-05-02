import os
from crossasr.tts import TTS
from crossasr.utils import make_dir


class Festival(TTS):

    def __init__(self):
        TTS.__init__(self, name="festival")

    def generateAudio(self, text: str, audio_dir: str, filename: str):
        base_dir = os.getcwd()
        tts_dir = os.path.join(base_dir, audio_dir, self.name)
        make_dir(tts_dir)
        wavfile = os.path.join(tts_dir, filename + ".wav")

        cmd = "festival -b \"(utt.save.wave (SayText \\\"" + \
            text + "\\\") \\\"" + wavfile + "\\\" 'riff)\""
        os.system(cmd)

        return os.path.relpath(wavfile, base_dir)
