import os
from crossasr.tts import TTS
from crossasr.utils import make_dir

class Espeak(TTS):

    def __init__(self):
        TTS.__init__(self, name="espeak")

    def generateAudio(self, text: str, audio_dir: str, filename: str):
        base_dir = os.getcwd()
        tts_dir = os.path.join(base_dir, audio_dir, self.name)
        make_dir(tts_dir)
        temp_dir = os.path.join(tts_dir, "temp")
        make_dir(temp_dir)
        tempfile = os.path.join(temp_dir, filename + ".wav")
        wavfile = os.path.join(tts_dir, filename + ".wav")

        cmd = "espeak \"" + text + "\" --stdout > " + tempfile
        os.system(cmd)
        os.system('ffmpeg -i ' + tempfile +
                  ' -acodec pcm_s16le -ac 1 -ar 16000 ' + wavfile + ' -y')

        return os.path.relpath(wavfile, base_dir)
