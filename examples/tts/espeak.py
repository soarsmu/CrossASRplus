import os
from crossasr.tts import TTS
from crossasr.utils import make_dir

class Espeak(TTS):

    def __init__(self):
        TTS.__init__(self, name="espeak")

    def generateAudio(self, text: str, audio_fpath: str):
        tempfile = audio_fpath.split(".")[0] + "-temp.wav"
        cmd = "espeak \"" + text + "\" --stdout > " + tempfile
        os.system(cmd)
        setting = " -acodec pcm_s16le -ac 1 -ar 16000 "
        os.system(f"ffmpeg -i {tempfile} {setting} {audio_fpath} -y")
