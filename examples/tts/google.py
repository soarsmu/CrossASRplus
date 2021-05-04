import os
from crossasr.tts import TTS
from crossasr.utils import make_dir

from gtts import gTTS

class Google(TTS):

    def __init__(self):
        TTS.__init__(self, name="google")

    def generateAudio(self, text: str, audio_fpath: str):
        tempfile = audio_fpath.split(".")[0] + "-temp.mp3"
        googleTTS = gTTS(text, lang='en-us')
        googleTTS.save(tempfile)
        setting = " -acodec pcm_s16le -ac 1 -ar 16000 "
        os.system(f"ffmpeg -i {tempfile} {setting} {audio_fpath} -y")



