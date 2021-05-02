import os
from crossasr.tts import TTS
from crossasr.utils import make_dir

from gtts import gTTS

class Google(TTS):

    def __init__(self):
        TTS.__init__(self, name="google")

    def generateAudio(self, text: str, audio_dir: str, filename: str):
        base_dir = os.getcwd()
        tts_dir = os.path.join(base_dir, audio_dir, self.name)
        make_dir(tts_dir)
        temp_dir = os.path.join(tts_dir, "temp")
        make_dir(temp_dir)
        tempfile = os.path.join(temp_dir, filename + ".mp3")
        wavfile = os.path.join(tts_dir, filename + ".wav")
        googleTTS = gTTS(text, lang='en-us')
        googleTTS.save(tempfile)
        os.system('ffmpeg -i ' + tempfile +
                  ' -acodec pcm_s16le -ac 1 -ar 16000 ' + wavfile + ' -y')
        return os.path.relpath(wavfile, base_dir)


