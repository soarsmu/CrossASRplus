import os
from crossasr.tts import TTS
from crossasr.utils import make_dir


class ResponsiveVoice(TTS):

    def __init__(self):
        TTS.__init__(self, name="rv")

    def generateAudio(self, text: str, audio_fpath: str):
        tempfile = audio_fpath.split(".")[0] + "-temp.mp3"
        cmd = "rvtts --voice english_us_male --text \"" + text + "\" -o " + tempfile
        os.system(cmd)
        setting = " -acodec pcm_s16le -ac 1 -ar 16000 "
        os.system(f"ffmpeg -i {tempfile} {setting} {audio_fpath} -y")

        

