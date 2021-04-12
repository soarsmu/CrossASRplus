import os
import sys

from gtts import gTTS

from utils import make_dir

class TTS:

    def __init__(self, name):
        self.name = name

    def getName(self):
        return self.name

    def setName(self, name: str):
        self.name = name

    def generateAudio(self, text:str, audiodir:str, filename:str):
        # abstract function need to be implemented by the child class
        raise NotImplementedError()


class Google(TTS):

    def __init__(self):
        TTS.__init__(self, name="google")

    def generateAudio(self, text:str, audio_dir:str, filename:str):
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


class ResponsiveVoice(TTS):

    def __init__(self):
        TTS.__init__(self, name="rv")

    def generateAudio(self, text:str, audio_dir:str, filename:str):
        base_dir = os.getcwd()
        tts_dir = os.path.join(base_dir, audio_dir, self.name)
        make_dir(tts_dir)
        temp_dir = os.path.join(tts_dir, "temp")
        make_dir(temp_dir)
        tempfile = os.path.join(temp_dir, filename + ".mp3")
        wavfile = os.path.join(tts_dir, filename + ".wav")


        cmd = "rvtts --voice english_us_male --text \"" + text + "\" -o " + tempfile

        os.system(cmd)
        os.system('ffmpeg -i ' + tempfile +
                ' -acodec pcm_s16le -ac 1 -ar 16000 ' + wavfile + ' -y')


class Espeak(TTS):

    def __init__(self):
        TTS.__init__(self, name="espeak")

    def generateAudio(self, text:str, audio_dir:str, filename:str):
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


class Festival(TTS):

    def __init__(self):
        TTS.__init__(self, name="festival")

    def generateAudio(self, text:str, audio_dir:str, filename:str):
        base_dir = os.getcwd()
        tts_dir = os.path.join(base_dir, audio_dir, self.name)
        make_dir(tts_dir)
        wavfile = os.path.join(tts_dir, filename + ".wav")

        cmd = "festival -b \"(utt.save.wave (SayText \\\"" + \
            text + "\\\") \\\"" + wavfile + "\\\" 'riff)\""
        os.system(cmd)


def test():
    text = "hello world!"
    audio_dir = "data/audio/"
    filename = "hello_world"

    google = Google()
    google.generateAudio(text=text, audio_dir=audio_dir, filename=filename)
    
    rv = ResponsiveVoice()
    rv.generateAudio(text=text, audio_dir=audio_dir, filename=filename)

    espeak = Espeak()
    espeak.generateAudio(text=text, audio_dir=audio_dir, filename=filename)

    festival = Festival()
    festival.generateAudio(text=text, audio_dir=audio_dir, filename=filename)

if __name__ == "__main__":
    test()
