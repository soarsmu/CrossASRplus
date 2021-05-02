import os

class TTS:

    def __init__(self, name):
        self.name = name

    def getName(self):
        return self.name

    def setName(self, name: str):
        self.name = name

    def getAudioPath(self, text: str, audio_dir: str, filename: str):
        base_dir = os.getcwd()
        tts_dir = os.path.join(base_dir, audio_dir, self.name)
        wavfile = os.path.join(tts_dir, filename + ".wav")

        return os.path.relpath(wavfile, base_dir)


    def generateAudio(self, text:str, audio_dir:str, filename:str):
        # abstract function need to be implemented by the child class
        raise NotImplementedError()
