import os
from crossasr.utils import make_dir

class ASR:
    def __init__(self, name):
        self.name = name
        self.transcription = ""

    def getName(self) :
        return self.name

    def setName(self, name:str):
        self.name = name

    def getTranscription(self):
        return self.transcription

    def setTranscription(self, transcription: str):
        self.transcription = transcription

    def recognizeAudio(self, audio_path: str) -> str:
        # abstract function need to be implemented by the child class
        raise NotImplementedError()
    
    def saveTranscription(self, transcription_dir: str, filename: str):
        transcription_dir = os.path.join(transcription_dir, self.getName())
        make_dir(transcription_dir)
        transcription_path = os.path.join(transcription_dir, filename + ".txt")
        with open(transcription_path, "w+") as f :
            f.write(self.getTranscription())
    
    def loadTranscription(self, transcription_dir: str, filename: str):
        transcription_dir = os.path.join(transcription_dir, self.getName())
        transcription_path = os.path.join(transcription_dir, filename + ".txt")
        f = open(transcription_path, "r") 
        lines = f.readlines()
        if len(lines) == 0 : return ""
        transcription = lines[0]
        f.close()

        return transcription