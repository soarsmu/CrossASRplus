import os
# import sys

# from gtts import gTTS

# from constant import GOOGLE, RV, ESPEAK, FESTIVAL

# from utils import make_dir, preprocess_text, create_filename_from_text

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


# def create_tts_by_name(name: str):
#     return {
#         GOOGLE: Google(),
#         RV: ResponsiveVoice(),
#         ESPEAK: Espeak(),
#         FESTIVAL: Festival()
#     }[name]

# def test():
#     text = "hello world!"
#     audio_dir = "data/audio/"

#     text = preprocess_text(text)
#     filename = create_filename_from_text(text)

#     ttses = [GOOGLE, RV, ESPEAK, FESTIVAL]

#     for tts_name in ttses :
#         tts = create_tts_by_name(tts_name)
#         tts.generateAudio(text=text, audio_dir=audio_dir, filename=filename)

# if __name__ == "__main__":
#     test()
