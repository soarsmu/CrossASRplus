import os
import numpy as np

from utils import preprocess_text, create_filename_from_text

from tts import TTS, Google, ResponsiveVoice, Espeak, Festival
from asr import ASR, DeepSpeech, DeepSpeech2, Wit, Wav2Letter

# class Classifier:
# 	def __init__(name):
# 	def classify(List[:text])

# # exract text and audio feature into emebedding feature consumable by Estimator


# class FeatureExtractor:

# 	# estimator to predict how likely the input to be a failed test case


# class Estimator:


class CrossASR:
    def __init__(self, tts: TTS, asrs: [ASR], audio_dir: str, transcription_dir: str) :
        self.tts = tts
        self.asrs = asrs
        self.audio_dir = audio_dir
        self.transcription_dir = transcription_dir

    def getTTS(self) :
        return self.tts

    def setTTS(self, tts: TTS) :
        self.tts = tts

    def getASRS(self) :
        return self.asrs
    
    def addASR(self, asr: ASR) :
        for curr_asr in self.asrs :
            if asr.getName() == curr_asr.getName() :
                # asr is already on the list of asrs
                return
        self.asrs.append(asr)

    def getAudioDir(self):
        return self.audio_dir

    def setAudioDir(self, audio_dir: str) :
        self.audio_dir = audio_dir
    

    def getTranscriptionDir(self):
        return self.transcription_dir

    def setTranscriptionDir(self, transcription_dir: str) :
        self.transcription_dir = transcription_dir
    

    def removeASR(self, asr_name: str):
        for i, asr in enumerate(self.asrs) :
            if asr_name == asr.getName() :
                break
        del self.asrs[i]

    def processText(self, text: str) :
        """
        Run CrossASR on a single text
        Description: Given a sentence as input, the program will generate a test case. The program needs some parameters, i.e. a TTS and ASRs used
        params:
        return:
        """
        text = preprocess_text(text)
        filename = create_filename_from_text(text)
        audio_path = self.getTTS().generateAudio(text=text, audio_dir=self.getAudioDir(), filename=filename)
        transcription_dir = os.path.join(self.getTranscriptionDir(), self.getTTS().getName())
        transcriptions = {}
        for asr in self.asrs :
            asr.recognizeAudio(audio_path=audio_path)
            asr.saveTranscription(
                transcription_dir=transcription_dir, filename=filename)
            transcriptions[asr.getName()] = asr.getTranscription()
        print(transcriptions)

    
    # """
	# Run CrossASR on a whole corpus**
	# given a corpus, which is a list of sentences, the CrossASR generates test cases.
	# There are 2 options, i.e. using FPP or without FPP
	# return:
	# """
    # def processCorpus(self, text: [str], use_estimator: boolean, paremeters, FeatureExtractor, Classifier)


def test(): 

    tts = Google()
    asrs = [DeepSpeech(), DeepSpeech2()]
    audio_dir = "data/audio/"
    transcription_dir = "data/transcription/"

    crossasr = CrossASR(tts=tts, asrs=asrs, audio_dir=audio_dir, transcription_dir=transcription_dir)
    crossasr.setTTS(ResponsiveVoice())
    crossasr.addASR(Wav2Letter())

    text = "hello world!"
    crossasr.processText(text)



if __name__ == "__main__" :
    test()
