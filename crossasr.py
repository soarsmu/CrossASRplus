import os
import numpy as np

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
    def __init__(self, tts: TTS, asrs: [ASR]) :
        self.tts = tts
        self.asrs = asrs

    def getTTS(self) :
        return self.tts
    def setTTS(self, tts: TTS) :
        self.tts = tts

    def getASRS(self) :
        return self.asrs
    
    def addASR(self, asr: ASR) :
        for curr_asr in self.asrs :
            if asr.getName() == curr_asr.getName() :
                # asr is on the list of asrs
                return
        self.asrs.append(asr)
    
    def removeASR(self, asr_name: str):
        for i, asr in enumerate(self.asrs) :
            if asr_name == asr.getName() :
                break
        del self.asrs[i]

    """
	Run CrossASR on a single text
	Description: Given a sentence as input, the program will generate a test case. The program needs some parameters, i.e. a TTS and ASRs used
	params:
	return:
	"""
    # def processText(self, text: str) :

    #     audio_path = self.getTTS().generateAudio()

    
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

    crossasr = CrossASR(tts=tts, asrs=asrs)
    crossasr.setTTS(ResponsiveVoice())
    crossasr.addASR(Wav2Letter())



if __name__ == "__main__" :
    test()
