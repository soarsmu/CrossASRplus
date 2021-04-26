import os, time
import numpy as np

from constant import UNDETERMINABLE_TEST_CASE, SUCCESSFUL_TEST_CASE, FAILED_TEST_CASE
from constant import DATA_DIR, EXECUTION_TIME_DIR, CASE_DIR
from constant import AUDIO_DIR, TRANSRCRIPTION_DIR

from utils import preprocess_text, create_filename_from_text, set_seed
from utils import make_dir, read_json, save_execution_time, get_execution_time

## constant for TTS
from constant import GOOGLE, RV, ESPEAK, FESTIVAL

## constant for ASR
from constant import DS, DS2, W2L, WIT

from tts import TTS, Google, ResponsiveVoice, Espeak, Festival, create_tts_by_name
from asr import ASR, DeepSpeech, DeepSpeech2, Wit, Wav2Letter, create_asr_by_name

from jiwer import wer

# class Classifier:
# 	def __init__(name):
# 	def classify(List[:text])

# # exract text and audio feature into emebedding feature consumable by Estimator


# class FeatureExtractor:

# 	# estimator to predict how likely the input to be a failed test case


# class Estimator:


class CrossASR:
    def __init__(self, tts: TTS, asrs: [ASR], output_dir: str) :
        self.tts = tts
        self.asrs = asrs
        
        self.audio_dir = os.path.join(output_dir, DATA_DIR, AUDIO_DIR) 
        self.transcription_dir = os.path.join(output_dir, DATA_DIR, TRANSRCRIPTION_DIR)
        self.execution_time_dir = os.path.join(output_dir, EXECUTION_TIME_DIR)
        self.case_dir = os.path.join(output_dir, CASE_DIR)


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
    
    def removeASR(self, asr_name: str):
        for i, asr in enumerate(self.asrs) :
            if asr_name == asr.getName() :
                break
        del self.asrs[i]

    def getOutputDir(self):
        return self.audio_dir

    def setOutputDir(self, output_dir: str) :
        self.output_dir = output_dir

        self.audio_dir = os.path.join(output_dir, DATA_DIR, AUDIO_DIR)
        self.transcription_dir = os.path.join(output_dir, DATA_DIR, TRANSRCRIPTION_DIR)
        self.execution_time_dir = os.path.join(output_dir, EXECUTION_TIME_DIR)
        self.case_dir = os.path.join(output_dir, CASE_DIR)

    def caseDeterminer(self, text:str, transcriptions: str): 
        # word error rate
        wers = {}

        is_determinable = False

        for k, transcription in transcriptions.items():
            word_error_rate = wer(transcription, text)
            wers[k] = word_error_rate
            if word_error_rate == 0:
                is_determinable = True

        cases = {}
        if is_determinable:
            for k in transcriptions.keys():
                if wers[k] == 0:
                    cases[k] = SUCCESSFUL_TEST_CASE
                else:
                    cases[k] = FAILED_TEST_CASE
        else:
            for k in transcriptions.keys():
                cases[k] = UNDETERMINABLE_TEST_CASE

        return cases

    def saveCase(self, case_dir: str, tts_name: str, asr_name: str, filename:str, case:str) :
        case_dir = os.path.join(case_dir, tts_name, asr_name)
        make_dir(case_dir)
        fpath = os.path.join(case_dir, filename + ".txt")
        file = open(fpath, "w+")
        file.write(case)
        file.close()

    def processText(self, text: str, filename: str, recompute: bool) :
        """
        Run CrossASR on a single text
        Description: Given a sentence as input, the program will generate a test case. The program needs some parameters, i.e. a TTS and ASRs used
        params:
        return:
        """
        execution_time = 0.

        directory = os.path.join(self.execution_time_dir, AUDIO_DIR, self.getTTS().getName())
        make_dir(directory)
        time_for_generating_audio_fpath = os.path.join(directory, filename + ".txt")
        
        if recompute :
            start_time = time.time()
            audio_path = self.getTTS().generateAudio(text=text, audio_dir=self.audio_dir, filename=filename)
            save_execution_time(
                fpath=time_for_generating_audio_fpath, execution_time=time.time() - start_time)
        execution_time += get_execution_time(
            fpath=time_for_generating_audio_fpath)
        
        transcription_dir = os.path.join(self.transcription_dir, self.getTTS().getName())
        
        transcriptions = {}
        for asr in self.asrs :
            asr.recognizeAudio(audio_path=audio_path)
            asr.saveTranscription(
                transcription_dir=transcription_dir, filename=filename)
            transcriptions[asr.getName()] = asr.getTranscription()
        
        print(transcriptions)

        cases = self.caseDeterminer(text, transcriptions)
        
        print(cases)
        
        for asr_name, case in cases.items() :
            self.saveCase(self.case_dir, self.getTTS().getName(), asr_name, filename, str(case))
    
    def processCorpus(self, text: [str]):
        # """
        # Run CrossASR on a whole corpus**
        # given a corpus, which is a list of sentences, the CrossASR generates test cases.
        # There are 2 options, i.e. using FPP or without FPP
        # return:
        # """
        # def processCorpus(self, text: [str], use_estimator: boolean, paremeters, FeatureExtractor, Classifier)
        1


def test(): 

    json_config_path = "config.json"
    config = read_json(json_config_path)

    set_seed(config["seed"])

    tts = create_tts_by_name(config["tts"])
    asrs = []
    for asr_name in config["asrs"]:
        asrs.append(create_asr_by_name(asr_name))

    crossasr = CrossASR(tts=tts, asrs=asrs, output_dir=config["output_dir"])
    
    text = "hello world!"
    text = preprocess_text(text)
    filename = "hello_world" 
    recompute = True
    crossasr.processText(text=text, filename=filename, recompute=recompute)


if __name__ == "__main__" :
    set_seed(2021)
    test()
