import os, time, random
import numpy as np

from constant import UNDETERMINABLE_TEST_CASE, SUCCESSFUL_TEST_CASE, FAILED_TEST_CASE
from constant import DATA_DIR, EXECUTION_TIME_DIR, CASE_DIR
from constant import AUDIO_DIR, TRANSRCRIPTION_DIR

from utils import preprocess_text, create_filename_from_text, set_seed
from utils import make_dir, read_json, save_execution_time, get_execution_time
from text import Text

## constant for TTS
from constant import GOOGLE, RV, ESPEAK, FESTIVAL

## constant for ASR
from constant import DS, DS2, W2L, WIT

from tts import TTS, Google, ResponsiveVoice, Espeak, Festival, create_tts_by_name
from asr import ASR, DeepSpeech, DeepSpeech2, Wit, Wav2Letter, create_asr_by_name

from jiwer import wer

from estimator import HuggingFaceTransformer, create_huggingface_estimator_by_name

class CrossASR:
    def __init__(self, tts: TTS, asrs: [ASR], output_dir: str, recompute=False, num_iteration=5, time_budget=3600, max_num_retry=0, estimator=None) :
        self.tts = tts
        self.asrs = asrs
        
        self.audio_dir = os.path.join(output_dir, DATA_DIR, AUDIO_DIR) 
        self.transcription_dir = os.path.join(output_dir, DATA_DIR, TRANSRCRIPTION_DIR)
        self.execution_time_dir = os.path.join(output_dir, EXECUTION_TIME_DIR)
        self.case_dir = os.path.join(output_dir, CASE_DIR)
        self.recompute = recompute
        self.num_iteration = num_iteration
        self.time_budget = time_budget
        self.max_num_retry = max_num_retry

        self.estimator = estimator


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
            word_error_rate = wer(text, transcription)
            wers[k] = word_error_rate
            if word_error_rate == 0:
                is_determinable = True

        case = {}
        if is_determinable:
            for k in transcriptions.keys():
                if wers[k] == 0:
                    case[k] = SUCCESSFUL_TEST_CASE
                else:
                    case[k] = FAILED_TEST_CASE
        else:
            for k in transcriptions.keys():
                case[k] = UNDETERMINABLE_TEST_CASE

        return case

    def saveCase(self, case_dir: str, tts_name: str, asr_name: str, filename:str, case:str) :
        case_dir = os.path.join(case_dir, tts_name, asr_name)
        make_dir(case_dir)
        fpath = os.path.join(case_dir, filename + ".txt")
        file = open(fpath, "w+")
        file.write(case)
        file.close()

    def processText(self, text: str, filename: str) :
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
        
        audio_path = self.getTTS().getAudioPath(
            text=text, audio_dir=self.audio_dir, filename=filename)
        
        if self.recompute or not os.path.exists(audio_path):
            start_time = time.time()
            self.getTTS().generateAudio(text=text, audio_dir=self.audio_dir, filename=filename)
            save_execution_time(
                fpath=time_for_generating_audio_fpath, execution_time=time.time() - start_time)
        
        ## add execution time for generating audio
        execution_time += get_execution_time(
            fpath=time_for_generating_audio_fpath)    
        
        transcription_dir = os.path.join(self.transcription_dir, self.getTTS().getName())
        
        transcriptions = {}
        for asr in self.asrs :
            directory = os.path.join(
                self.execution_time_dir, TRANSRCRIPTION_DIR, self.getTTS().getName(), asr.getName())
            make_dir(directory)
            time_for_recognizing_audio_fpath = os.path.join(
                directory, filename + ".txt")

            if self.recompute :
                start_time = time.time()
                asr.recognizeAudio(audio_path=audio_path)
                asr.saveTranscription(
                    transcription_dir=transcription_dir, filename=filename)
                save_execution_time(fpath=time_for_recognizing_audio_fpath, execution_time=time.time() - start_time)
            
            transcription = asr.loadTranscription(
                transcription_dir=transcription_dir, filename=filename)
            num_retry = 0
            while transcription == "" and num_retry < self.max_num_retry :
                start_time = time.time()
                asr.recognizeAudio(audio_path=audio_path)
                asr.saveTranscription(
                    transcription_dir=transcription_dir, filename=filename)
                save_execution_time(
                    fpath=time_for_recognizing_audio_fpath, execution_time=time.time() - start_time)
                transcription = asr.loadTranscription(
                    transcription_dir=transcription_dir, filename=filename)

                if asr.getName() == WIT :
                    random_number = float(random.randint(9, 47))/10.
                    time.sleep(random_number)

                num_retry += 1

            transcriptions[asr.getName()] = transcription

            time_for_recognizing_audio_fpath
            ## add execution time for generating audio
            execution_time += get_execution_time(
                fpath=time_for_recognizing_audio_fpath)    
            
        # print(transcriptions)

        cases = self.caseDeterminer(text, transcriptions)
        
        # print(cases)
        
        for asr_name, case in cases.items() :
            self.saveCase(self.case_dir, self.getTTS().getName(), asr_name, filename, str(case))

        # print(f"Execution time: {execution_time}")
        return cases, execution_time
    
    def processCorpus(self, texts: [Text]):
        # """
        # Run CrossASR on a corpus
        # given a corpus, which is a list of sentences, the CrossASR generates test cases.
        # There are 2 options, i.e. using FPP or without FPP
        # return:
        # """
        # def processCorpus(self, text: [str], use_estimator: boolean, paremeters, FeatureExtractor, Classifier)

        def processOneIteration(texts: [Text]):
            execution_time = 0.
            processed_texts = []
            cases = []
            start_time = time.time()

            i = 0
            for text in texts :
                case, exec_time = self.processText(text=text.getText(), filename=f"{text.getId()}")
                cases.append(case)
                execution_time += exec_time
                i += 1
                if execution_time + time.time() - start_time > self.time_budget :
                    # print(f"Number of processed texts {i}")
                    break
            processed_texts = texts[:i]
            remaining_texts = texts[i:]

            assert len(texts) == (len(processed_texts) + len(remaining_texts))
            
            return processed_texts, remaining_texts, cases
        
        processed_texts = []
        cases = []
        num_failed_test_cases = []
        remaining_texts = texts
        for _ in range(self.num_iteration): 
            # print(len(remaining_texts))
            curr_processed_texts, remaining_texts, curr_cases = processOneIteration(
                remaining_texts)
            processed_texts.extend(curr_processed_texts)
            cases.extend(curr_cases)
            if self.estimator :
                labels = get_labels_from_cases(cases)
                self.trainEstimator(processed_texts, labels)
                remaining_texts = self.rank(remaining_texts)
            num_failed_test_cases.append(calculate_cases(curr_cases, mode=FAILED_TEST_CASE))

        # print(len(processed_texts))
        print(sum(num_failed_test_cases))

        ## TODO: 
        # save raw output, 
        # calculate the number of failed test cases in each iteration
        # create visualisation

    def get_text_only(self, texts:[Text]) -> [str]:
        res = []
        for t in texts :
            res.append(t.getText()) 
        return res    

    
    def trainEstimator(self, processed_texts, labels):
        train_texts = self.get_text_only(processed_texts)
        self.estimator.fit(train_texts, labels)

    def rank(self, texts:[Text]): 
        
        ranking = self.estimator.predict(self.get_text_only(texts))
        
        ## https://stackoverflow.com/questions/6618515/sorting-list-based-on-values-from-another-list
        texts = [x for _, x in reversed(sorted(zip(ranking, texts)))]
        
        return texts
        
def calculate_cases(cases, mode=FAILED_TEST_CASE):
    count = 0
    for c in cases :
        for _, v in c.items() :
            if v == mode :
                count += 1
    return count




def get_labels_from_cases(cases) :
    def determine_label(case) :
        if UNDETERMINABLE_TEST_CASE in case.values() :
            return UNDETERMINABLE_TEST_CASE
        if FAILED_TEST_CASE :
            return FAILED_TEST_CASE
        return SUCCESSFUL_TEST_CASE

    labels = []
    for case in cases :
        label = determine_label(case)
        labels.append(label)
    
    return labels

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
    crossasr.processText(text=text, filename=filename)

def test_corpus(): 

    json_config_path = "config.json"
    config = read_json(json_config_path)

    set_seed(config["seed"])

    tts = create_tts_by_name(config["tts"])
    asrs = []
    for asr_name in config["asrs"]:
        asrs.append(create_asr_by_name(asr_name))

    kwargs = {
        "recompute" : bool(config["recompute"]),
        "time_budget" : int(config["time_budget"]),
        "num_iteration" : int(config["num_iteration"]),
        "max_num_retry": int(config["max_num_retry"])
    }

    if config["estimator"] :
        if config["estimator_type"] == "huggingface":
            kwargs["estimator"] = create_huggingface_estimator_by_name(str(config["estimator"]))
    
    crossasr = CrossASR(tts=tts, asrs=asrs, output_dir=config["output_dir"], **kwargs)
    
    corpus_path = config["input_corpus"]
    file = open(corpus_path)
    corpus = file.readlines()
    texts = []
    i = 1
    # for text in corpus :
    for text in corpus:
        texts.append(Text(i, text[:-1]))
        i += 1
    crossasr.processCorpus(texts=texts)

if __name__ == "__main__" :
    # test()
    test_corpus()
