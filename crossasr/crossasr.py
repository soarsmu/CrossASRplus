import os, time, random
import numpy as np
import json

import crossasr.constant
from crossasr.constant import INDETERMINABLE_TEST_CASE, SUCCESSFUL_TEST_CASE, FAILED_TEST_CASE
from crossasr.constant import DATA_DIR, EXECUTION_TIME_DIR, CASE_DIR
from crossasr.constant import AUDIO_DIR, TRANSCRIPTION_DIR

from crossasr.utils import preprocess_text
from crossasr.utils import make_dir, read_json, save_execution_time, get_execution_time
from crossasr.text import Text

from crossasr.tts import TTS
from crossasr.asr import ASR

from jiwer import wer


class CrossASR:
    def __init__(self, tts: TTS, asrs: [ASR], output_dir: str, recompute=False, num_iteration=5, time_budget=3600, max_num_retry=0, text_batch_size=400, estimator=None) :
        self.tts = tts
        self.asrs = asrs
        
        self.audio_dir = os.path.join(output_dir, DATA_DIR, AUDIO_DIR) 
        self.transcription_dir = os.path.join(output_dir, DATA_DIR, TRANSCRIPTION_DIR)
        self.execution_time_dir = os.path.join(output_dir, EXECUTION_TIME_DIR)
        self.case_dir = os.path.join(output_dir, CASE_DIR)
        self.recompute = recompute
        self.num_iteration = num_iteration
        self.time_budget = time_budget
        self.max_num_retry = max_num_retry
        self.text_batch_size = text_batch_size
        self.estimator = estimator

        asrs_dir = "_".join([asr.getName() for asr in asrs])
        result_dir = os.path.join(output_dir, "result", tts.getName(), asrs_dir, f"text_batch_size_{text_batch_size}")
        make_dir(result_dir)
        experiment_name = f"with-estimator-{estimator.getName().replace('/','-')}" if estimator else "without-estimator"
        self.outputfile_failed_test_case = os.path.join(result_dir, experiment_name)
        


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
        self.transcription_dir = os.path.join(output_dir, DATA_DIR, TRANSCRIPTION_DIR)
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
                case[k] = INDETERMINABLE_TEST_CASE

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
            print(audio_path)
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
                self.execution_time_dir, TRANSCRIPTION_DIR, self.getTTS().getName(), asr.getName())
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

            transcriptions[asr.getName()] = preprocess_text(transcription)

            ## add execution time for generating audio
            execution_time += get_execution_time(
                fpath=time_for_recognizing_audio_fpath)    
            

        cases = self.caseDeterminer(text, transcriptions)
        # if sum(cases.values()) == 0 :
        #     print(text)
        #     print(transcriptions["wav2vec2"])
        #     print(cases)
        #     print()
        
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

        def processOneIteration(curr_texts: [Text], processed_texts: [Text], cases):
            start_time = time.time()

            if self.estimator and len(processed_texts) > 0:
                labels = get_labels_from_cases(cases)
                self.trainEstimator(processed_texts, labels)
                curr_texts = self.rank(curr_texts)

            execution_time = 0.
            
            i = 0
            for text in curr_texts :
                # print("================")
                # print(f"{text.getId()}")
                case, exec_time = self.processText(text=text.getText(), filename=f"{text.getId()}")
                cases.append(case)
                execution_time += exec_time
                i += 1
                if execution_time + time.time() - start_time > self.time_budget :
                    # print(f"Number of processed texts {i}")
                    break
            processed_texts.extend(curr_texts[:i])

        
        processed_texts = []
        cases = []
        num_failed_test_cases = []
        num_failed_test_cases_per_asr = {}
        num_processed_texts = []
        for asr in self.asrs:
            num_failed_test_cases_per_asr[asr.getName()] = []
        
        for i in range(self.num_iteration): 
            curr_texts = texts[i*self.text_batch_size:(i+1)*self.text_batch_size]
            processOneIteration(curr_texts, processed_texts, cases)
            num_failed_test_cases.append(calculate_cases(cases, mode=FAILED_TEST_CASE))
            for asr in self.asrs :
                num_failed_test_cases_per_asr[asr.getName()].append(calculate_cases_per_asr(
                    cases, mode=FAILED_TEST_CASE, asr_name=asr.getName()))
            num_processed_texts.append(len(processed_texts))

        data = {}
        data["number_of_failed_test_cases_all"] = num_failed_test_cases
        data["number_of_failed_test_cases_per_asr"] = num_failed_test_cases_per_asr
        data["number_of_processed_texts"] = num_processed_texts
        with open(self.outputfile_failed_test_case + ".json", 'w') as outfile:
            json.dump(data, outfile, indent=2, sort_keys=True)

        # print(len(processed_texts))
        # print(num_failed_test_cases[-1])
        # np.save(self.outputfile_failed_test_case, num_failed_test_cases)

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
        
def calculate_cases(cases, mode:str):
    count = 0
    for c in cases :
        for _, v in c.items() :
            if v == mode :
                count += 1
    return count


def calculate_cases_per_asr(cases, mode:str, asr_name:str):
    count = 0
    for c in cases:
        for k, v in c.items():
            if k == asr_name and v == mode:
                count += 1
    return count


def get_labels_from_cases(cases) :
    def determine_label(case) :
        if INDETERMINABLE_TEST_CASE in case.values() :
            return INDETERMINABLE_TEST_CASE
        if FAILED_TEST_CASE :
            return FAILED_TEST_CASE
        return SUCCESSFUL_TEST_CASE

    labels = []
    for case in cases :
        label = determine_label(case)
        labels.append(label)
    
    return labels