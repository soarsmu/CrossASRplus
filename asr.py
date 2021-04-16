import os
import sys
import subprocess

## constant for TTS
from constant import GOOGLE, RV, ESPEAK, FESTIVAL

## constant for ASR
from constant import DS, DS2, W2L, WIT

from utils import make_dir

from wit import Wit as WitAPI

WIT_ACCESS_TOKEN = os.getenv("WIT_ACCESS_TOKEN")
wit_client = WitAPI(WIT_ACCESS_TOKEN)


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


class DeepSpeech(ASR):
    def __init__(self):
        ASR.__init__(self, name=DS)

    def recognizeAudio(self, audio_path: str) -> str:
        cmd = "deepspeech --model models/deepspeech/deepspeech-0.9.3-models.pbmm --scorer models/deepspeech/deepspeech-0.9.3-models.scorer --audio " + audio_path

        proc = subprocess.Popen([cmd], stdout=subprocess.PIPE, shell=True)
        (out, err) = proc.communicate()

        transcription = out.decode("utf-8")[:-1]
        # print("DeepSpeech transcription: %s" % transcription)
        
        self.setTranscription(transcription)

        return transcription


class DeepSpeech2(ASR):
    def __init__(self):
        ASR.__init__(self, name=DS2)

    def recognizeAudio(self, audio_path: str) -> str:
        # audio_path = "/" + audio_path
        cmd = "docker exec -it deepspeech2 curl http://localhost:5000/transcribe?fpath=" + audio_path

        proc = subprocess.Popen([cmd], stdout=subprocess.PIPE, shell=True)
        (out, err) = proc.communicate()

        transcription = out.decode("utf-8").split("\n")[-2]
        transcription = transcription[:-1]

        # print("DeepSpeech2 transcription: %s" % transcription)
        self.setTranscription(transcription)
        return transcription


class Wav2Letter(ASR):
    def __init__(self):
        ASR.__init__(self, name=W2L)

    def recognizeAudio(self, audio_path: str) -> str:
        cmd = "docker exec -it wav2letter sh -c \"cat /root/host/" + audio_path + \
            " | /root/wav2letter/build/inference/inference/examples/simple_streaming_asr_example --input_files_base_path /root/host/models/wav2letter/\""

        proc = subprocess.Popen([cmd],
                                stdout=subprocess.PIPE, shell=True)
        (out, err) = proc.communicate()

        transcription = self.concatWav2letterTranscription(out)

        # print(f"Wav2letter transcription: {transcription}")
        self.setTranscription(transcription)
        return transcription


    def concatWav2letterTranscription(self, out):
        lines = out.splitlines()[21:-2]
        transcription = ""

        j = 0
        for line in lines:
            line = line.decode()
            part = line.split(",")[-1]
            if part != "":
                transcription += part

        transcription = transcription[:-1]

        return transcription


class Wit(ASR):
    def __init__(self):
        ASR.__init__(self, name=WIT)

    def recognizeAudio(self, audio_path: str) -> str:
        transcription = ""
        with open(audio_path, 'rb') as audio:
            try:
                wit_transcription = wit_client.speech(
                    audio, {'Content-Type': 'audio/wav'})

                if wit_transcription != None:
                    if "text" in wit_transcription:
                        transcription = str(wit_transcription["text"])
                    else:
                        transcription = ""
                else:
                    transcription = ""
            except Exception as e:
                # print("Could not request results from Wit.ai service; {0}".format(e))
                transcription = ""
        
        self.setTranscription(transcription)
        # print(f"Wit transcription: {transcription}")
        return transcription


def create_asr_by_name(name: str):
    return {
        DS: DeepSpeech(),
        DS2: DeepSpeech2(),
        W2L: Wav2Letter(),
        WIT: Wit()
    }

def test():
    audio_dir = "data/audio/"
    transcription_dir = "data/transcription/"

    tts_name = GOOGLE
    filename = "hello_world"

    audio_path = os.path.join(audio_dir, tts_name, filename + ".wav")
    transcription_dir = os.path.join(transcription_dir, tts_name)

    ds = create_asr_by_name(DS)
    ds.recognizeAudio(audio_path=audio_path)
    ds.saveTranscription(transcription_dir=transcription_dir, filename=filename)

    ds2 = create_asr_by_name(DS2)
    transcription = ds2.recognizeAudio(audio_path=audio_path)
    ds2.saveTranscription(transcription_dir=transcription_dir, filename=filename)
    
    wit = create_asr_by_name(WIT)
    transcription = wit.recognizeAudio(audio_path=audio_path)
    wit.saveTranscription(transcription_dir=transcription_dir, filename=filename)

    w2l = create_asr_by_name(W2L)
    w2l.recognizeAudio(audio_path=audio_path)
    w2l.saveTranscription(transcription_dir=transcription_dir, filename=filename)

if __name__ == "__main__":
    test()
