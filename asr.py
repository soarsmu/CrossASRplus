import os
import sys
import subprocess

from utils import make_dir

from wit import Wit as WitAPI

WIT_ACCESS_TOKEN = os.getenv("WIT_ACCESS_TOKEN")
wit_client = WitAPI(WIT_ACCESS_TOKEN)


class ASR:
    def __init__(self, name):
        self.name = name

    def recognizeAudio(audio_path: str) -> str:
        # abstract function need to be implemented by the child class
        raise NotImplementedError()


class DeepSpeech(ASR):
    def __init__(self):
        ASR.__init__(self, name="deepspeech")

    def recognizeAudio(self, audio_path: str) -> str:
        cmd = "deepspeech --model models/deepspeech/deepspeech-0.9.3-models.pbmm --scorer models/deepspeech/deepspeech-0.9.3-models.scorer --audio " + audio_path

        proc = subprocess.Popen([cmd], stdout=subprocess.PIPE, shell=True)
        (out, err) = proc.communicate()

        transcription = out.decode("utf-8")[:-1]
        print("DeepSpeech transcription: %s" % transcription)

        return transcription


class DeepSpeech2(ASR):
    def __init__(self):
        ASR.__init__(self, name="deepspeech2")

    def recognizeAudio(self, audio_path: str) -> str:
        # audio_path = "/" + audio_path
        cmd = "docker exec -it deepspeech2 curl http://localhost:5000/transcribe?fpath=" + audio_path

        proc = subprocess.Popen([cmd], stdout=subprocess.PIPE, shell=True)
        (out, err) = proc.communicate()

        transcription = out.decode("utf-8").split("\n")[-2]
        transcription = transcription[:-1]

        print("DeepSpeech2 transcription: %s" % transcription)

        return transcription


class Wav2Letter(ASR):
    def __init__(self):
        ASR.__init__(self, name="wav2letter")

    def recognizeAudio(self, audio_path: str) -> str:
        cmd = "docker exec -it wav2letter sh -c \"cat /root/host/" + audio_path + \
            " | /root/wav2letter/build/inference/inference/examples/simple_streaming_asr_example --input_files_base_path /root/host/models/wav2letter/\""

        proc = subprocess.Popen([cmd],
                                stdout=subprocess.PIPE, shell=True)
        (out, err) = proc.communicate()

        transcription = self.concatWav2letterTranscription(out)

        print(f"Wav2letter transcription: {transcription}")

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
        ASR.__init__(self, name="wit")

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
        
        # print(f"Wit transcription: {transcription}")
        return transcription


def test():
    text = "hello world!"
    audio_dir = "data/audio/"
    tts_name = "google"
    filename = "hello_world"

    audio_path = os.path.join(audio_dir, tts_name, filename + ".wav")

    # ds = DeepSpeech()
    # transcription = ds.recognizeAudio(audio_path=audio_path)

    # ds2 = DeepSpeech2()
    # transcription = ds2.recognizeAudio(audio_path=audio_path)
    
    # wit = Wit()
    # transcription = wit.recognizeAudio(audio_path=audio_path)

    w2l = Wav2Letter()
    transcription = w2l.recognizeAudio(audio_path=audio_path)

if __name__ == "__main__":
    test()
