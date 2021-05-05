from crossasr.asr import ASR
import subprocess

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

        # print("DeepSpeech2 transcription: %s" % transcription)
        return transcription