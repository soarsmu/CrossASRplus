from crossasr.asr import ASR
import subprocess

class DeepSpeech(ASR):
    def __init__(self):
        ASR.__init__(self, name="deepspeech")

    def recognizeAudio(self, audio_path: str) -> str:
        cmd = "deepspeech --model asr_models/deepspeech/deepspeech-0.9.3-models.pbmm --scorer asr_models/deepspeech/deepspeech-0.9.3-models.scorer --audio " + audio_path

        proc = subprocess.Popen([cmd], stdout=subprocess.PIPE, shell=True)
        (out, _) = proc.communicate()

        transcription = out.decode("utf-8")[:-1]
        # print("DeepSpeech transcription: %s" % transcription)

        self.setTranscription(transcription)

        return transcription

