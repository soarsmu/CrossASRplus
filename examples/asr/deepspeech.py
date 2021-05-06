from crossasr.asr import ASR
import utils

class DeepSpeech(ASR):
    def __init__(self):
        ASR.__init__(self, name="deepspeech")

    def recognizeAudio(self, audio_fpath: str) -> str:
        transcription = utils.deepspeechRecognizeAudio(audio_fpath)
        return transcription

