from crossasr.asr import ASR
import utils

class DeepSpeech(ASR):
    def __init__(self, name="deepspeech"):
        ASR.__init__(self, name=name)

    def recognizeAudio(self, audio_fpath: str) -> str:
        transcription = utils.deepspeechRecognizeAudio(audio_fpath)
        return transcription

