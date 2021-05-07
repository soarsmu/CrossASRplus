from crossasr.asr import ASR
import utils

class DeepSpeech2(ASR):
    def __init__(self, name="deepspeech2"):
        ASR.__init__(self, name=name)

    def recognizeAudio(self, audio_fpath: str) -> str:
        transcription = utils.deepspeech2RecognizeAudio(audio_fpath)
        return transcription
