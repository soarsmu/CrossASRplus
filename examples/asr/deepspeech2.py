from crossasr.asr import ASR
import utils

class DeepSpeech2(ASR):
    def __init__(self):
        ASR.__init__(self, name="deepspeech2")

    def recognizeAudio(self, audio_fpath: str) -> str:
        transcription = utils.deepspeech2RecognizeAudio(audio_fpath)
        return transcription