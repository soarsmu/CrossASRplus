from crossasr.asr import ASR

import utils

class Wit(ASR):
    def __init__(self):
        ASR.__init__(self, name="wit")

    def recognizeAudio(self, audio_fpath: str) -> str:
        transcription =  utils.witRecognizeAudio(audio_fpath)
        return transcription
