from crossasr.asr import ASR
import utils

class Wav2Letter(ASR):
    def __init__(self, name="wav2letter"):
        ASR.__init__(self, name=name)

    def recognizeAudio(self, audio_fpath: str) -> str:
        transcription = utils.wav2letterRecognizeAudio(audio_fpath)
        return transcription



