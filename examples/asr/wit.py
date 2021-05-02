import os
from crossasr.asr import ASR
from wit import Wit as WitAPI

WIT_ACCESS_TOKEN = os.getenv("WIT_ACCESS_TOKEN")
wit_client = WitAPI(WIT_ACCESS_TOKEN)


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
            except Exception:
                # print("Could not request results from Wit.ai service; {0}".format(e))
                transcription = ""

        self.setTranscription(transcription)
        # print(f"Wit transcription: {transcription}")
        return transcription
