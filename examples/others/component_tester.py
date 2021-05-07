import os 
from crossasr.utils import preprocess_text
from utils import create_tts_by_name
from utils import create_asr_by_name
from utils import create_huggingface_estimator_by_name

def create_filename_from_text(s) :
    return "_".join(s.split())


def tts_test():
    audio_dir = "output/data/audio/"
    
    text = "hello world!"
    text = preprocess_text(text)
    filename = "hello_world"

    # ttses = ["google", "rv", "espeak", "festival"]
    ttses = ["google"]

    for tts_name in ttses :
        tts = create_tts_by_name(tts_name)
        tts.generateAudio(text=text, audio_dir=audio_dir, filename=filename)

def asr_test():
    audio_dir = "output/data/audio/"
    transcription_dir = "output/data/transcription/"

    tts_name = "google"
    filename = "hello_world"

    audio_fpath = os.path.join(audio_dir, tts_name, filename + ".wav")
    transcription_dir = os.path.join(transcription_dir, tts_name)

    for asr_name in ["deepspeech", "deepspeech2", "wav2letter", "wit", "wav2vec2"] :
        asr = create_asr_by_name(asr_name)
        asr.recognizeAudio(audio_fpath=audio_fpath)
        asr.saveTranscription(transcription_dir=transcription_dir, filename=filename)

if __name__ == "__main__":
    # tts_test()
    asr_test()
