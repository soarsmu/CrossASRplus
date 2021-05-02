import random
import numpy as np

import torch

from tts.rv import ResponsiveVoice
from tts.google import Google
from tts.espeak import Espeak
from tts.festival import Festival

from asr.deepspeech import DeepSpeech
from asr.deepspeech2 import DeepSpeech2
from asr.wav2letter import Wav2Letter
from asr.wit import Wit
from asr.wav2vec2 import Wav2Vec2

def set_seed(seed: int) :
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def create_tts_by_name(name: str):
    return {
        "google" : Google(),
        "rv" : ResponsiveVoice(),
        "espeak" : Espeak(),
        "festival" : Festival()
    }[name]

# def test():
#     text = "hello world!"
#     audio_dir = "data/audio/"

#     text = preprocess_text(text)
#     filename = create_filename_from_text(text)

#     ttses = [GOOGLE, RV, ESPEAK, FESTIVAL]

#     for tts_name in ttses :
#         tts = create_tts_by_name(tts_name)
#         tts.generateAudio(text=text, audio_dir=audio_dir, filename=filename)

# if __name__ == "__main__":
#     test()


def create_asr_by_name(name: str):
    return {
        "deepspeech" : DeepSpeech(),
        "deepspeech2" : DeepSpeech2(),
        "wav2letter" : Wav2Letter(),
        "wit" : Wit(),
        "wav2vec2": Wav2Vec2()
    }[name]


# def test():
#     audio_dir = "data/audio/"
#     transcription_dir = "data/transcription/"

#     tts_name = GOOGLE
#     filename = "hello_world"

#     audio_path = os.path.join(audio_dir, tts_name, filename + ".wav")
#     transcription_dir = os.path.join(transcription_dir, tts_name)

#     # ds = create_asr_by_name(DS)
#     # ds.recognizeAudio(audio_path=audio_path)
#     # ds.saveTranscription(transcription_dir=transcription_dir, filename=filename)

#     # ds2 = create_asr_by_name(DS2)
#     # ds2.recognizeAudio(audio_path=audio_path)
#     # ds2.saveTranscription(transcription_dir=transcription_dir, filename=filename)

#     # wit = create_asr_by_name(WIT)
#     # wit.recognizeAudio(audio_path=audio_path)
#     # wit.saveTranscription(transcription_dir=transcription_dir, filename=filename)

#     w2l = create_asr_by_name(W2L)
#     w2l.recognizeAudio(audio_path=audio_path)
#     w2l.saveTranscription(transcription_dir=transcription_dir, filename=filename)

# if __name__ == "__main__":
#     test()


# def test():

#     json_config_path = "config.json"
#     config = read_json(json_config_path)

#     set_seed(config["seed"])

#     tts = create_tts_by_name(config["tts"])
#     asrs = []
#     for asr_name in config["asrs"]:
#         asrs.append(create_asr_by_name(asr_name))

#     crossasr = CrossASR(tts=tts, asrs=asrs, output_dir=config["output_dir"])

#     text = "hello world!"
#     text = preprocess_text(text)
#     filename = "hello_world"
#     crossasr.processText(text=text, filename=filename)

# def test_corpus():

#     json_config_path = "config.json"
#     config = read_json(json_config_path)

#     set_seed(config["seed"])

#     tts = create_tts_by_name(config["tts"])
#     asrs = []
#     for asr_name in config["asrs"]:
#         asrs.append(create_asr_by_name(asr_name))

#     kwargs = {
#         "recompute" : bool(config["recompute"]),
#         "time_budget" : int(config["time_budget"]),
#         "num_iteration" : int(config["num_iteration"]),
#         "text_batch_size" : int(config["text_batch_size"]),
#         "max_num_retry": int(config["max_num_retry"])
#     }

#     if config["estimator"] :
#         if config["estimator_type"] == "huggingface":
#             kwargs["estimator"] = create_huggingface_estimator_by_name(str(config["estimator"]))

#     # for tbs in [400] :
#     #     # for estimator_name in ["albert-base-v2", "facebook/bart-base", "bert-base-cased", "bert-base-uncased", "distilbert-base-uncased"]:
#     #     # for estimator_name in ["valhalla/distilbart-mnli-12-1", "albert-base-v2", "facebook/bart-base", "bert-base-cased", "bert-base-uncased", "distilbert-base-uncased"]:
#     #     for estimator_name in ["xlnet-base-cased", "roberta-base", "gpt2"]:
#     #         kwargs["text_batch_size"] = tbs
#     #         kwargs["estimator"] = create_huggingface_estimator_by_name(estimator_name)

#     crossasr = CrossASR(tts=tts, asrs=asrs, output_dir=config["output_dir"], **kwargs)

#     corpus_path = os.path.join(config["output_dir"], constant.CORPUS_PATH)
#     file = open(corpus_path)
#     corpus = file.readlines()
#     texts = []
#     i = 1
#     for text in corpus:
#         texts.append(Text(i, text[:-1]))
#         i += 1
#     crossasr.processCorpus(texts=texts)

# if __name__ == "__main__" :
#     # test()
#     test_corpus()
