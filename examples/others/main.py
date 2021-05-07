import os
import numpy as np
import json

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

from crossasr.utils import read_json, preprocess_text, read_corpus
from crossasr import Text, CrossASR

from utils import set_seed, create_tts_by_name, create_asr_by_name, create_huggingface_estimator_by_name


def test():

    json_config_path = "config.json"
    config = read_json(json_config_path)

    set_seed(config["seed"])

    tts = create_tts_by_name(config["tts"])
    asrs = []
    for asr_name in config["asrs"]:
        asrs.append(create_asr_by_name(asr_name))

    crossasr = CrossASR(tts=tts, asrs=asrs, output_dir=config["output_dir"])

    text = "hello world!"
    text = preprocess_text(text)
    filename = "hello_world"
    crossasr.processText(text=text, filename=filename)

def test_corpus():

    json_config_path = "config.json"
    config = read_json(json_config_path)

    set_seed(config["seed"])

    tts = create_tts_by_name(config["tts"])
    asrs = []
    for asr_name in config["asrs"]:
        asrs.append(create_asr_by_name(asr_name))

    kwargs = {
        "recompute" : config["recompute"],
        "time_budget" : config["time_budget"],
        "num_iteration" : config["num_iteration"],
        "text_batch_size" : config["text_batch_size"],
        "max_num_retry": config["max_num_retry"],
        "seed":  config["seed"]
    }

    if config["estimator"] :
        if config["estimator_type"] == "huggingface":
            kwargs["estimator"] = create_huggingface_estimator_by_name(str(config["estimator"]))

    for tbs in range(6400, 8001, 400):
        # for estimator_name in ["albert-base-v2", "facebook/bart-base", "bert-base-cased", "bert-base-uncased", "distilbert-base-uncased", "xlnet-base-cased", "roberta-base"]:
        # for estimator_name in ["valhalla/distilbart-mnli-12-1", "albert-base-v2", "facebook/bart-base", "bert-base-cased", "bert-base-uncased", "distilbert-base-uncased"]:
        # for estimator_name in ["xlnet-base-cased", "roberta-base"]:
        for estimator_name in ["facebook/bart-base"]:
            kwargs["text_batch_size"] = tbs
            kwargs["estimator"] = create_huggingface_estimator_by_name(estimator_name)

    # for estimator_name in ["albert-base-v2", "facebook/bart-base", "bert-base-cased", "bert-base-uncased", "distilbert-base-uncased", "xlnet-base-cased", "roberta-base"]:
    # for estimator_name in ["valhalla/distilbart-mnli-12-1", "albert-base-v2", "facebook/bart-base", "bert-base-cased", "bert-base-uncased", "distilbert-base-uncased"]:
    # for estimator_name in ["xlnet-base-cased", "roberta-base"]:
    # for estimator_name in ["facebook/bart-base"]:
    #     kwargs["estimator"] = create_huggingface_estimator_by_name(estimator_name)

            crossasr = CrossASR(tts=tts, asrs=asrs, output_dir=config["output_dir"], **kwargs)
            corpus_fpath = os.path.join(config["output_dir"], config["corpus_fpath"])
            file = open(corpus_fpath)
            corpus = file.readlines()
            texts = []
            i = 1
            for text in corpus:
                texts.append(Text(i, text[:-1]))
                i += 1
            # texts = read_corpus(corpus_fpath=corpus_fpath)
            crossasr.processCorpus(texts=texts)


if __name__ == "__main__":
    # test()
    test_corpus()
