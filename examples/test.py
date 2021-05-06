import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import os
from crossasr import CrossASR
import json
import utils

if __name__ == "__main__":

    config = utils.readJson("config-demo.json")

    tts = utils.getTTS(config["tts"])
    asrs = utils.getASRS(config["asrs"])

    crossasr = CrossASR(tts=tts, asrs=asrs, **utils.parseConfig(config))

    corpus_fpath = os.path.join(config["output_dir"], config["corpus_fpath"])
    texts = utils.readCorpus(corpus_fpath=corpus_fpath)
    crossasr.processCorpus(texts=texts)
    crossasr.printStatistic()


    
