import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import os, sys
from crossasr import CrossASR
import json
import utils

if __name__ == "__main__":

    config = utils.readJson(sys.argv[1]) # read json configuration file

    tts = utils.getTTS(config["tts"])
    asrs = utils.getASRS(config["asrs"])
    estimator = utils.getEstimator(config["estimator"])

    crossasr = CrossASR(tts=tts, asrs=asrs, estimator=estimator, **utils.parseConfig(config))

    corpus_fpath = os.path.join(config["output_dir"], config["corpus_fpath"])
    texts = utils.readCorpus(corpus_fpath=corpus_fpath)
    crossasr.processCorpus(texts=texts)
    crossasr.printStatistic()


    
