import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import json

from crossasr.utils import preprocess_text, read_json
from utils import set_seed

def generate_europarl_corpus():

    # download full data from https://www.kaggle.com/djonafegnem/europarl-parallel-corpus-19962011
    data = "corpus/europarl-parallel-corpus-19962011/"

    df = pd.DataFrame([""], columns=["English"])
    fpaths = []
    for (_, _, filenames) in os.walk(data):
        print(filenames)
        for f in filenames:
            if ".csv" in f:
                fpath = data + f
                fpaths.append(fpath)
    
    fpaths = sorted(fpaths)
    for fpath in fpaths :
        print(fpath)
        d = pd.read_csv(fpath, delimiter=',')
        df = pd.concat([df["English"], d["English"]])
        df = pd.DataFrame(df, columns=["English"])

    df = df.rename(columns={'English': 'sentence'})

    fpath = "corpus/europarl-full.csv"
    df.to_csv(fpath, index=False)

    return fpath


def get_sample_data(df, n):
    sample = df[:n]

    return pd.DataFrame(sample, columns=["sentence"])


def preprocess_data(df, n):

    clean_df = df["sentence"].apply(preprocess_text)

    # remove empty string
    clean_df = [i for i in clean_df if i and i != ""]

    return clean_df[:n]


if __name__ == "__main__":

    json_config_path = "config.json"
    config = read_json(json_config_path)

    print(config)
    seed = config["seed"]

    set_seed(seed)

    print("start: " + str(datetime.now()))

    output_file = generate_europarl_corpus()

    print("generate corpus: " + str(datetime.now()))

    fpath = output_file

    df = pd.read_csv(fpath, delimiter=',')
    print("read data: " + str(datetime.now()))

    # drop null
    df = df.dropna()

    # drop duplicates
    df = df.drop_duplicates()

    # reset index
    df = df.reset_index(drop=True)

    print("get sample: " + str(datetime.now()))
    # get sample data
    
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    N = 20000
    sample_df = get_sample_data(df, int(2 * N))

    print("preprocess data: " + str(datetime.now()))
    
    # text preprocessing
    data = preprocess_data(sample_df, N)

    print("write data: " + str(datetime.now()))

    # TODO: make the folder first 
    outfile = os.path.join(config["output_dir"], config["corpus_fpath"])

    file = open(outfile, "w+")
    for s in data:
        file.write("%s\n" % s)
    file.close()
