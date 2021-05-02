import os, sys
import re, string
import random
import numpy as np
import json
from normalise import normalise, tokenize_basic


def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def remove_hex(text):
    """
    Example: 
    "\xe3\x80\x90Hello \xe3\x80\x91 World!"
    """
    res = []
    i = 0
    while i < len(text):
        if text[i] == "\\" and i+1 < len(text) and text[i+1] == "x":
            i += 3
            res.append(" ")
        else:
            res.append(text[i])
        i += 1
    # text = text.encode('utf-8')
    # text = text.encode('ascii', 'ignore')
    # text = text.encode('ascii', errors='ignore')
    # text = unicode(text)
    # text = re.sub(r'[^\x00-\x7f]', r'', text)
    # filter(lambda x: x in printable, text)
    return "".join(res)

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_multiple_whitespace(text):
    """
    remove multiple whitespace
    it covers tabs and newlines also
    """
    return re.sub(' +', ' ', text.replace('\n', ' ').replace('\t', ' ')).strip()

def normalize_text(text):
    return " ".join(normalise(text, tokenizer=tokenize_basic, verbose=False))

def substitute_word(text):
    """
    word subsitution to make it consistent
    """
    words = text.split(" ")
    preprocessed = []
    for w in words:
        substitution = ""
        if w == "mister":
            substitution = "mr"
        elif w == "missus":
            substitution = "mrs"
        else:
            substitution = w
        preprocessed.append(substitution)
    return " ".join(preprocessed)

def preprocess_text(text):
    text = remove_hex(text)
    text = remove_punctuation(text)
    try:  
        # NOTE: it seems that the normalisation
        #       process is not deterministic
        text = normalize_text(text)
    except :
        text = ""
    # need to remove punctuation again as normalise sometimes add punctuation
    text = remove_punctuation(text)
    text = text.lower()
    text = substitute_word(text)
    text = remove_multiple_whitespace(text)
    return text

def read_json(config_path: str):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def save_execution_time(fpath:str, execution_time):
    with open(fpath, "w+") as f :
        f.write(f"{execution_time:.4f}")


def get_execution_time(fpath:str) :
    f = open(fpath, "r")
    val = f.readlines()[0]
    f.close()
    return float(val)
