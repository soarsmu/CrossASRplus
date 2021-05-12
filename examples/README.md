# Usage Example of CrossASR++

This documentation contains step-by-step walk through of our tool

1. [Prepare Environment](##1-prepare-environment)
2. [Prepare TTSes](##2-prepare-ttses)
3. [Prepare ASRs](##3-prepare-asrs)
4. [Prepare Failure Estimator](##4-prepare-failure-estimator)
5. [Usage Scenario for Adding TTS, ASR, and Estimator](##5-usage-scenario-for-adding-tts-asr-and-estimator)
6. [Runnning CrossASR++ with the Same Setting with CrossASR](##6-runnning-crossasr++-with-the-same-setting-with-crossasr)
7. [Usage Scenario for Testing a Specific ASR](##7-usage-scenario-for-testing-a-specific-asr)
8. [Usage Scenario for Running Using another Estimator from HuggingFace](##8-usage-scenario-for-running-using-another-estimator-from-huggingface)
9. [Download Experiment Data from CrossASR++](##9-download-experiment-data-from-crossasr++)

## 1. Prepare Environment

### 1.1. Install the Python development environment

```bash
sudo apt update
sudo apt install python3-dev python3-pip python3-venv
```

### 1.2. Create a virtual environment

Create a new virtual environment by choosing a Python interpreter and making a ./env directory to hold it:

```bash
python3 -m venv --system-site-packages ~/./env
```

Activate the virtual environment using a shell-specific command:

```bash
source ~/./env/bin/activate  # sh, bash, or zsh

bash install_requirement.sh
```


### Preparation

Make a folder to save the output

```bash
if [ ! -d "output/" ]
then
    mkdir output/
fi

if [ ! -d "output/audio/" ]
then
    mkdir output/audio/
fi
```

## 2. Prepare TTSes

### 2.1. Google

We use [gTTS](https://pypi.org/project/gTTS/) (Google Text-to-Speech), a Python library and CLI tool to interface with Google Translate text-to-speech API. CrossASRv2 use gTTS 2.2.2

```bash
pip install gTTS
```

#### Trial
```bash
mkdir output/audio/google/
gtts-cli 'hello world google' --output output/audio/google/hello.mp3
ffmpeg -i output/audio/google/hello.mp3  -acodec pcm_s16le -ac 1 -ar 16000 output/audio/google/hello.wav -y
```

### 2.2. ResponsiveVoice

We use [rvTTS](https://pypi.org/project/rvtts/), a cli tool for converting text to mp3 files using ResponsiveVoice's API. CrossASRv2 uses rvtts 1.0.1

```bash
pip install rvtts
```

#### Trial
```bash
mkdir output/audio/rv/
rvtts --voice english_us_male --text "hello responsive voice trial" -o output/audio/rv/hello.mp3
ffmpeg -i output/audio/rv/hello.mp3  -acodec pcm_s16le -ac 1 -ar 16000 output/audio/rv/hello.wav -y
```

### 2.3. Festival

[Festival](http://www.cstr.ed.ac.uk/projects/festival/) is a free TTS written in C++. It is developed by The Centre for Speech Technology Research at the University of Edinburgh. Festival are distributed under an X11-type licence allowing unrestricted commercial and non-commercial use alike. Festival is a command-line program that already installed on Ubuntu 16.04. CrossASRv2 uses Festival 2.5.0

#### Trial
```bash
sudo apt install festival -y
mkdir output/audio/festival/
festival -b "(utt.save.wave (SayText \"hello festival \") \"output/audio/festival/hello.wav\" 'riff)"
```

### 2.4. Espeak

[eSpeak](http://espeak.sourceforge.net/) is a compact open source software speech synthesizer for English and other languages. CrossASRv2 uses Espeak 1.48.03

```bash
sudo apt install espeak -y

mkdir output/audio/espeak/
espeak "hello e speak" --stdout > output/audio/espeak/hello.riff
ffmpeg -i output/audio/espeak/hello.riff  -acodec pcm_s16le -ac 1 -ar 16000 output/audio/espeak/hello.wav -y
```

## 3. Prepare ASRs

### 3.1. Deepspeech

[DeepSpeech](https://github.com/mozilla/DeepSpeech) is an open source Speech-To-Text engine, using a model trained by machine learning techniques based on [Baidu's Deep Speech research paper](https://arxiv.org/abs/1412.5567). **CrossASR++ uses [Deepspeech-0.9.3](https://github.com/mozilla/DeepSpeech/releases/tag/v0.9.3)**

```bash
pip install deepspeech===0.9.3

if [ ! -d "asr_models/" ]
then 
    mkdir asr_models
fi

cd asr_models
mkdir deepspeech
cd deepspeech 
curl -LO https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.pbmm
curl -LO https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.scorer
cd ../../
```

Please follow [this link for more detailed installation](https://github.com/mozilla/DeepSpeech/tree/v0.9.3).

#### Trial
```bash
deepspeech --model asr_models/deepspeech/deepspeech-0.9.3-models.pbmm --scorer asr_models/deepspeech/deepspeech-0.9.3-models.scorer --audio output/audio/google/hello.wav
```

### 3.2. Deepspeech2

[DeepSpeech2](https://github.com/PaddlePaddle/DeepSpeech) is an open-source implementation of end-to-end Automatic Speech Recognition (ASR) engine, based on [Baidu's Deep Speech 2 paper](http://proceedings.mlr.press/v48/amodei16.pdf), with [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) platform.

#### Setup a docker container for Deepspeech2

[Original Source](https://github.com/PaddlePaddle/DeepSpeech#running-in-docker-container)

```bash
cd asr_models/
git clone https://github.com/PaddlePaddle/DeepSpeech.git
cd DeepSpeech
git checkout tags/v1.1
cp ../../asr/deepspeech2_api.py .
cd models/librispeech/
sh download_model.sh
cd ../../../../
cd asr_models/DeepSpeech/models/lm
sh download_lm_en.sh
cd ../../../../
docker pull paddlepaddle/paddle:1.6.2-gpu-cuda10.0-cudnn7

# run this command from examples folder
# please remove --gpus '"device=1"' if you only have one gpu
docker run --name deepspeech2 --rm --gpus '"device=1"' -it -v $(pwd)/asr_models/DeepSpeech:/DeepSpeech -v $(pwd)/output/:/DeepSpeech/output/  paddlepaddle/paddle:1.6.2-gpu-cuda10.0-cudnn7 /bin/bash

apt-get update
apt-get install git -y
cd DeepSpeech
sh setup.sh
apt-get install libsndfile1-dev -y
``` 

**in case you found error when running the `setup.sh`**

Error solution for `ImportError: No module named swig_decoders`
```bash
pip install paddlepaddle-gpu==1.6.2.post107
cd DeepSpeech
pip install soundfile
pip install llvmlite===0.31.0
pip install resampy
pip install python_speech_features

wget http://prdownloads.sourceforge.net/swig/swig-3.0.12.tar.gz
tar xvzf swig-3.0.12.tar.gz
cd swig-3.0.12
apt-get install automake -y 
./autogen.sh
./configure
make
make install

cd ../decoders/swig/
sh setup.sh
cd ../../
```

#### Run Deepspeech2 as an API (inside docker container)
```bash
pip install flask 

# run inside /DeepSpeech folder in the container
CUDA_VISIBLE_DEVICES=0 python deepspeech2_api.py \
    --mean_std_path='models/librispeech/mean_std.npz' \
    --vocab_path='models/librispeech/vocab.txt' \
    --model_path='models/librispeech' \
    --lang_model_path='models/lm/common_crawl_00.prune01111.trie.klm'
```
Then detach from the docker using ctrl+p & ctrl+q after you see `Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)`

#### Run Client from the Terminal (outside docker container)

```bash
# run from examples folder in the host machine (outside docker)
docker exec -it deepspeech2 curl http://localhost:5000/transcribe?fpath=output/audio/google/hello.wav
```

### 3.3. Wav2letter++

[wav2letter++](https://github.com/facebookresearch/wav2letter) is a highly efficient end-to-end automatic speech recognition (ASR) toolkit written entirely in C++ by Facebook Research, leveraging ArrayFire and flashlight.

Please find the lastest image of [wav2letter's docker](https://hub.docker.com/r/wav2letter/wav2letter/tags).

```bash
cd asr_models/
mkdir wav2letter
cd wav2letter

for f in acoustic_model.bin tds_streaming.arch decoder_options.json feature_extractor.bin language_model.bin lexicon.txt tokens.txt ; do wget http://dl.fbaipublicfiles.com/wav2letter/inference/examples/model/${f} ; done

ls -sh
cd ../../
```

#### Run docker inference API
```bash
# run from examples folder
docker run --name wav2letter -it --rm -v $(pwd)/output/:/root/host/output/ -v $(pwd)/asr_models/:/root/host/models/ --ipc=host -a stdin -a stdout -a stderr wav2letter/wav2letter:inference-latest
```
Then detach from the docker using ctrl+p & ctrl+q 

#### Run Client from the Terminal

```bash
docker exec -it wav2letter sh -c "cat /root/host/output/audio/google/hello.wav | /root/wav2letter/build/inference/inference/examples/simple_streaming_asr_example --input_files_base_path /root/host/models/wav2letter/"
```

Detail of [wav2letter++ installation](https://github.com/facebookresearch/wav2letter/wiki#Installation) and [wav2letter++ inference](https://github.com/facebookresearch/wav2letter/wiki/Inference-Run-Examples)

### 3.4. Wit

[Wit](https://wit.ai/) gives an API interface for ASR. We use [pywit](https://github.com/wit-ai/pywit), the Python SDK for Wit. You need to create an WIT account to get access token. CrossASRv2 uses Wit 6.0.0

#### install pywit
```bash
pip install wit
```

#### Setup Wit access token
```bash
export WIT_ACCESS_TOKEN=<your Wit access token>
```

#### Check using HTTP API
```bash
curl -XPOST 'https://api.wit.ai/speech?' \
    -i -L \
    -H "Authorization: Bearer $WIT_ACCESS_TOKEN" \
    -H "Content-Type: audio/wav" \
    --data-binary "@output/audio/google/hello.wav"
```

**Success Response**
```bash
HTTP/1.1 100 Continue
Date: Fri, 11 Sep 2020 05:55:51 GMT

HTTP/1.1 200 OK
Content-Type: application/json
Date: Fri, 11 Sep 2020 05:55:52 GMT
Connection: keep-alive
Content-Length: 85

{
  "entities": {},
  "intents": [],
  "text": "hello world google",
  "traits": {}
}
```

## 4. Prepare Failure Estimator

### 4.1. HuggingFace Estimator

HuggingFace provides thousands pretrained language model for NLP tasks. HuggingFace hosts SOTA language models. People continuesly add their pretrained models to HuggingFace. Thus it help us in developing good estimator fast and catch up with the latest SOTA models. HuggingFace library is available at `transformers` PyPi.

```bash
pip install torch
pip install transformers
```


## 5. Usage Scenario for Adding TTS, ASR, and Estimator


### 5.1. Creating ResponsiveVoice TTS
User can add a new TTS by defining a class derived from `TTS` interface. The steps for 

**Defining a class derived from TTS interface**
```python
class ResponsiveVoice(TTS):

    def __init__(self, name="rv"):
        TTS.__init__(self, name=name)

    def generateAudio(self, text: str, audio_fpath: str) -> str:
        utils.rvGenerateAudio(text, audioa_fpath)
```

**Add the Class to the Pool**

Add the new TTS to the TTS pool in the `examples/pool.py`
```python
tts_pool = [Google(), Espeak(), Festival(), ResponsiveVoice()]
```
The function `getTTS` in the `examples/utils.py` will be called from the CrossASR++ main program 


### 5.1. Creating Wit ASR

**Defining a class derived from ASR interface**

```python
class Wit(ASR):
    def __init__(self, name="wit"):
        ASR.__init__(self, name=name)
        
    def recognizeAudio(self, audio_fpath: str) -> str:
        transcription = utils.witRecognizeAudio(audio_fpath)
        return transcription
```

**Add the Class to the Pool**

Add the new ASR to the ASR pool in the `examples/pool.py`
```python
asr_pool = [Wav2Vec2(), DeepSpeech(), DeepSpeech2(), Wav2Letter(), Wit()]
```
The function `getASR` in the `examples/utils.py` will be called from the CrossASR++ main program

### 5.1. Creating ALBERT Estimator

**Defining Class**
```python
class HuggingFaceTransformer(Estimator):
    def __init__(self, name, max_sequence_length=128):
        Estimator.__init__(self, name=name)

        ## init boiler plate
        self.model = AutoModelForSequenceClassification.from_pretrained(
            name, num_labels=NUM_LABELS) ## init model
        self.tokenizer = AutoTokenizer.from_pretrained(name) ## init tokenizer
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.max_sequence_length = max_sequence_length

    def fit(self, X: [str], y: [int]):

        self.model.to(self.device)
        self.model.train()

        train_texts = X
        train_labels = y

        train_encodings = self.tokenizer(
            train_texts, truncation=True, padding=True, max_length=self.max_sequence_length)
        train_dataset = HuggingFaceDataset(train_encodings, train_labels)

        training_args = TrainingArguments(
            output_dir='./results',          # output directory
            num_train_epochs=1,              # total number of training epochs
            per_device_train_batch_size=8,   # batch size per device during training
            per_device_eval_batch_size=64,   # batch size for evaluation
            learning_rate=5e-05,
            weight_decay=0.01,               # strength of weight decay
            logging_dir='./logs',            # directory for storing logs
            logging_steps=10,
        )

        trainer = Trainer(
            model=self.model,               # the instantiated 🤗 Transformers model to be trained
            args=training_args,             # training arguments, defined above
            train_dataset=train_dataset     # training dataset
        )

        trainer.train()

    def predict(self, X: [str]):
        self.model.eval()
        test_texts = X
        test_labels = [0] * len(X)
        test_encodings = self.tokenizer(
            test_texts, truncation=True, padding=True, max_length=self.max_sequence_length)
        test_dataset = HuggingFaceDataset(test_encodings, test_labels)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=16, shuffle=False)

        res = []
        for batch in test_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            outputs = self.model(
                input_ids, attention_mask=attention_mask, labels=labels)
            preds = softmax(outputs.logits.detach().cpu().numpy(), axis=1)
            res.extend(preds)
        res = np.array(res)
        failed_probability = res[:, FAILED_TEST_CASE]
        
        return failed_probability

```

**Add the Class to the Pool**

Override `getEstimator` function in the `examples/utils.py`
```python
def getEstimator(name: str):
    return HuggingFaceTransformer(name=name)
```
The function `getEstimator` will be called from the CrossASR++ main program. We can easily use another estimator from HuggingFace by replacing `albert-base-v2` into another model, e.g. `bert-base-uncased`. 


## 6. Runnning CrossASR++ with the Same Setting with CrossASR

The available TTS are ResponsiveVoice, Google, Espeak, and Festival. The available ASR are DeepSpeech, DeepSpeech2, Wav2Letter++, and Wit. The estimator used in CrossASR is ALBERT-base-v2.

To run CrossASR++, we need to specify the TTS, ASRs, and estimator used in `config.json`. This is an example for the same setting used in CrossASR.
```json
{
    ...
    "tts" : "google",
    "asrs" : ["deepspeech", "deepspeech2", "wav2letter", "wit"],
    "estimator": "albert-base-v2",
    ...
}
```

To run the program, execute this command from the `examples` folder
```bash
python run_crossasr.py config.json
``` 

This program will generate test cases for all ASRs in the folder located in the `output_dir` specified in the `config.json`. In the `output_dir` there will be `data` folder to save the audio files and their transcriptions, `execution_time` folder to save the execution time for generating audio files and recognizing them, `cases` folder to save the cases status, i.e. failed test cases, succesfull test cases, and indeterminable test cases.

## 7. Usage Scenario for Running using a Text

This program will do cross-referencing using a piece of text. This function is helpful to check the functionaility of each component in CrossASR++.

Run the program
```bash
python cross_reference.py config_text.json
``` 

## 7. Usage Scenario for Testing a Specific ASR

Recently, researchers from AI community published a new transformer based ASR, Wav2Vec2. Here the step to add Wav2Vec2 into our pipeline

Installing libraries needed Wav2Vec2
```bash
pip install torch
pip install transformers
```

Defining Class
```python
class Wav2Vec2(ASR):
    def __init__(self):
        ASR.__init__(self, name="wav2vec2")
        self.tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-large-960h")
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
            
    def recognizeAudio(self, audio_fpath: str) -> str:
        audio_input, _ = sf.read(audio_fpath) # load audio
        input_values = tokenizer(audio_input, return_tensors="pt").input_values
        logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.tokenizer.batch_decode(predicted_ids)[0]
        self.setTranscription(transcription)
        return transcription
```

Add class to the ASR pool in the `examples/utils.py`
```python
def get_asr_pool() :
    asrs = []
    # add all of possibles ASR here
    asrs.append(Wav2Vec2())

    return asrs
```

Add the ASR unique name to the `config.json`. Set the `target_asr` with the target ASR name. The data saved are are the ground truth text and its audio where Wav2Vec2 wrongly transcribe the text. The data is saved at `failed_test_cases` dir inside `output_dir` specified at `config.json`. The data may be used to retrain the ASR model.
```json
{
    ...
    "asrs" : ["deepspeech", "deepspeech2", "wav2letter", "wit", "wav2vec2"],
    ...
    "target_asr": "wav2vec2",
    ...
}
```

Run the program
```bash
python test_asr.py config_corpus.json
``` 

## 8. Usage Scenario for Running Using another Estimator from HuggingFace

Estimators from HuggingFace are customizable easily. We only need to change `estimator` in the `config.json` with another thousands models available at https://huggingface.co/models.
```json
{
    ...
    "estimator": "<any HuggingFace model>",
    ...
}
```

## 9. Download Experiment Data from CrossASR++

We provide our experiment data to support the community doing experiment in CrossASR++ faster. Please download the data at this [GDrive Link](https://drive.google.com/drive/folders/1_6YYDaZ03EZFUf-k6EXU-DkLNR_WMmjj?usp=sharing)

