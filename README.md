# CrossASR++ 

## Installation

**Installation with pip**

CrossASR++ is designed and tested to run with Python 3. CrossASR++ can be installed from the PyPi repository using pip:

```pip install crossasr```

**Manual installation**

The most recent version of CrossASR++ can be downloaded or cloned from this repository:

```git clone https://github.com/mhilmiasyrofi/CrossASRv2```

Install CrossASR++ with the following command from the project folder CrossASRv2, using pip:

```pip install .```

## Usage

### 1. Adding a TTS

To add a TTS, you need to create a class inherited from `TTS` interface. You must override the function for generating an audio.

```python
class NewTTS(TTS):
  def __init__(self):
    TTS.__init__(self, name="new_TTS")
  def generateAudio(self, text, audio\_dir, name):
    # override this function for generating an audio
```

### 2. Adding an ASR

To add an ASR, you need to create a class inherited from `ASR` interface. You must override the function for recognizing an audio.

```python
class NewASR(ASR):
  def __init__(self):
    ASR.__init__(self, name="new_ASR")
  def recognizeAudio(self, audio):
    # override this function for recognizing an audio
```

### 3. Adding an Estimator

To add an Estimator, you need to create a class inherited from `Estimator` interface. You must override the function for training and predicting.
```python
class NewEstimator(Estimator):
  def __init__(self):
    Estimator.__init__(self,name="new_estimator")
  def fit(self, X, y):
    # override this function for training the estimator
  def predict(self, X):
    # override this function for predicting using the estimator
```

For each class (TTS, ASR, and Estimator), you **must** define the name of the class name in the constructor with a unique name. This name will be useful for saving the output data also, including audios and transcriptions.

### Folder Structure to Save Data

CrossASR++ automatically save the audio files and their transcriptions (along with their execution times). 

## Real-world Examples

We provide real examples for cross-referencing ASRs in folder [`examples`](examples/README.md). It provides clear instruction on how to create a real TTS, ASR, and Estimator and how to test a specific ASR.