# CrossASR++ 

Developers need to perform adequate testing to ensure the quality of Automatic Speech Recognition (ASR) systems. However, manually collecting required test cases is tedious and time-consuming. Our recent work proposes, namely [CrossASR](https://github.com/soarsmu/CrossASR), a differential testing method for ASR systems. This method first utilizes Text-to-Speech (TTS) to generate audios from texts automatically and then feed these audios into different ASR systems for cross-referencing to uncover failed test cases. It also leverages a failure estimator to find test cases more efficiently. Such a method is inherently self-improvable: the performance can increase by leveraging more advanced TTS and ASR systems. 

So in this accompanying tool, we devote more engineering and propose **CrossASR++, an easy-to-use ASR testing tool that can be conveniently extended to incorporate different TTS and ASR systems and failure estimators**. We also make CrossASR++ chunk texts dynamically and enable the estimator to work in a more efficient and flexible way. We demonstrate that the new features can help CrossASR++ discover more failed test cases.

Please check our Tool Demo Video at [https://www.youtube.com/watch?v=ddRk-f0QV-g](https://www.youtube.com/watch?v=ddRk-f0QV-g)

PDF preprint is [available](https://arxiv.org/pdf/2105.14881.pdf)

## Installation

### 1. PyPI installation

CrossASR++ is designed and tested to run with Python 3. CrossASR++ can be installed from the PyPi repository using this command

```pip install crossasr```

### 2. Manual installation

The most recent version of CrossASR++ can be cloned from this repository using this command

```git clone https://github.com/soarsmu/CrossASRplus```

Install CrossASR++ with the following command from the project folder CrossASRplus, using this command

```pip install .```

## Extensibility

We devote more engineering effort to enhancing the extensibility of CrossASR++. We reimplement all necessary processes in CrossASR and pay attention to the extensibility of the tool. The extensibility is mainly enhanced by modeling the TTS, ASR, and failure estimator with several interfaces, i.e. abstract base classes. Users can add a new TTS, a new ASR or a new failure estimator by simply inheriting the base class and implementing necessary methods.

We have 3 base classes, i.e. `ASR`, `TTS`, and `Estimator`. When inheriting from each class, users need to specify a name in the constructor. This name will be associated with a folder for saving the audio files and transcriptions. Thus having a unique name for each class is required. When inheriting `ASR` base class, users must override the `recognizeAudio()` method which takes an audio as input and returns recognized transcription. TTS and failure estimator can be added similarly. In `TTS` base class, the method `generateAudio()` must be overrided by inherited classes. This method converts a piece of text into audio. In `Estimator` base class, methods `fit()` and `predict()` must be overrided by inherited classes. These methods are used for training and predicting, respectively.


### 1. Adding a TTS

To add a TTS, you need to create a class inherited from `TTS` interface. You must override the function for generating an audio.

```python
class TTS:

    def __init__(self, name):
        self.name = name

    def generateAudio(self, text:str, audio_fpath: str):
        """
        Generate audio from text. Save the audio at audio_fpath. 
        This is an abstract function that needs to be implemented by the child class

        :param text: input text
        :param audio_fpath: location to save the audio
        """
        raise NotImplementedError()
```

### 2. Adding an ASR

To add an ASR, you need to create a class inherited from `ASR` interface. You must override the function for recognizing an audio.

```python
class ASR:
    
    def __init__(self, name):
        self.name = name
    
    def recognizeAudio(self, audio_fpath: str) -> str:
        """
        Recognize audio file. Return the transcription
        This is an abstract function that needs to be implemented by the child class

        :param audio_fpath: location to load the audio
        :return transcription: transcription from the audio
        """
        raise NotImplementedError()
```

### 3. Adding an Estimator

To add an Estimator, you need to create a class inherited from `Estimator` interface. You must override the function for training and predicting.
```python
class Estimator:
    def __init__(self, name:str):
        self.name = name

    def fit(self, X:[str], y:[int]):
        raise NotImplementedError()

    def predict(self, X:[str]):
        raise NotImplementedError()

```
### Real-world Examples

To make CrossASR++ a plug-and-play tool, we have incorporated some latest components. The suppported TTSes are [Google Translate’s TTS](https://pypi.org/project/gTTS/), [ResponsiveVoice](https://pypi.org/project/rvtts/), [Festival](https://www.cstr.ed.ac.uk/projects/festival/), and [Espeak](http://espeak.sourceforge.net). The supported ASRs are [DeepSpeech](https://pypi.org/project/deepspeech/), [DeepSpeech2](https://github.com/PaddlePaddle/DeepSpeech), [Wit](https://wit.ai), and [wav2letter++](https://github.com/flashlight/wav2letter). CrossASR++ supports any transformed-based classifier available at [HuggingFace](https://huggingface.co). CrossASR++ can also be easily extended to leverage more advanced tools in the future.

We provide real examples for cross-referencing ASR systems in folder [`examples`](https://github.com/soarsmu/CrossASRplus/tree/main/examples). It provides clear instruction on how to create the suppported TTS, ASR, and Estimator and how to test a specific ASR system.

### Automatically Save Data

CrossASR++ automatically save the audio files and their transcriptions (along with their execution times) to help researchers save their time when developing failure estimators. 


### Please cite our work!

```
@INPROCEEDINGS{Asyrofi2021CrossASRplus,  
    author={M. H. {Asyrofi} and Z. {Yang} and D. {Lo}},  
    booktitle={Proceedings of the 29th ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering (ESEC/FSE '21), August 23--28, 2021, Athens, Greece},
    title={CrossASR++: : A Modular Differential Testing Framework for Automatic Speech Recognition},   
    year={2021},  volume={},  number={},  
    pages={},  
    doi={10.1145/3468264.3473124}}
    
@INPROCEEDINGS{Asyrofi2020CrossASR,  
    author={M. H. {Asyrofi} and F. {Thung} and D. {Lo} and L. {Jiang}},  
    booktitle={2020 IEEE International Conference on Software Maintenance and Evolution (ICSME)},
    title={CrossASR: Efficient Differential Testing of Automatic Speech Recognition via Text-To-Speech},   
    year={2020},  volume={},  number={},  
    pages={640-650},  
    doi={10.1109/ICSME46990.2020.00066}}
    
```


