from tts.rv import ResponsiveVoice
from tts.google import Google
from tts.espeak import Espeak
from tts.festival import Festival

from asr.deepspeech import DeepSpeech
from asr.deepspeech2 import DeepSpeech2
from asr.wav2letter import Wav2Letter
from asr.wit import Wit
from asr.wav2vec2 import Wav2Vec2

tts_pool = [ResponsiveVoice(), Google(), Espeak(), Festival()]
asr_pool = [Wav2Vec2(), DeepSpeech(), DeepSpeech2(), Wav2Letter(), Wit()]
