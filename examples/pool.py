from tts.google import Google
from tts.espeak import Espeak
from tts.festival import Festival
from tts.rv import ResponsiveVoice

from asr.deepspeech import DeepSpeech
from asr.deepspeech2 import DeepSpeech2
from asr.wav2letter import Wav2Letter
from asr.wav2vec2 import Wav2Vec2
from asr.wit import Wit

tts_pool = [Google(), Espeak(), Festival(), ResponsiveVoice()]
asr_pool = [Wav2Vec2(), DeepSpeech(), DeepSpeech2(), Wav2Letter(), Wit()]
