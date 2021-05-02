import gc
import torch
import soundfile as sf
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
from crossasr.asr import ASR

class Wav2Vec2(ASR):
    def __init__(self):
        ASR.__init__(self, name="wav2vec2")
        self.tokenizer = None
        self.model = None

    def recognizeAudio(self, audio_path: str) -> str:

        # load pretrained model
        if not self.tokenizer:
            self.tokenizer = Wav2Vec2Tokenizer.from_pretrained(
                "facebook/wav2vec2-base-960h")
        if not self.model:
            self.model = Wav2Vec2ForCTC.from_pretrained(
                "facebook/wav2vec2-base-960h")
        # self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # self.model.to(self.device)

        # load audio
        audio_input, _ = sf.read(audio_path)

        # transcribe
        input_values = self.tokenizer(
            audio_input, return_tensors="pt").input_values
        # input_values = input_values.to(self.device)

        logits = self.model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.tokenizer.batch_decode(predicted_ids)[0]
        # transcription = transcription.lower()
        self.setTranscription(transcription)

        del audio_input, input_values, logits, predicted_ids
        torch.cuda.empty_cache()
        gc.collect()

        return transcription
