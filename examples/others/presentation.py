

class TTS:

    def __init__(self, name):
        self.name = name

    def generateAudio(self, text: str, audio_fpath: str):
        """
        Generate audio from text. Save the audio at audio_fpath. 
        This is an abstract function that needs to be implemented by the child class

        :param text: input text
        :param audio_fpath: location to save the audio
        """
        raise NotImplementedError()


class ResponsiveVoice(TTS):

    def __init__(self):
        TTS.__init__(self, name="rv")

    def generateAudio(self, text: str, audio_fpath: str):
        tempfile = audio_fpath.split(".")[0] + "-temp.mp3"
        cmd = f"rvtts --voice english_us_male --text {text} -o {tempfile}"
        os.system(cmd)
        setting = " -acodec pcm_s16le -ac 1 -ar 16000 "
        os.system(f"ffmpeg -i {tempfile} {setting} {audio_fpath} -y")


class ASR:
    
    def __init__(self, name):
        self.name = name

    def recognizeAudio(self, audio_fpath: str) -> str:
        """
        Recognize an audio file. Return the transcription. 
        This is an abstract function that needs to be implemented by the child class

        :param audio_fpath: location to the audio file path 
        :return transcription: transcription from the input audio
        """
        raise NotImplementedError()




class Wit(ASR):

    def __init__(self):
        ASR.__init__(self, name="wit")
        WIT_ACCESS_TOKEN = os.getenv("WIT_ACCESS_TOKEN")
        self.wit_client = WitAPI(WIT_ACCESS_TOKEN)

    def recognizeAudio(self, audio_fpath: str) -> str:
        with open(audio_fpath, 'rb') as audio:
            # API request to Wit
            response = wit_client.speech(
                audio_file=audio, 
                headers={'Content-Type': 'audio/wav'})
            # get transcription
            if response != None:
                transcription = str(response["text"])
        return transcription


class Estimator:
    
    def __init__(self, name: str):
        self.name = name

    def fit(self, X: [str], y: [int]):
        """
        Train the estimator

        :param X: a list of sentences 
        :param y: a list cases 
        """
        raise NotImplementedError()

    def predict(self, X: [str]):
        """
        Predict using the trained estimator

        :param X: a list of sentences
        :return failed_probability: a list of probabilities
        """
        raise NotImplementedError()
