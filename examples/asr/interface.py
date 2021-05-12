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
