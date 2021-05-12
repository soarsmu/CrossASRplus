
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
