from crossasr.asr import ASR
import subprocess

class Wav2Letter(ASR):
    def __init__(self):
        ASR.__init__(self, name="wav2letter")

    def recognizeAudio(self, audio_path: str) -> str:
        cmd = "docker exec -it wav2letter sh -c \"cat /root/host/" + audio_path + \
            " | /root/wav2letter/build/inference/inference/examples/simple_streaming_asr_example --input_files_base_path /root/host/models/wav2letter/\""

        proc = subprocess.Popen([cmd],
                                stdout=subprocess.PIPE, shell=True)
        (out, _) = proc.communicate()

        # print(out)
        transcription = self.concatWav2letterTranscription(out)

        # print(f"Wav2letter transcription: {transcription}")
        self.setTranscription(transcription)
        return transcription

    def concatWav2letterTranscription(self, out):
        lines = out.splitlines()[21:-2]
        transcription = ""

        for line in lines:
            line = line.decode()
            part = line.split(",")[-1]
            if part != "":
                transcription += part

        transcription = transcription[:-1]

        return transcription


