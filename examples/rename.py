import os 

for i in reversed(range(885, 20000)):
    folder = "europarl-seed2021/execution_time/transcription/google/deepspeech2/"
    cmd = f"mv {folder}{i}.txt {folder}{i+1}.txt"
    os.system(cmd)

for i in reversed(range(885, 20000)):
    folder = "europarl-seed2021/execution_time/transcription/google/wit/"
    cmd = f"mv {folder}{i}.txt {folder}{i+1}.txt"
    os.system(cmd)

for i in reversed(range(885, 20000)):
    folder = "europarl-seed2021/execution_time/transcription/google/wav2letter/"
    cmd = f"mv {folder}{i}.txt {folder}{i+1}.txt"
    os.system(cmd)

for i in reversed(range(885, 20000)):
    folder = "europarl-seed2021/execution_time/transcription/google/wav2vec2/"
    cmd = f"mv {folder}{i}.txt {folder}{i+1}.txt"
    os.system(cmd)
