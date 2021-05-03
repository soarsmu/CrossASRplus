import os

# for i in range(1, 20000) :
for i in range(1, 3) :
    # folder = "output/europarl-seed2021/data/transcription/rv/wav2vec2/"
    folder = ""
    extension = ".txt"
    cmd = f"mv {folder}{i}{extension} {folder}{i-1}{extension}"
    os.system(cmd)
