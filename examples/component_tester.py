# def test():
#     text = "hello world!"
#     audio_dir = "data/audio/"

#     text = preprocess_text(text)
#     filename = create_filename_from_text(text)

#     ttses = [GOOGLE, RV, ESPEAK, FESTIVAL]

#     for tts_name in ttses :
#         tts = create_tts_by_name(tts_name)
#         tts.generateAudio(text=text, audio_dir=audio_dir, filename=filename)

# if __name__ == "__main__":
#     test()

# def test():
#     audio_dir = "data/audio/"
#     transcription_dir = "data/transcription/"

#     tts_name = GOOGLE
#     filename = "hello_world"

#     audio_path = os.path.join(audio_dir, tts_name, filename + ".wav")
#     transcription_dir = os.path.join(transcription_dir, tts_name)

#     # ds = create_asr_by_name(DS)
#     # ds.recognizeAudio(audio_path=audio_path)
#     # ds.saveTranscription(transcription_dir=transcription_dir, filename=filename)

#     # ds2 = create_asr_by_name(DS2)
#     # ds2.recognizeAudio(audio_path=audio_path)
#     # ds2.saveTranscription(transcription_dir=transcription_dir, filename=filename)

#     # wit = create_asr_by_name(WIT)
#     # wit.recognizeAudio(audio_path=audio_path)
#     # wit.saveTranscription(transcription_dir=transcription_dir, filename=filename)

#     w2l = create_asr_by_name(W2L)
#     w2l.recognizeAudio(audio_path=audio_path)
#     w2l.saveTranscription(transcription_dir=transcription_dir, filename=filename)

# if __name__ == "__main__":
#     test()
